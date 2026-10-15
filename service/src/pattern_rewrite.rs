use std::fs;
use std::path::Path;
use std::time::Instant;

use tree_sitter::Parser;

use crate::lang::LanguageSpec;
use crate::pattern_match::{collect_files, find_matches_in_tree, parse_pattern, MatchEnv, PatternNode};
use crate::types::{RewriteEdit, RewriteFileResult, RewriteResult};


fn substitute_template(template: &str, env: &MatchEnv) -> String {
    let mut result = String::with_capacity(template.len() * 2);
    let chars: Vec<char> = template.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i] == '$' && i + 1 < len {
            let rest = &chars[i + 1..];

            if rest.starts_with(&['$', '$']) {
                let name_start = 2;
                let name_end = name_start
                    + rest[name_start..]
                        .iter()
                        .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || **c == '_')
                        .count();

                if name_end > name_start {
                    let name: String = rest[name_start..name_end].iter().collect();
                    if let Some(texts) = env.multis.get(&name) {
                        result.push_str(&texts.join(", "));
                        i += 1 + name_end;
                        continue;
                    }
                }

                result.push_str("$$$");
                i += 3;
                continue;
            }

            if rest[0].is_ascii_uppercase() || rest[0] == '_' {
                let name_end = rest
                    .iter()
                    .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || **c == '_')
                    .count();

                if name_end > 0 {
                    let name: String = rest[..name_end].iter().collect();
                    if let Some(text) = env.singles.get(&name) {
                        result.push_str(text);
                        i += 1 + name_end;
                        continue;
                    }
                }
            }

            result.push('$');
            i += 1;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}


fn rewrite_file(
    pattern: &PatternNode,
    template: &str,
    source: &str,
    spec: &LanguageSpec,
) -> Vec<RewriteEdit> {
    let mut parser = Parser::new();
    if parser.set_language(&spec.language).is_err() {
        return Vec::new();
    }

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let source_bytes = source.as_bytes();
    let mut raw_matches = Vec::new();
    find_matches_in_tree(pattern, tree.root_node(), source_bytes, &mut raw_matches);

    let mut edits: Vec<RewriteEdit> = Vec::new();

    for (node, env) in &raw_matches {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let start_pos = node.start_position();
        let end_pos = node.end_position();
        let original = node.utf8_text(source_bytes).unwrap_or("").to_string();
        let replacement = substitute_template(template, env);

        if original == replacement {
            continue;
        }

        let overlaps = edits.iter().any(|e| {
            start_byte < e.byte_end && end_byte > e.byte_start
        });
        if overlaps {
            continue;
        }

        edits.push(RewriteEdit {
            line_start: start_pos.row + 1,
            column_start: start_pos.column + 1,
            line_end: end_pos.row + 1,
            column_end: end_pos.column + 1,
            byte_start: start_byte,
            byte_end: end_byte,
            original,
            replacement,
        });
    }

    edits.sort_by_key(|e| std::cmp::Reverse(e.byte_start));
    edits
}


/// Apply byte-offset edits to source text, producing rewritten output.
///
/// source: &str — Original source code.
/// edits: &[RewriteEdit] — Edits sorted by byte_start descending (back-to-front).
fn apply_edits(source: &str, edits: &[RewriteEdit]) -> String {
    let mut output = source.to_string();
    for edit in edits {
        if edit.byte_start <= output.len() && edit.byte_end <= output.len() {
            output.replace_range(edit.byte_start..edit.byte_end, &edit.replacement);
        }
    }
    output
}


/// Rewrite code matching a pattern across files in a directory.
///
/// pattern_str: &str — Code pattern with metavariables ($NAME, $$$, $_).
/// replacement: &str — Replacement template with metavariable substitution.
/// lang_name: &str — Language name to match against.
/// root: &Path — Directory or file to search within.
/// respect_gitignore: bool — Whether to honor .gitignore rules.
/// limit: usize — Maximum number of files to rewrite.
/// dry_run: bool — When true, compute edits but do not write files.
/// specs: &[LanguageSpec] — Language specs for file detection and parsing.
pub fn pattern_rewrite(
    pattern_str: &str,
    replacement: &str,
    lang_name: &str,
    root: &Path,
    respect_gitignore: bool,
    limit: usize,
    dry_run: bool,
    specs: &[LanguageSpec],
) -> Result<RewriteResult, String> {
    if pattern_str.trim().is_empty() {
        return Err("pattern cannot be empty".to_string());
    }
    if !root.exists() {
        return Err(format!("path does not exist: {}", root.display()));
    }

    let start = Instant::now();
    let lang_lower = lang_name.to_lowercase();

    let spec = specs
        .iter()
        .find(|s| s.name.to_lowercase() == lang_lower)
        .ok_or_else(|| format!("unsupported language: {lang_name}"))?;

    let pattern = parse_pattern(pattern_str, spec)?;
    let files = collect_files(root, respect_gitignore, &lang_lower, specs);
    let searched_files = files.len();

    let mut file_results = Vec::new();
    let mut total_edits = 0usize;
    let effective_limit = limit.max(1);

    for (file_path, _file_lang) in &files {
        let content = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        if content.contains('\0') {
            continue;
        }

        let edits = rewrite_file(&pattern, replacement, &content, spec);
        if edits.is_empty() {
            continue;
        }

        let edit_count = edits.len();
        let rewritten = apply_edits(&content, &edits);

        if !dry_run {
            if let Err(e) = fs::write(file_path, &rewritten) {
                return Err(format!("failed to write {file_path}: {e}"));
            }
        }

        file_results.push(RewriteFileResult {
            file_path: file_path.clone(),
            edits,
            rewritten_source: if dry_run { Some(rewritten) } else { None },
        });

        total_edits += edit_count;

        if file_results.len() >= effective_limit {
            break;
        }
    }

    Ok(RewriteResult {
        files: file_results,
        searched_files,
        total_edits,
        dry_run,
        duration_ms: start.elapsed().as_millis(),
    })
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_template_expands_ellipsis_capture() {
        let mut env = MatchEnv::default();
        env.multis.insert("ARGS".to_string(), vec!["a".to_string(), "b".to_string()]);
        env.singles.insert("NAME".to_string(), "f".to_string());
        assert_eq!(substitute_template("$NAME($$$ARGS)", &env), "f(a, b)");
    }

    #[test]
    fn substitute_template_handles_trailing_ellipsis() {
        let env = MatchEnv::default();
        assert_eq!(substitute_template("call($$$", &env), "call($$$");
        assert_eq!(substitute_template("$$$", &env), "$$$");
    }
}
