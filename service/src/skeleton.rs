use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::path::Path;

use serde::Serialize;

use crate::extract::{extract_path_with_options, ExtractOptions};
use crate::lang::LanguageSpec;
use crate::types::{ExtractedSymbol, SymbolKind};


#[derive(Debug, Clone)]
pub struct SkeletonOptions {
    pub max_depth: usize,
    pub include_signatures: bool,
    pub include_line_numbers: bool,
    pub collapse_threshold: usize,
    pub respect_gitignore: bool,
}

impl Default for SkeletonOptions {
    fn default() -> Self {
        Self {
            max_depth: 20,
            include_signatures: true,
            include_line_numbers: false,
            collapse_threshold: 0,
            respect_gitignore: true,
        }
    }
}


#[derive(Debug, Clone, Serialize)]
pub struct SkeletonSymbol {
    pub name: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    pub line_start: usize,
    pub line_end: usize,
}


#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum SkeletonNode {
    #[serde(rename = "directory")]
    Directory {
        name: String,
        children: Vec<SkeletonNode>,
        file_count: usize,
        symbol_count: usize,
    },
    #[serde(rename = "file")]
    File {
        name: String,
        language: String,
        symbols: Vec<SkeletonSymbol>,
    },
    #[serde(rename = "collapsed")]
    Collapsed {
        name: String,
        file_count: usize,
        symbol_count: usize,
    },
}


#[derive(Debug, Clone, Serialize)]
pub struct SkeletonResult {
    pub root: SkeletonNode,
    pub total_files: usize,
    pub total_symbols: usize,
    pub total_directories: usize,
    pub duration_ms: u128,
}


struct FileEntry {
    rel_path: String,
    language: String,
    symbols: Vec<SkeletonSymbol>,
}


pub fn build_skeleton(
    path: &Path,
    specs: &[LanguageSpec],
    options: &SkeletonOptions,
) -> Result<SkeletonResult, String> {
    let start = std::time::Instant::now();

    let root = path
        .canonicalize()
        .map_err(|e| format!("cannot resolve path: {e}"))?;

    let extract_opts = ExtractOptions {
        respect_gitignore: options.respect_gitignore,
    };

    let results = extract_path_with_options(&root, specs, extract_opts)?;

    let root_str = root.to_string_lossy().replace('\\', "/");

    let mut entries: Vec<FileEntry> = Vec::with_capacity(results.len());

    for file_result in &results {
        let normalized = file_result.file_path.replace('\\', "/");
        let rel = if normalized.starts_with(&root_str) {
            let stripped = &normalized[root_str.len()..];
            stripped.trim_start_matches('/')
        } else {
            &normalized
        };

        let symbols: Vec<SkeletonSymbol> = file_result
            .symbols
            .iter()
            .filter(|s| is_skeleton_worthy(s.kind))
            .map(|s| symbol_to_skeleton(s, options))
            .collect();

        entries.push(FileEntry {
            rel_path: rel.to_string(),
            language: file_result.language.clone(),
            symbols,
        });
    }

    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    let root_name = root
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string());

    let entry_refs: Vec<(&str, &FileEntry)> = entries
        .iter()
        .map(|entry| (entry.rel_path.as_str(), entry))
        .collect();
    let tree = build_tree(&root_name, &entry_refs, options, 0);

    let total_files = entries.len();
    let total_symbols: usize = entries.iter().map(|e| e.symbols.len()).sum();
    let total_directories = count_directories(&tree);
    let duration_ms = start.elapsed().as_millis();

    Ok(SkeletonResult {
        root: tree,
        total_files,
        total_symbols,
        total_directories,
        duration_ms,
    })
}


fn is_skeleton_worthy(kind: SymbolKind) -> bool {
    !matches!(
        kind,
        SymbolKind::Import
            | SymbolKind::Variable
            | SymbolKind::Decorator
            | SymbolKind::HtmlTag
            | SymbolKind::CssSelector
            | SymbolKind::Heading
            | SymbolKind::DataKey
    )
}


fn symbol_to_skeleton(sym: &ExtractedSymbol, options: &SkeletonOptions) -> SkeletonSymbol {
    let kind_str = format!("{:?}", sym.kind).to_ascii_lowercase();

    let sig = if options.include_signatures {
        fix_signature(sym)
    } else {
        None
    };

    let parent = clean_parent(&sym.parent);

    SkeletonSymbol {
        name: sym.name.clone(),
        kind: kind_str,
        signature: sig,
        parent,
        visibility: sym.visibility.clone(),
        line_start: sym.line_start,
        line_end: sym.line_end,
    }
}


fn fix_signature(sym: &ExtractedSymbol) -> Option<String> {
    let raw = sym.signature.as_deref()?;
    let trimmed = raw.trim();

    if trimmed.starts_with('@') {
        return build_signature_from_source(sym);
    }

    Some(trimmed.to_string())
}


fn build_signature_from_source(sym: &ExtractedSymbol) -> Option<String> {
    for line in sym.source.lines() {
        let t = line.trim();
        if t.starts_with('@') || t.is_empty() {
            continue;
        }
        if t.starts_with("def ") || t.starts_with("async def ")
            || t.starts_with("fn ") || t.starts_with("pub fn ")
            || t.starts_with("class ") || t.starts_with("pub struct ")
        {
            return Some(t.trim_end_matches('{').trim_end_matches(':').trim_end().to_string());
        }
        return Some(t.to_string());
    }
    None
}


fn clean_parent(parent: &Option<String>) -> Option<String> {
    let p = parent.as_deref()?;
    let trimmed = p.trim();

    if trimmed.starts_with("from ") || trimmed.starts_with("import ") {
        return None;
    }

    if trimmed.len() > 80 {
        return None;
    }

    Some(trimmed.to_string())
}


fn build_tree(
    root_name: &str,
    entries: &[(&str, &FileEntry)],
    options: &SkeletonOptions,
    depth: usize,
) -> SkeletonNode {
    let mut dirs: BTreeMap<&str, Vec<(&str, &FileEntry)>> = BTreeMap::new();
    let mut root_files: Vec<(&str, &FileEntry)> = Vec::new();

    for &(rel_path, entry) in entries {
        if let Some((dir_name, rest)) = rel_path.split_once('/') {
            dirs.entry(dir_name).or_default().push((rest, entry));
        } else {
            root_files.push((rel_path, entry));
        }
    }

    let mut children: Vec<SkeletonNode> = Vec::new();

    for (dir_name, sub_entries) in &dirs {
        let file_count = sub_entries.len();
        let symbol_count: usize = sub_entries.iter().map(|(_, e)| e.symbols.len()).sum();

        let collapse = (options.collapse_threshold > 0
            && file_count <= options.collapse_threshold
            && symbol_count == 0)
            || depth + 1 >= options.max_depth;
        if collapse {
            children.push(SkeletonNode::Collapsed {
                name: dir_name.to_string(),
                file_count,
                symbol_count,
            });
            continue;
        }

        children.push(build_tree(dir_name, sub_entries, options, depth + 1));
    }

    for (rel_path, file) in root_files {
        children.push(SkeletonNode::File {
            name: rel_path.to_string(),
            language: file.language.clone(),
            symbols: file.symbols.clone(),
        });
    }

    let file_count = entries.len();
    let symbol_count: usize = entries.iter().map(|(_, e)| e.symbols.len()).sum();

    SkeletonNode::Directory {
        name: root_name.to_string(),
        children,
        file_count,
        symbol_count,
    }
}


fn count_directories(node: &SkeletonNode) -> usize {
    match node {
        SkeletonNode::Directory { children, .. } => {
            1 + children.iter().map(count_directories).sum::<usize>()
        }
        SkeletonNode::Collapsed { .. } => 1,
        SkeletonNode::File { .. } => 0,
    }
}


pub fn render_markdown(result: &SkeletonResult, options: &SkeletonOptions) -> String {
    let mut out = String::with_capacity(4096);
    render_node(&result.root, &mut out, "", true, options);
    out
}


fn render_node(
    node: &SkeletonNode,
    out: &mut String,
    prefix: &str,
    is_last: bool,
    options: &SkeletonOptions,
) {
    let connector = if prefix.is_empty() {
        ""
    } else if is_last {
        "└── "
    } else {
        "├── "
    };

    let child_prefix = if prefix.is_empty() {
        String::new()
    } else if is_last {
        format!("{prefix}    ")
    } else {
        format!("{prefix}│   ")
    };

    match node {
        SkeletonNode::Directory {
            name,
            children,
            file_count,
            symbol_count,
        } => {
            let _ = writeln!(
                out,
                "{prefix}{connector}{name}/ ({file_count} files, {symbol_count} symbols)"
            );

            for (i, child) in children.iter().enumerate() {
                let last = i == children.len() - 1;
                render_node(child, out, &child_prefix, last, options);
            }
        }

        SkeletonNode::File {
            name,
            language,
            symbols,
        } => {
            let _ = writeln!(
                out,
                "{prefix}{connector}{name} [{language}] ({} symbols)",
                symbols.len()
            );

            let sym_prefix = &child_prefix;

            for (i, sym) in symbols.iter().enumerate() {
                let sym_last = i == symbols.len() - 1;
                let sym_connector = if sym_last { "└── " } else { "├── " };

                let kind_short = kind_prefix(&sym.kind);
                let parent_str = sym
                    .parent
                    .as_ref()
                    .map(|p| format!(" ({p})"))
                    .unwrap_or_default();

                let line_str = if options.include_line_numbers {
                    format!(" L{}-{}", sym.line_start, sym.line_end)
                } else {
                    String::new()
                };

                if let Some(sig) = &sym.signature {
                    let clean = clean_signature(sig, &sym.kind);
                    let _ = writeln!(
                        out,
                        "{sym_prefix}{sym_connector}{kind_short} {clean}{parent_str}{line_str}"
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "{sym_prefix}{sym_connector}{kind_short} {}{parent_str}{line_str}",
                        sym.name
                    );
                }
            }
        }

        SkeletonNode::Collapsed {
            name,
            file_count,
            symbol_count,
        } => {
            let _ = writeln!(
                out,
                "{prefix}{connector}{name}/ ({file_count} files, {symbol_count} symbols)"
            );
        }
    }
}


fn kind_prefix(kind: &str) -> &str {
    match kind {
        "function" => "fn",
        "method" => "fn",
        "class" => "class",
        "struct" => "struct",
        "enum" => "enum",
        "interface" => "interface",
        "trait" => "trait",
        "impl" => "impl",
        "constant" => "const",
        "variable" => "var",
        "import" => "import",
        "export" => "export",
        "type" => "type",
        "field" => "field",
        "property" => "prop",
        "module" => "mod",
        "namespace" => "ns",
        "decorator" => "deco",
        "annotation" => "anno",
        "macro" => "macro",
        "rule" => "rule",
        "selector" => "sel",
        "mediaquery" => "media",
        "keyframes" => "keyframes",
        "heading" => "heading",
        "codeblock" => "code",
        "link" => "link",
        _ => kind,
    }
}


fn clean_signature(sig: &str, kind: &str) -> String {
    let trimmed = sig.trim();

    let line = if let Some(pos) = trimmed.find('\n') {
        &trimmed[..pos]
    } else {
        trimmed
    };

    let cleaned = line
        .trim_end_matches('{')
        .trim_end_matches(':')
        .trim_end();

    let (max_len, keep) = match kind {
        "function" | "method" => (120, 117),
        _ => (100, 97),
    };

    if cleaned.len() > max_len {
        let mut end = keep;
        while !cleaned.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &cleaned[..end])
    } else {
        cleaned.to_string()
    }
}


#[cfg(test)]
mod tests {
    use super::clean_signature;

    #[test]
    fn clean_signature_truncates_on_char_boundary() {
        let sig = format!("def ff(x='{}'):", "\u{4e2d}".repeat(60));
        let cleaned = clean_signature(&sig, "function");
        assert!(cleaned.ends_with("..."));
        assert!(cleaned.len() <= 120);
    }
}
