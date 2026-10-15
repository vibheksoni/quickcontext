use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use ignore::WalkBuilder;
use tree_sitter::{Node, Parser};

use crate::lang::{self, LanguageSpec};
use crate::types::{PatternMatchCapture, PatternMatchItem, PatternMatchResult};


const QUICK_IGNORE_FILENAME: &str = ".quick-ignore";
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024;
const META_CHAR: char = '$';


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetaVariable {
    Capture(String),
    Dropped,
    Ellipsis,
    EllipsisCapture(String),
}


#[derive(Clone, Debug)]
pub enum PatternNode {
    Terminal {
        text: String,
        kind_id: u16,
        is_named: bool,
    },
    MetaVar {
        meta_var: MetaVariable,
    },
    Internal {
        kind_id: u16,
        children: Vec<PatternNode>,
    },
}


#[derive(Clone, Debug, Default)]
pub struct MatchEnv {
    pub singles: HashMap<String, String>,
    pub multis: HashMap<String, Vec<String>>,
}


/// Extract a MetaVariable from a source text token.
///
/// src: &str — Token text from the parsed pattern AST.
fn extract_meta_var(src: &str) -> Option<MetaVariable> {
    if src == "$$$" {
        return Some(MetaVariable::Ellipsis);
    }

    if let Some(name) = src.strip_prefix("$$$") {
        if name.starts_with('_') {
            return Some(MetaVariable::Ellipsis);
        }
        if is_valid_meta_name(name) {
            return Some(MetaVariable::EllipsisCapture(name.to_string()));
        }
        return None;
    }

    if !src.starts_with(META_CHAR) {
        return None;
    }

    let name = &src[1..];
    if name.is_empty() || !name.starts_with(|c: char| c.is_ascii_uppercase() || c == '_') {
        return None;
    }
    if !is_valid_meta_name(name) {
        return None;
    }

    if name.starts_with('_') {
        Some(MetaVariable::Dropped)
    } else {
        Some(MetaVariable::Capture(name.to_string()))
    }
}


/// Check if a string is a valid metavariable name.
///
/// name: &str — Candidate name after stripping $ prefix.
fn is_valid_meta_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}


/// Check if a PatternNode is an ERROR node containing "$" text.
///
/// node: &PatternNode — Node to check.
fn is_dollar_error(node: &PatternNode) -> bool {
    match node {
        PatternNode::Internal { kind_id, children } if *kind_id == u16::MAX => {
            children.iter().any(|c| matches!(c,
                PatternNode::Terminal { text, .. } if text == "$"
            ))
        }
        PatternNode::Terminal { text, kind_id, .. } if *kind_id == u16::MAX => {
            text == "$"
        }
        _ => false,
    }
}


/// Extract the identifier text from a Terminal or Internal node.
///
/// node: &PatternNode — Node to extract text from.
fn terminal_text(node: &PatternNode) -> Option<&str> {
    match node {
        PatternNode::Terminal { text, .. } => Some(text.as_str()),
        _ => None,
    }
}


/// Merge split metavariables in a children list.
///
/// Tree-sitter parses "$NAME" as ERROR("$") + identifier("NAME") in most
/// languages. This pass detects that pattern and merges them into a single
/// MetaVar node.
///
/// children: Vec<PatternNode> — Raw children from node_to_pattern.
fn fixup_meta_vars(children: Vec<PatternNode>) -> Vec<PatternNode> {
    let mut result = Vec::with_capacity(children.len());
    let mut i = 0;

    while i < children.len() {
        if is_dollar_error(&children[i]) && i + 1 < children.len() {
            if let Some(name) = terminal_text(&children[i + 1]) {
                let combined = format!("${name}");
                if let Some(meta_var) = extract_meta_var(&combined) {
                    result.push(PatternNode::MetaVar { meta_var });
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        }

        result.push(children[i].clone());
        i += 1;
    }

    result
}


/// Convert a tree-sitter AST node into a PatternNode tree.
///
/// Recursively walks the AST, detecting metavariables by text content.
/// Applies fixup pass to merge split metavariables (ERROR("$") + identifier).
///
/// node: Node — Tree-sitter node from the parsed pattern.
/// source: &[u8] — Pattern source bytes for text extraction.
fn node_to_pattern(node: Node, source: &[u8]) -> PatternNode {
    let text = node.utf8_text(source).unwrap_or("").to_string();

    if let Some(meta_var) = extract_meta_var(&text) {
        return PatternNode::MetaVar { meta_var };
    }

    if node.child_count() == 0 {
        return PatternNode::Terminal {
            text,
            kind_id: node.kind_id(),
            is_named: node.is_named(),
        };
    }

    let mut children = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.is_missing() {
            continue;
        }
        children.push(node_to_pattern(child, source));
    }

    children = fixup_meta_vars(children);

    PatternNode::Internal {
        kind_id: node.kind_id(),
        children,
    }
}


/// Unwrap syntactic wrappers from a pattern AST.
///
/// Tree-sitter wraps parsed code in root nodes (module, program, source_file)
/// and statement wrappers (expression_statement, etc.). For pattern matching
/// we want the innermost meaningful node — the actual expression or statement
/// the user intended to match.
///
/// Recursively unwraps single-child Internal nodes until we reach a node with
/// multiple children or a Terminal/MetaVar leaf.
///
/// node: PatternNode — Root pattern node from node_to_pattern.
fn unwrap_root(node: PatternNode) -> PatternNode {
    match node {
        PatternNode::Internal { children, .. } if children.len() == 1 => {
            unwrap_root(children.into_iter().next().unwrap())
        }
        other => other,
    }
}


/// Parse a code pattern string into a PatternNode tree for a given language.
///
/// pattern: &str — Code pattern with metavariables ($NAME, $$$, $_).
/// spec: &LanguageSpec — Language spec for tree-sitter parsing.
pub fn parse_pattern(pattern: &str, spec: &LanguageSpec) -> Result<PatternNode, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&spec.language)
        .map_err(|e| format!("failed to set language: {e}"))?;

    let tree = parser
        .parse(pattern, None)
        .ok_or_else(|| "failed to parse pattern".to_string())?;

    let root = tree.root_node();
    if root.child_count() == 0 {
        return Err("pattern produced empty AST".to_string());
    }

    let mut pattern_node = node_to_pattern(root, pattern.as_bytes());
    pattern_node = unwrap_root(pattern_node);
    Ok(pattern_node)
}


/// Match a single PatternNode against a single AST node.
///
/// goal: &PatternNode — Pattern node to match.
/// candidate: Node — AST node from the candidate source.
/// source: &[u8] — Candidate source bytes.
/// env: &mut MatchEnv — Capture environment for metavariables.
fn match_one(goal: &PatternNode, candidate: Node, source: &[u8], env: &mut MatchEnv) -> bool {
    match goal {
        PatternNode::Terminal { text, kind_id, .. } => {
            if candidate.kind_id() != *kind_id {
                return false;
            }
            if text.is_empty() {
                return true;
            }
            let cand_text = candidate.utf8_text(source).unwrap_or("");
            cand_text == text
        }

        PatternNode::MetaVar { meta_var } => {
            let cand_text = candidate.utf8_text(source).unwrap_or("").to_string();
            match meta_var {
                MetaVariable::Capture(name) => {
                    if let Some(existing) = env.singles.get(name) {
                        return *existing == cand_text;
                    }
                    env.singles.insert(name.clone(), cand_text);
                    true
                }
                MetaVariable::Dropped => true,
                MetaVariable::Ellipsis | MetaVariable::EllipsisCapture(_) => true,
            }
        }

        PatternNode::Internal { kind_id, children } => {
            let same_kind = candidate.kind_id() == *kind_id;
            if !same_kind && (!candidate.is_named() || children.is_empty()) {
                return false;
            }

            let mut cursor = candidate.walk();
            let cand_children: Vec<Node> = candidate.children(&mut cursor).collect();

            if same_kind {
                return match_children(children, &cand_children, source, env);
            }

            let mut flex_env = env.clone();
            if match_children(children, &cand_children, source, &mut flex_env) {
                *env = flex_env;
                return true;
            }

            false
        }
    }
}


/// Check if a PatternNode is an ellipsis variant.
///
/// node: &PatternNode — Pattern node to check.
fn is_ellipsis(node: &PatternNode) -> Option<Option<String>> {
    match node {
        PatternNode::MetaVar { meta_var: MetaVariable::Ellipsis } => Some(None),
        PatternNode::MetaVar { meta_var: MetaVariable::EllipsisCapture(name) } => {
            Some(Some(name.clone()))
        }
        _ => None,
    }
}


/// Match pattern children against candidate children with ellipsis support.
///
/// goals: &[PatternNode] — Pattern child nodes.
/// candidates: &[Node] — Candidate AST child nodes.
/// source: &[u8] — Candidate source bytes.
/// env: &mut MatchEnv — Capture environment.
fn match_children(
    goals: &[PatternNode],
    candidates: &[Node],
    source: &[u8],
    env: &mut MatchEnv,
) -> bool {
    let mut gi = 0;
    let mut ci = 0;

    while gi < goals.len() {
        if let Some(capture_name) = is_ellipsis(&goals[gi]) {
            gi += 1;

            if gi >= goals.len() {
                let mut matched = Vec::new();
                while ci < candidates.len() {
                    matched.push(
                        candidates[ci].utf8_text(source).unwrap_or("").to_string(),
                    );
                    ci += 1;
                }
                if let Some(name) = capture_name {
                    env.multis.insert(name, matched);
                }
                return true;
            }

            let mut matched = Vec::new();
            loop {
                if ci >= candidates.len() {
                    return false;
                }
                let mut trial_env = env.clone();
                if match_one(&goals[gi], candidates[ci], source, &mut trial_env) {
                    if let Some(name) = capture_name {
                        env.multis.insert(name, matched);
                    }
                    *env = trial_env;
                    gi += 1;
                    ci += 1;
                    break;
                }
                matched.push(
                    candidates[ci].utf8_text(source).unwrap_or("").to_string(),
                );
                ci += 1;
            }
            continue;
        }

        if ci >= candidates.len() {
            return false;
        }

        if !match_one(&goals[gi], candidates[ci], source, env) {
            return false;
        }

        gi += 1;
        ci += 1;
    }

    true
}


/// Recursively walk an AST tree and collect all nodes matching the pattern.
///
/// pattern: &PatternNode — Compiled pattern to match against.
/// node: Node — Current AST node being examined.
/// source: &[u8] — Source bytes of the candidate file.
pub fn find_matches_in_tree<'a>(
    pattern: &PatternNode,
    node: Node<'a>,
    source: &[u8],
    results: &mut Vec<(Node<'a>, MatchEnv)>,
) {
    let mut cursor = node.walk();
    loop {
        let current = cursor.node();
        let mut env = MatchEnv::default();
        if match_one(pattern, current, source, &mut env) {
            results.push((current, env));
        }

        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return;
            }
        }
    }
}


/// Match a pattern against a single file's source code.
///
/// pattern: &PatternNode — Compiled pattern tree.
/// source: &str — File source code.
/// spec: &LanguageSpec — Language spec for parsing.
fn match_file(
    pattern: &PatternNode,
    source: &str,
    spec: &LanguageSpec,
) -> Vec<(usize, usize, usize, usize, String, MatchEnv)> {
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

    let mut results = Vec::new();
    for (node, env) in raw_matches {
        let start = node.start_position();
        let end = node.end_position();
        let matched_text = node.utf8_text(source_bytes).unwrap_or("").to_string();
        results.push((
            start.row + 1,
            start.column + 1,
            end.row + 1,
            end.column + 1,
            matched_text,
            env,
        ));
    }

    results
}


/// Search for AST pattern matches across files in a directory.
///
/// pattern_str: &str — Code pattern with metavariables ($NAME, $$$).
/// lang_name: &str — Language name to match against (e.g. "rust", "python").
/// root: &Path — Directory or file to search within.
/// respect_gitignore: bool — Whether to honor .gitignore rules.
/// limit: usize — Maximum number of matches to return.
/// specs: &[LanguageSpec] — Language specs for file detection and parsing.
pub fn pattern_search(
    pattern_str: &str,
    lang_name: &str,
    root: &Path,
    respect_gitignore: bool,
    limit: usize,
    specs: &[LanguageSpec],
) -> Result<PatternMatchResult, String> {
    if pattern_str.trim().is_empty() {
        return Err("pattern cannot be empty".to_string());
    }
    if !root.exists() {
        return Err(format!("path does not exist: {}", root.display()));
    }

    let start = Instant::now();
    let effective_limit = limit.max(1);
    let lang_lower = lang_name.to_lowercase();

    let spec = specs
        .iter()
        .find(|s| s.name.to_lowercase() == lang_lower)
        .ok_or_else(|| format!("unsupported language: {lang_name}"))?;

    let pattern = parse_pattern(pattern_str, spec)?;
    let files = collect_files(root, respect_gitignore, &lang_lower, specs);
    let searched_files = files.len();

    let mut all_matches = Vec::new();

    for (file_path, file_lang) in &files {
        let content = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        if content.contains('\0') {
            continue;
        }

        let file_matches = match_file(&pattern, &content, spec);

        for (line_start, col_start, line_end, col_end, matched_text, env) in file_matches {
            let mut captures = Vec::new();
            for (name, text) in &env.singles {
                captures.push(PatternMatchCapture {
                    name: name.clone(),
                    text: text.clone(),
                });
            }
            for (name, texts) in &env.multis {
                captures.push(PatternMatchCapture {
                    name: name.clone(),
                    text: texts.join(", "),
                });
            }

            all_matches.push(PatternMatchItem {
                file_path: file_path.clone(),
                language: file_lang.clone(),
                matched_text,
                line_start,
                column_start: col_start,
                line_end,
                column_end: col_end,
                captures,
            });

            if all_matches.len() > effective_limit {
                break;
            }
        }

        if all_matches.len() > effective_limit {
            break;
        }
    }

    let truncated = all_matches.len() > effective_limit;
    all_matches.truncate(effective_limit);

    Ok(PatternMatchResult {
        matches: all_matches,
        searched_files,
        duration_ms: start.elapsed().as_millis(),
        truncated,
    })
}


/// Collect files matching a specific language from a directory.
///
/// root: &Path — Directory or file to walk.
/// respect_gitignore: bool — Honor .gitignore rules.
/// lang_lower: &str — Lowercase language name to filter by.
/// specs: &[LanguageSpec] — Language specs for file detection.
pub(crate) fn collect_files(
    root: &Path,
    respect_gitignore: bool,
    lang_lower: &str,
    specs: &[LanguageSpec],
) -> Vec<(String, String)> {
    if root.is_file() {
        let path_str = root.to_string_lossy().to_string();
        let detected = lang::detect_language(&path_str, specs);
        if let Some(spec) = detected {
            if spec.name.to_lowercase() == *lang_lower {
                return vec![(path_str, spec.name.to_string())];
            }
        }
        return Vec::new();
    }

    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(true)
        .add_custom_ignore_filename(QUICK_IGNORE_FILENAME);

    if !respect_gitignore {
        builder.git_ignore(false).git_global(false).git_exclude(false);
    }

    let mut entries = Vec::new();

    for entry in builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        if let Ok(meta) = entry.metadata() {
            if meta.len() > MAX_FILE_SIZE {
                continue;
            }
        }

        let path_str = entry.path().to_string_lossy().to_string();
        let detected = lang::detect_language(&path_str, specs);
        if let Some(spec) = detected {
            if spec.name.to_lowercase() == *lang_lower {
                entries.push((path_str, spec.name.to_string()));
            }
        }
    }

    entries
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_meta_var_capture() {
        assert_eq!(
            extract_meta_var("$NAME"),
            Some(MetaVariable::Capture("NAME".to_string()))
        );
        assert_eq!(
            extract_meta_var("$FOO_BAR"),
            Some(MetaVariable::Capture("FOO_BAR".to_string()))
        );
    }

    #[test]
    fn test_extract_meta_var_dropped() {
        assert_eq!(extract_meta_var("$_"), Some(MetaVariable::Dropped));
        assert_eq!(extract_meta_var("$_FOO"), Some(MetaVariable::Dropped));
    }

    #[test]
    fn test_extract_meta_var_ellipsis() {
        assert_eq!(extract_meta_var("$$$"), Some(MetaVariable::Ellipsis));
        assert_eq!(
            extract_meta_var("$$$ARGS"),
            Some(MetaVariable::EllipsisCapture("ARGS".to_string()))
        );
        assert_eq!(extract_meta_var("$$$_"), Some(MetaVariable::Ellipsis));
    }

    #[test]
    fn test_extract_meta_var_invalid() {
        assert_eq!(extract_meta_var("hello"), None);
        assert_eq!(extract_meta_var("$lowercase"), None);
        assert_eq!(extract_meta_var("$"), None);
        assert_eq!(extract_meta_var(""), None);
    }

    #[test]
    fn test_match_python_def() {
        let specs = crate::lang::registry();
        let py = specs.iter().find(|s| s.name == "python").unwrap();
        let pattern = parse_pattern("def $NAME($$$):", py).unwrap();
        let results = match_file(&pattern, "def hello(x, y):\n    pass\n", py);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].5.singles.get("NAME").unwrap(), "hello");
    }

    #[test]
    fn test_match_rust_fn() {
        let specs = crate::lang::registry();
        let rs = specs.iter().find(|s| s.name == "rust").unwrap();
        let pattern = parse_pattern("fn $NAME($$$)", rs).unwrap();
        let src = "fn hello(x: i32) -> bool { true }\nfn world() { }";
        let results = match_file(&pattern, src, rs);
        assert_eq!(results.len(), 2);
        let names: Vec<&str> = results.iter()
            .map(|r| r.5.singles.get("NAME").unwrap().as_str())
            .collect();
        assert!(names.contains(&"hello"));
        assert!(names.contains(&"world"));
    }

    #[test]
    fn test_match_rust_pub_fn() {
        let specs = crate::lang::registry();
        let rs = specs.iter().find(|s| s.name == "rust").unwrap();
        let pattern = parse_pattern("pub fn $NAME($$$)", rs).unwrap();
        let src = "pub fn visible() { }\nfn private() { }";
        let results = match_file(&pattern, src, rs);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].5.singles.get("NAME").unwrap(), "visible");
    }

    #[test]
    fn test_match_rust_struct() {
        let specs = crate::lang::registry();
        let rs = specs.iter().find(|s| s.name == "rust").unwrap();
        let pattern = parse_pattern("struct $NAME", rs).unwrap();
        let src = "struct Foo { x: i32 }\nstruct Bar;";
        let results = match_file(&pattern, src, rs);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_is_valid_meta_name() {
        assert!(is_valid_meta_name("NAME"));
        assert!(is_valid_meta_name("FOO_BAR"));
        assert!(is_valid_meta_name("A1"));
        assert!(is_valid_meta_name("_"));
        assert!(!is_valid_meta_name(""));
        assert!(!is_valid_meta_name("lowercase"));
    }
}
