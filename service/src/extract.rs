use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use sha2::{Digest, Sha256};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Node, Parser, Query, QueryCursor};

use crate::lang::{self, LanguageSpec};
use crate::role_classify;
use crate::types::{
    CompactExtractionResult, ExtractStats, ExtractedSymbol, ExtractionResult, FileScanEntry, SymbolKind,
};

const QUICK_IGNORE_FILENAME: &str = ".quick-ignore";

const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    ".git/",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "dist/",
    "build/",
    "target/",
    ".next/",
    ".nuxt/",
    ".idea/",
    ".vscode/",
    ".cache/",
    "*.log",
    "*.lock",
    ".DS_Store",
    "Thumbs.db",
    "*.css",
    "*.scss",
    "*.less",
    "*.sass",
];

#[derive(Debug, Clone, Copy)]
pub struct ExtractOptions {
    pub respect_gitignore: bool,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            respect_gitignore: true,
        }
    }
}

pub fn extract_path(path: &Path, specs: &[LanguageSpec]) -> Result<Vec<ExtractionResult>, String> {
    extract_path_with_options(path, specs, ExtractOptions::default())
}

pub fn extract_path_with_options(
    path: &Path,
    specs: &[LanguageSpec],
    options: ExtractOptions,
) -> Result<Vec<ExtractionResult>, String> {
    if !path.exists() {
        return Err(format!("path does not exist: {}", path.display()));
    }

    if path.is_file() {
        let path_str = path.to_string_lossy();
        let spec = lang::detect_language(&path_str, specs)
            .ok_or_else(|| format!("unsupported file type: {}", path.display()))?;
        let source = std::fs::read_to_string(path).map_err(|e| format!("failed to read file: {e}"))?;
        let meta = std::fs::metadata(path).ok();
        let mut result = extract_file(&path_str, &source, spec);
        result.file_hash = Some(sha256_hex(source.as_bytes()));
        result.file_size = meta.as_ref().map(|m| m.len());
        result.file_mtime = meta.as_ref().and_then(|m| {
            m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs())
        });
        Ok(vec![result])
    } else {
        Ok(extract_directory(path, specs, options))
    }
}

fn extract_directory(dir: &Path, specs: &[LanguageSpec], options: ExtractOptions) -> Vec<ExtractionResult> {
    let mut builder = ignore::WalkBuilder::new(dir);
    builder.hidden(true).add_custom_ignore_filename(QUICK_IGNORE_FILENAME);

    if options.respect_gitignore {
        builder.git_ignore(true).git_global(true).git_exclude(true);
    } else {
        builder.git_ignore(false).git_global(false).git_exclude(false);
    }

    if let Ok(default_ignore_matcher) = build_default_ignore(dir) {
        builder.filter_entry(move |entry| {
            let is_dir = entry.file_type().map_or(false, |ft| ft.is_dir());
            !default_ignore_matcher
                .matched_path_or_any_parents(entry.path(), is_dir)
                .is_ignore()
        });
    }

    let mut results = Vec::new();
    let walker = builder.build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().map_or(false, |ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let path_str = path.to_string_lossy();

        let spec = match lang::detect_language(&path_str, specs) {
            Some(s) => s,
            None => continue,
        };

        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let meta = std::fs::metadata(path).ok();
        let mut result = extract_file(&path_str, &source, spec);
        result.file_hash = Some(sha256_hex(source.as_bytes()));
        result.file_size = meta.as_ref().map(|m| m.len());
        result.file_mtime = meta.as_ref().and_then(|m| {
            m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs())
        });

        if !result.symbols.is_empty() || !result.errors.is_empty() {
            results.push(result);
        }
    }

    results
}

/// Stream compact extraction results as JSONL to the given writer.
///
/// dir: &Path — Directory to extract from.
/// specs: &[LanguageSpec] — Language spec registry.
/// options: ExtractOptions — Gitignore control.
/// writer: &mut W — Output sink for JSONL (one compact result per line).
/// Returns: ExtractStats — Aggregate stats across all files.
pub fn extract_compact_streaming<W: std::io::Write>(
    dir: &Path,
    specs: &[LanguageSpec],
    options: ExtractOptions,
    writer: &mut W,
) -> ExtractStats {
    let start = std::time::Instant::now();
    let mut stats = ExtractStats {
        total_files: 0,
        total_symbols: 0,
        languages: std::collections::HashMap::new(),
        duration_ms: 0,
    };

    let mut builder = ignore::WalkBuilder::new(dir);
    builder.hidden(true).add_custom_ignore_filename(QUICK_IGNORE_FILENAME);

    if options.respect_gitignore {
        builder.git_ignore(true).git_global(true).git_exclude(true);
    } else {
        builder.git_ignore(false).git_global(false).git_exclude(false);
    }

    if let Ok(default_ignore_matcher) = build_default_ignore(dir) {
        builder.filter_entry(move |entry| {
            let is_dir = entry.file_type().map_or(false, |ft| ft.is_dir());
            !default_ignore_matcher
                .matched_path_or_any_parents(entry.path(), is_dir)
                .is_ignore()
        });
    }

    let walker = builder.build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().map_or(false, |ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let path_str = path.to_string_lossy();

        let spec = match lang::detect_language(&path_str, specs) {
            Some(s) => s,
            None => continue,
        };

        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let meta = std::fs::metadata(path).ok();
        let mut result = extract_file_compact(&path_str, &source, spec);
        result.file_hash = Some(sha256_hex(source.as_bytes()));
        result.file_size = meta.as_ref().map(|m| m.len());
        result.file_mtime = meta.as_ref().and_then(|m| {
            m.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
        });

        if result.symbols.is_empty() && result.errors.is_empty() {
            continue;
        }

        let sym_count = result.symbols.len();
        *stats.languages.entry(spec.name.to_string()).or_insert(0) += sym_count;
        stats.total_symbols += sym_count;
        stats.total_files += 1;

        let compact = CompactExtractionResult::from_full(&result);
        if let Ok(json) = serde_json::to_string(&compact) {
            let _ = writeln!(writer, "{json}");
        }
    }

    stats.duration_ms = start.elapsed().as_millis();
    stats
}


pub fn build_default_ignore(root: &Path) -> Result<Gitignore, ignore::Error> {
    let mut builder = GitignoreBuilder::new(root);
    for pattern in DEFAULT_IGNORE_PATTERNS {
        builder.add_line(None, pattern)?;
    }
    let quick_ignore = root.join(QUICK_IGNORE_FILENAME);
    if quick_ignore.is_file() {
        builder.add(quick_ignore);
    }
    builder.build()
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn compiled_query(spec: &LanguageSpec) -> Result<&'static Query, String> {
    static QUERY_CACHE: OnceLock<HashMap<&'static str, Query>> = OnceLock::new();
    let cache = QUERY_CACHE.get_or_init(|| {
        let mut cache = HashMap::new();
        for spec in lang::registry() {
            if let Ok(query) = Query::new(&spec.language, spec.query) {
                cache.insert(spec.name, query);
            }
        }
        cache
    });

    cache
        .get(spec.name)
        .ok_or_else(|| format!("query compile error: missing cached query for {}", spec.name))
}

pub fn collect_supported_files(
    root: &Path,
    specs: &[LanguageSpec],
    options: ExtractOptions,
) -> Vec<(String, String)> {
    if !root.exists() {
        return Vec::new();
    }

    if root.is_file() {
        let path_str = root.to_string_lossy().to_string();
        if let Some(spec) = lang::detect_language(&path_str, specs) {
            return vec![(path_str, spec.name.to_string())];
        }
        return Vec::new();
    }

    let mut builder = ignore::WalkBuilder::new(root);
    builder.hidden(true).add_custom_ignore_filename(QUICK_IGNORE_FILENAME);

    if options.respect_gitignore {
        builder.git_ignore(true).git_global(true).git_exclude(true);
    } else {
        builder.git_ignore(false).git_global(false).git_exclude(false);
    }

    if let Ok(default_ignore_matcher) = build_default_ignore(root) {
        builder.filter_entry(move |entry| {
            let is_dir = entry.file_type().map_or(false, |ft| ft.is_dir());
            !default_ignore_matcher
                .matched_path_or_any_parents(entry.path(), is_dir)
                .is_ignore()
        });
    }

    let mut files = Vec::new();
    let walker = builder.build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().map_or(false, |ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let path_str = path.to_string_lossy().to_string();

        let spec = match lang::detect_language(&path_str, specs) {
            Some(s) => s,
            None => continue,
        };

        files.push((path_str, spec.name.to_string()));
    }

    files
}

pub fn collect_file_signatures_fast(
    root: &Path,
    specs: &[LanguageSpec],
    options: ExtractOptions,
) -> HashMap<String, u64> {
    let mut signatures = HashMap::new();
    for (file_path, _) in collect_supported_files(root, specs, options) {
        let sig = file_signature(Path::new(&file_path));
        signatures.insert(file_path, sig);
    }
    signatures
}

pub fn scan_supported_files(
    root: &Path,
    specs: &[LanguageSpec],
    options: ExtractOptions,
) -> Vec<FileScanEntry> {
    let mut files = Vec::new();

    for (file_path, language) in collect_supported_files(root, specs, options) {
        let meta = match std::fs::metadata(&file_path) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let file_mtime = meta
            .modified()
            .ok()
            .and_then(|ts| ts.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        let Some(file_mtime) = file_mtime else {
            continue;
        };

        files.push(FileScanEntry {
            file_path,
            language,
            file_size: meta.len(),
            file_mtime,
        });
    }

    files
}

pub fn file_signature(path: &Path) -> u64 {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return 0,
    };

    let modified = meta
        .modified()
        .ok()
        .and_then(|ts| ts.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());

    (meta.len() ^ modified).wrapping_mul(1099511628211)
}

pub fn extract_file(file_path: &str, source: &str, spec: &LanguageSpec) -> ExtractionResult {
    extract_file_with_mode(file_path, source, spec, true)
}

pub fn extract_file_compact(file_path: &str, source: &str, spec: &LanguageSpec) -> ExtractionResult {
    extract_file_with_mode(file_path, source, spec, false)
}

fn extract_file_with_mode(
    file_path: &str,
    source: &str,
    spec: &LanguageSpec,
    include_source_details: bool,
) -> ExtractionResult {
    let mut result = ExtractionResult {
        file_path: file_path.to_string(),
        language: spec.name.to_string(),
        symbols: Vec::new(),
        errors: Vec::new(),
        file_hash: None,
        file_size: None,
        file_mtime: None,
    };

    let mut parser = Parser::new();
    if parser.set_language(&spec.language).is_err() {
        result.errors.push(format!("failed to set language: {}", spec.name));
        return result;
    }

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => {
            result.errors.push("parse returned None".to_string());
            return result;
        }
    };

    let query = match compiled_query(spec) {
        Ok(query) => query,
        Err(e) => {
            result.errors.push(e);
            return result;
        }
    };

    let capture_names = query.capture_names();
    let root = tree.root_node();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, root, source.as_bytes());

    while let Some(m) = matches.next() {
        let pattern_idx = m.pattern_index;
        let kind = match spec.pattern_kinds.get(pattern_idx) {
            Some(k) => *k,
            None => {
                result.errors.push(format!("unmapped pattern index: {pattern_idx}"));
                continue;
            }
        };

        let mut name_text: Option<&str> = None;
        let mut definition_node: Option<Node> = None;
        let mut body_node: Option<Node> = None;
        let mut params_text: Option<&str> = None;
        let mut return_type_text: Option<&str> = None;
        let mut doc_text: Option<&str> = None;

        for cap in m.captures {
            let cap_name = capture_names.get(cap.index as usize).copied().unwrap_or("");
            match cap_name {
                "name" => name_text = cap.node.utf8_text(source.as_bytes()).ok(),
                "definition" => definition_node = Some(cap.node),
                "body" => body_node = Some(cap.node),
                "params" => params_text = cap.node.utf8_text(source.as_bytes()).ok(),
                "return_type" => return_type_text = cap.node.utf8_text(source.as_bytes()).ok(),
                "doc" => doc_text = cap.node.utf8_text(source.as_bytes()).ok(),
                _ => {}
            }
        }

        let def_node = match definition_node {
            Some(n) => n,
            None => continue,
        };

        let symbol_name = match name_text {
            Some(n) => n.to_string(),
            None => {
                let src = def_node.utf8_text(source.as_bytes()).unwrap_or("");
                extract_fallback_name(src, kind)
            }
        };

        let definition_text = def_node.utf8_text(source.as_bytes()).unwrap_or("");
        let full_source = if include_source_details {
            definition_text.to_string()
        } else {
            String::new()
        };
        let signature = extract_signature(definition_text);
        let docstring = if include_source_details {
            doc_text
                .map(|s| s.to_string())
                .or_else(|| extract_docstring(source, &def_node, body_node.as_ref()))
        } else {
            None
        };
        let parent = sanitize_parent_name(find_parent_name(def_node, source));
        let visibility = extract_visibility(source, &def_node, spec.name);

        result.symbols.push(ExtractedSymbol {
            name: symbol_name,
            kind,
            language: spec.name.to_string(),
            file_path: file_path.to_string(),
            line_start: def_node.start_position().row,
            line_end: def_node.end_position().row,
            byte_start: def_node.start_byte(),
            byte_end: def_node.end_byte(),
            source: full_source,
            signature,
            docstring,
            params: if include_source_details {
                params_text.map(|s| s.to_string())
            } else {
                None
            },
            return_type: if include_source_details {
                return_type_text.map(|s| s.to_string())
            } else {
                None
            },
            parent,
            visibility,
            role: None,
        });
    }

    dedup_symbols(&mut result.symbols);
    role_classify::classify_symbols(&mut result.symbols, None);
    result
}

fn dedup_symbols(symbols: &mut Vec<ExtractedSymbol>) {
    let ranges: Vec<(usize, usize, String, SymbolKind)> = symbols
        .iter()
        .map(|s| (s.byte_start, s.byte_end, s.name.clone(), s.kind))
        .collect();

    let mut remove = vec![false; symbols.len()];
    for i in 0..ranges.len() {
        if remove[i] {
            continue;
        }
        for j in (i + 1)..ranges.len() {
            if remove[j] {
                continue;
            }
            if ranges[i].2 != ranges[j].2 || ranges[i].3 != ranges[j].3 {
                continue;
            }
            if ranges[j].0 >= ranges[i].0 && ranges[j].1 <= ranges[i].1 {
                remove[j] = true;
            } else if ranges[i].0 >= ranges[j].0 && ranges[i].1 <= ranges[j].1 {
                remove[i] = true;
                break;
            }
        }
    }

    let mut idx = 0;
    symbols.retain(|_| {
        let keep = !remove[idx];
        idx += 1;
        keep
    });
}

fn extract_signature(source: &str) -> Option<String> {
    let first_line = source.lines().next()?;
    let trimmed = first_line.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

fn extract_docstring(source: &str, def_node: &Node, body_node: Option<&Node>) -> Option<String> {
    let body = body_node?;
    let first_child = body.child(0)?;
    let kind = first_child.kind();

    if kind == "expression_statement" {
        let inner = first_child.child(0)?;
        if inner.kind() == "string" || inner.kind() == "concatenated_string" {
            let text = inner.utf8_text(source.as_bytes()).ok()?;
            return Some(strip_docstring_quotes(text));
        }
    }

    if kind == "string" || kind == "concatenated_string" {
        let text = first_child.utf8_text(source.as_bytes()).ok()?;
        return Some(strip_docstring_quotes(text));
    }

    let comment = find_preceding_comment(source, def_node)?;
    Some(comment)
}

fn strip_docstring_quotes(s: &str) -> String {
    let trimmed = s.trim();
    for delim in &["\"\"\"", "'''", "/**", "*/", "///", "//!", "//", "#"] {
        if trimmed.starts_with(delim) || trimmed.ends_with(delim) {
            let mut result = trimmed.to_string();
            for d in &["\"\"\"", "'''", "/**", "*/"] {
                result = result.trim_start_matches(d).trim_end_matches(d).to_string();
            }
            return result.trim().to_string();
        }
    }
    trimmed.to_string()
}

fn find_preceding_comment(source: &str, node: &Node) -> Option<String> {
    let mut prev = node.prev_sibling()?;

    if prev.kind() == "comment" || prev.kind() == "line_comment" || prev.kind() == "block_comment" {
        let text = prev.utf8_text(source.as_bytes()).ok()?;
        return Some(clean_comment(text));
    }

    if prev.kind() == "decorator" || prev.kind() == "decorated_definition" {
        prev = prev.prev_sibling()?;
        if prev.kind() == "comment" || prev.kind() == "line_comment" || prev.kind() == "block_comment" {
            let text = prev.utf8_text(source.as_bytes()).ok()?;
            return Some(clean_comment(text));
        }
    }

    None
}

fn clean_comment(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() == 1 {
        let line = lines[0].trim();
        for prefix in &["///", "//!", "//", "#", "--"] {
            if let Some(rest) = line.strip_prefix(prefix) {
                return rest.trim().to_string();
            }
        }
        return line.to_string();
    }

    let mut cleaned = Vec::new();
    for line in &lines {
        let trimmed = line.trim();
        let mut out = trimmed;
        for prefix in &["///", "//!", "//", "*", "#", "--"] {
            if let Some(rest) = out.strip_prefix(prefix) {
                out = rest.trim();
                break;
            }
        }
        cleaned.push(out);
    }

    cleaned.join("\n").trim().to_string()
}

fn find_parent_name(node: Node, source: &str) -> Option<String> {
    let mut current = node.parent()?;

    loop {
        let k = current.kind();
        let is_container = matches!(
            k,
            "class_definition"
                | "class_declaration"
                | "impl_item"
                | "trait_item"
                | "module"
                | "namespace_definition"
                | "module_definition"
                | "module_declaration"
                | "struct_item"
                | "enum_item"
                | "interface_declaration"
                | "object"
        );

        if is_container {
            for i in 0..current.child_count() as u32 {
                if let Some(child) = current.child(i) {
                    let ck = child.kind();
                    if ck == "identifier" || ck == "type_identifier" || ck == "name" || ck == "constant" || ck == "scope_resolution" {
                        if let Ok(text) = child.utf8_text(source.as_bytes()) {
                            return Some(text.to_string());
                        }
                    }
                }
            }
            return current
                .utf8_text(source.as_bytes())
                .ok()
                .and_then(|s| s.lines().next())
                .map(|s| s.trim().to_string());
        }

        current = current.parent()?;
    }
}

fn sanitize_parent_name(parent: Option<String>) -> Option<String> {
    let trimmed = parent?.trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with("from ") || trimmed.starts_with("import ") {
        return None;
    }
    if trimmed.len() > 80 {
        return None;
    }
    Some(trimmed)
}

#[cfg(test)]
mod tests {
    use super::{extract_file_compact, sanitize_parent_name};
    use crate::lang;

    #[test]
    fn sanitize_parent_name_drops_import_like_parents() {
        assert_eq!(sanitize_parent_name(Some("import re".to_string())), None);
        assert_eq!(sanitize_parent_name(Some("from os import path".to_string())), None);
    }

    #[test]
    fn sanitize_parent_name_keeps_real_container_names() {
        assert_eq!(
            sanitize_parent_name(Some("CollectionManager".to_string())),
            Some("CollectionManager".to_string())
        );
        assert_eq!(
            sanitize_parent_name(Some(" QuickContext ".to_string())),
            Some("QuickContext".to_string())
        );
    }

    #[test]
    fn extract_file_compact_omits_source_heavy_fields() {
        let spec = lang::python::spec();
        let source = r#"
class CodeSearcher:
    def search_hybrid(self, query):
        return query
"#;

        let result = extract_file_compact("fixture.py", source, &spec);
        assert_eq!(result.symbols.len(), 2);

        for symbol in result.symbols {
            assert!(symbol.source.is_empty());
            assert!(symbol.docstring.is_none());
            assert!(symbol.params.is_none());
            assert!(symbol.return_type.is_none());
            assert!(symbol.signature.is_some());
        }
    }
}

fn extract_visibility(source: &str, def_node: &Node, language: &str) -> Option<String> {
    let start = def_node.start_byte();
    let line_start = source[..start].rfind('\n').map_or(0, |i| i + 1);
    let header = &source[line_start..start];
    let lower = header.to_ascii_lowercase();

    let candidates: &[&str] = match language {
        "rust" => &["pub", "pub(crate)", "pub(super)", "pub(in"],
        "java" | "csharp" | "cpp" | "php" => &["public", "private", "protected", "internal"],
        "typescript" | "javascript" => &["export", "public", "private", "protected"],
        _ => &["public", "private", "protected", "internal", "export", "pub"],
    };

    for v in candidates {
        if lower.contains(v) {
            return Some(v.to_string());
        }
    }
    None
}

fn extract_fallback_name(source: &str, kind: SymbolKind) -> String {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return format!("{:?}", kind).to_ascii_lowercase();
    }

    let mut chars = trimmed.chars().peekable();
    let mut name = String::new();

    while let Some(ch) = chars.peek().copied() {
        if ch.is_ascii_alphabetic() || ch == '_' {
            break;
        }
        chars.next();
    }

    while let Some(ch) = chars.peek().copied() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == ':' {
            name.push(ch);
            chars.next();
        } else {
            break;
        }
    }

    if name.is_empty() {
        format!("{:?}", kind).to_ascii_lowercase()
    } else {
        name.trim_end_matches(['(', ':']).to_string()
    }
}
