use std::path::Path;


const SYMBOL_WEIGHT_FUNCTION: f64 = 2.0;
const SYMBOL_WEIGHT_CLASS: f64 = 1.8;
const SYMBOL_WEIGHT_STRUCT: f64 = 1.8;
const SYMBOL_WEIGHT_TRAIT: f64 = 1.6;
const SYMBOL_WEIGHT_ENUM: f64 = 1.6;
const SYMBOL_WEIGHT_INTERFACE: f64 = 1.6;
const SYMBOL_WEIGHT_METHOD: f64 = 2.0;
const SYMBOL_WEIGHT_MODULE: f64 = 1.4;
const SYMBOL_WEIGHT_CONSTANT: f64 = 1.3;
const SYMBOL_WEIGHT_PROPERTY: f64 = 1.3;
const SYMBOL_WEIGHT_CONSTRUCTOR: f64 = 1.5;
const SYMBOL_WEIGHT_TYPE_ALIAS: f64 = 1.3;
const SYMBOL_WEIGHT_DECORATOR: f64 = 1.1;
const SYMBOL_WEIGHT_VARIABLE: f64 = 1.0;
const SYMBOL_WEIGHT_IMPORT: f64 = 0.9;
const SYMBOL_WEIGHT_HTML_TAG: f64 = 0.8;
const SYMBOL_WEIGHT_CSS_SELECTOR: f64 = 0.8;
const SYMBOL_WEIGHT_HEADING: f64 = 0.7;
const SYMBOL_WEIGHT_DATA_KEY: f64 = 0.6;

const TEST_FILE_PENALTY: f64 = 0.6;
const CONFIG_FILE_PENALTY: f64 = 0.7;
const DOC_FILE_PENALTY: f64 = 0.65;
const GENERATED_FILE_PENALTY: f64 = 0.5;
const VENDOR_FILE_PENALTY: f64 = 0.4;

const COVERAGE_EXPONENT: f64 = 1.5;
const COVERAGE_MAX_BOOST: f64 = 2.0;

const ROLE_BOOST_ORCHESTRATION: f64 = 1.5;
const ROLE_BOOST_DEFINITION: f64 = 1.2;
const ROLE_BOOST_IMPLEMENTATION: f64 = 1.1;
const ROLE_BOOST_IMPORT: f64 = 0.9;

const COMPLEXITY_THRESHOLD: usize = 5;
const REFERENCE_THRESHOLD: usize = 5;


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileCategory {
    Source,
    Test,
    Config,
    Documentation,
    Generated,
    Vendor,
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileRole {
    Orchestration,
    Definition,
    Implementation,
    Import,
}


pub struct BoostResult {
    pub symbol_boost: f64,
    pub category_boost: f64,
    pub coverage_boost: f64,
    pub combined: f64,
}


/// Map a SymbolKind string to its structural weight.
///
/// kind: &str — Lowercase symbol kind name from extraction.
fn symbol_weight(kind: &str) -> f64 {
    match kind {
        "function" => SYMBOL_WEIGHT_FUNCTION,
        "method" => SYMBOL_WEIGHT_METHOD,
        "class" => SYMBOL_WEIGHT_CLASS,
        "struct" => SYMBOL_WEIGHT_STRUCT,
        "trait" => SYMBOL_WEIGHT_TRAIT,
        "enum" => SYMBOL_WEIGHT_ENUM,
        "interface" => SYMBOL_WEIGHT_INTERFACE,
        "constructor" => SYMBOL_WEIGHT_CONSTRUCTOR,
        "module" => SYMBOL_WEIGHT_MODULE,
        "constant" => SYMBOL_WEIGHT_CONSTANT,
        "property" => SYMBOL_WEIGHT_PROPERTY,
        "type_alias" => SYMBOL_WEIGHT_TYPE_ALIAS,
        "decorator" => SYMBOL_WEIGHT_DECORATOR,
        "variable" => SYMBOL_WEIGHT_VARIABLE,
        "import" => SYMBOL_WEIGHT_IMPORT,
        "html_tag" => SYMBOL_WEIGHT_HTML_TAG,
        "css_selector" => SYMBOL_WEIGHT_CSS_SELECTOR,
        "heading" => SYMBOL_WEIGHT_HEADING,
        "data_key" => SYMBOL_WEIGHT_DATA_KEY,
        _ => 1.0,
    }
}


/// Classify a file path into a category for scoring penalties.
///
/// path: &str — File path to classify.
pub fn classify_file(path: &str) -> FileCategory {
    let path_lower = path.to_lowercase().replace('\\', "/");
    let filename = Path::new(&path_lower)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    if is_vendor_path(&path_lower) {
        return FileCategory::Vendor;
    }

    if is_generated_file(&path_lower, &filename) {
        return FileCategory::Generated;
    }

    if is_test_file(&path_lower, &filename) {
        return FileCategory::Test;
    }

    if is_config_file(&filename) {
        return FileCategory::Config;
    }

    if is_doc_file(&path_lower, &filename) {
        return FileCategory::Documentation;
    }

    FileCategory::Source
}


/// Check if path is inside a vendor/dependency directory.
///
/// path_lower: &str — Lowercased, forward-slash normalized path.
fn is_vendor_path(path_lower: &str) -> bool {
    let segments = [
        "vendor/", "node_modules/", "third_party/", "third-party/",
        "external/", "deps/", ".cargo/registry/",
    ];
    segments.iter().any(|s| path_lower.contains(s))
}


/// Check if file appears to be auto-generated.
///
/// path_lower: &str — Lowercased, forward-slash normalized path.
/// filename: &str — Lowercased filename.
fn is_generated_file(path_lower: &str, filename: &str) -> bool {
    if path_lower.contains("generated/")
        || path_lower.contains("/gen/")
        || path_lower.contains("/app/immutable/")
        || path_lower.contains("/immutable/chunks/")
        || path_lower.contains("/immutable/nodes/")
        || path_lower.contains("/immutable/workers/")
    {
        return true;
    }
    let suffixes = [
        ".generated.", ".auto.", ".pb.go", ".pb.rs", "_generated.",
        ".g.dart", ".freezed.dart",
    ];
    suffixes.iter().any(|s| filename.contains(s))
}


/// Check if file is a test file based on path and naming conventions.
///
/// path_lower: &str — Lowercased, forward-slash normalized path.
/// filename: &str — Lowercased filename.
fn is_test_file(path_lower: &str, filename: &str) -> bool {
    let dir_markers = [
        "test/", "tests/", "spec/", "specs/",
        "__tests__/", "__test__/", "testing/",
    ];
    if dir_markers.iter().any(|d| path_lower.contains(d)) {
        return true;
    }
    let name_no_ext = filename.rsplit_once('.').map(|(n, _)| n).unwrap_or(filename);
    name_no_ext.starts_with("test_")
        || name_no_ext.ends_with("_test")
        || name_no_ext.ends_with(".test")
        || name_no_ext.ends_with(".spec")
        || name_no_ext.ends_with("_spec")
        || name_no_ext.starts_with("spec_")
}


/// Check if file is a configuration file.
///
/// filename: &str — Lowercased filename.
fn is_config_file(filename: &str) -> bool {
    let exact = [
        "cargo.toml", "package.json", "tsconfig.json", "pyproject.toml",
        "setup.py", "setup.cfg", "go.mod", "go.sum", "pom.xml",
        "build.gradle", "makefile", "cmakelists.txt", "dockerfile",
        ".eslintrc", ".prettierrc", ".babelrc", "jest.config.js",
        "webpack.config.js", "vite.config.ts", "rollup.config.js",
        ".gitignore", ".dockerignore", ".editorconfig",
        "requirements.txt", "poetry.lock", "yarn.lock", "package-lock.json",
        "cargo.lock", "gemfile", "gemfile.lock", "composer.json",
    ];
    if exact.iter().any(|e| filename == *e) {
        return true;
    }
    filename.starts_with('.') && !filename.contains('/')
}


/// Check if file is documentation.
///
/// path_lower: &str — Lowercased, forward-slash normalized path.
/// filename: &str — Lowercased filename.
fn is_doc_file(path_lower: &str, filename: &str) -> bool {
    let dir_markers = ["docs/", "doc/", "documentation/"];
    if dir_markers.iter().any(|d| path_lower.contains(d)) {
        return true;
    }
    let names = [
        "readme.md", "changelog.md", "contributing.md", "license",
        "license.md", "license.txt", "authors", "authors.md",
    ];
    names.iter().any(|n| filename == *n)
}


/// Compute symbol composition boost for a file based on extracted symbol kinds.
///
/// symbol_kinds: &[&str] — Lowercase symbol kind strings from tree-sitter extraction.
pub fn compute_symbol_boost(symbol_kinds: &[&str]) -> f64 {
    if symbol_kinds.is_empty() {
        return 1.0;
    }
    let total_weight: f64 = symbol_kinds.iter().map(|k| symbol_weight(k)).sum();
    total_weight / symbol_kinds.len() as f64
}


/// Compute file category penalty/boost multiplier.
///
/// category: FileCategory — Classification of the file.
pub fn compute_category_boost(category: FileCategory) -> f64 {
    match category {
        FileCategory::Source => 1.0,
        FileCategory::Test => TEST_FILE_PENALTY,
        FileCategory::Config => CONFIG_FILE_PENALTY,
        FileCategory::Documentation => DOC_FILE_PENALTY,
        FileCategory::Generated => GENERATED_FILE_PENALTY,
        FileCategory::Vendor => VENDOR_FILE_PENALTY,
    }
}


/// Compute coverage boost based on how many unique query terms appear in the document.
///
/// query_terms: &[String] — Tokenized query terms.
/// content: &str — File content to check against.
pub fn compute_coverage_boost(query_terms: &[String], content: &str) -> f64 {
    if query_terms.is_empty() {
        return 1.0;
    }
    let content_lower = content.to_lowercase();
    let matched = query_terms
        .iter()
        .filter(|t| content_lower.contains(t.as_str()))
        .count();
    let coverage = matched as f64 / query_terms.len() as f64;
    1.0 + coverage.powf(COVERAGE_EXPONENT) * COVERAGE_MAX_BOOST
}


/// Count control flow complexity indicators in source code.
///
/// content: &str — File source content.
pub fn count_complexity(content: &str) -> usize {
    let patterns = [
        "if ", "if(", "else if", "elif ", "else{", "else {",
        "for ", "for(", "while ", "while(",
        "match ", "switch ", "switch(",
        "catch ", "catch(", "except ", "except:",
        "? ", "&&", "||", "??",
    ];
    let content_lower = content.to_lowercase();
    patterns.iter().map(|p| content_lower.matches(p).count()).sum()
}


/// Count unique identifier references (call-like patterns) in source code.
///
/// content: &str — File source content.
pub fn count_references(content: &str) -> usize {
    let mut count = 0usize;
    let mut seen = std::collections::HashSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        for word in trimmed.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if word.len() >= 3 && word.chars().next().map_or(false, |c| c.is_alphabetic()) {
                if seen.insert(word.to_lowercase()) {
                    count += 1;
                }
            }
        }
    }
    count
}


/// Classify a file's role based on symbol composition, complexity, and references.
///
/// symbol_kinds: &[&str] — Symbol kinds from tree-sitter extraction.
/// content: &str — File source content for complexity/reference analysis.
pub fn classify_role(symbol_kinds: &[&str], content: &str) -> FileRole {
    if symbol_kinds.is_empty() {
        return FileRole::Implementation;
    }

    let import_count = symbol_kinds.iter().filter(|k| **k == "import").count();
    let import_ratio = import_count as f64 / symbol_kinds.len() as f64;
    if import_ratio > 0.6 {
        return FileRole::Import;
    }

    let definition_kinds = [
        "class", "struct", "enum", "interface", "trait", "type_alias",
    ];
    let def_count = symbol_kinds
        .iter()
        .filter(|k| definition_kinds.contains(k))
        .count();
    let def_ratio = def_count as f64 / symbol_kinds.len() as f64;
    if def_ratio > 0.3 {
        return FileRole::Definition;
    }

    let complexity = count_complexity(content);
    let references = count_references(content);
    if complexity >= COMPLEXITY_THRESHOLD && references >= REFERENCE_THRESHOLD {
        return FileRole::Orchestration;
    }

    FileRole::Implementation
}


/// Classify a file's role from content alone (no symbol kinds needed).
///
/// Uses complexity and reference heuristics only. Suitable for text_search
/// hot path where tree-sitter extraction is too expensive.
///
/// content: &str — File source content.
pub fn classify_role_from_content(content: &str) -> FileRole {
    let complexity = count_complexity(content);
    let references = count_references(content);
    if complexity >= COMPLEXITY_THRESHOLD && references >= REFERENCE_THRESHOLD {
        return FileRole::Orchestration;
    }
    FileRole::Implementation
}


/// Compute role-based boost multiplier.
///
/// role: FileRole — Classified role of the file.
pub fn compute_role_boost(role: FileRole) -> f64 {
    match role {
        FileRole::Orchestration => ROLE_BOOST_ORCHESTRATION,
        FileRole::Definition => ROLE_BOOST_DEFINITION,
        FileRole::Implementation => ROLE_BOOST_IMPLEMENTATION,
        FileRole::Import => ROLE_BOOST_IMPORT,
    }
}


/// Compute combined structural boost for a file.
///
/// path: &str — File path for category detection.
/// symbol_kinds: &[&str] — Symbol kinds extracted from the file.
/// query_terms: &[String] — Tokenized query terms.
/// content: &str — File content for coverage analysis.
pub fn compute_boost(
    path: &str,
    symbol_kinds: &[&str],
    query_terms: &[String],
    content: &str,
) -> BoostResult {
    let category = classify_file(path);
    let symbol_boost = compute_symbol_boost(symbol_kinds);
    let category_boost = compute_category_boost(category);
    let coverage_boost = compute_coverage_boost(query_terms, content);
    let combined = symbol_boost * category_boost * coverage_boost;

    BoostResult {
        symbol_boost,
        category_boost,
        coverage_boost,
        combined,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_weight_hierarchy() {
        assert_eq!(symbol_weight("function"), 2.0);
        assert_eq!(symbol_weight("method"), 2.0);
        assert!(symbol_weight("function") > symbol_weight("variable"));
        assert!(symbol_weight("class") > symbol_weight("import"));
        assert_eq!(symbol_weight("unknown_kind"), 1.0);
    }

    #[test]
    fn test_classify_file_source() {
        assert_eq!(classify_file("src/main.rs"), FileCategory::Source);
        assert_eq!(classify_file("engine/src/cli.py"), FileCategory::Source);
    }

    #[test]
    fn test_classify_file_test() {
        assert_eq!(classify_file("tests/test_auth.py"), FileCategory::Test);
        assert_eq!(classify_file("src/__tests__/handler.test.ts"), FileCategory::Test);
        assert_eq!(classify_file("src/test_utils.py"), FileCategory::Test);
    }

    #[test]
    fn test_classify_file_config() {
        assert_eq!(classify_file("Cargo.toml"), FileCategory::Config);
        assert_eq!(classify_file("package.json"), FileCategory::Config);
        assert_eq!(classify_file(".gitignore"), FileCategory::Config);
    }

    #[test]
    fn test_classify_file_doc() {
        assert_eq!(classify_file("README.md"), FileCategory::Documentation);
        assert_eq!(classify_file("docs/api.md"), FileCategory::Documentation);
    }

    #[test]
    fn test_classify_file_generated() {
        assert_eq!(classify_file("src/generated/schema.rs"), FileCategory::Generated);
        assert_eq!(classify_file("api.pb.go"), FileCategory::Generated);
    }

    #[test]
    fn test_classify_file_vendor() {
        assert_eq!(classify_file("vendor/lib/foo.go"), FileCategory::Vendor);
        assert_eq!(classify_file("node_modules/express/index.js"), FileCategory::Vendor);
    }

    #[test]
    fn test_compute_symbol_boost_functions() {
        let kinds = vec!["function", "method", "function"];
        assert!((compute_symbol_boost(&kinds) - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_symbol_boost_mixed() {
        let kinds = vec!["function", "import", "variable"];
        let boost = compute_symbol_boost(&kinds);
        assert!(boost > 1.0);
        assert!(boost < 2.0);
    }

    #[test]
    fn test_compute_symbol_boost_empty() {
        assert_eq!(compute_symbol_boost(&[]), 1.0);
    }

    #[test]
    fn test_category_boost_values() {
        assert_eq!(compute_category_boost(FileCategory::Source), 1.0);
        assert!(compute_category_boost(FileCategory::Test) < 1.0);
        assert!(compute_category_boost(FileCategory::Vendor) < compute_category_boost(FileCategory::Test));
    }

    #[test]
    fn test_coverage_boost_all_terms() {
        let terms = vec!["auth".to_string(), "token".to_string()];
        let content = "authentication token validation auth";
        let boost = compute_coverage_boost(&terms, content);
        assert!(boost > 2.0);
    }

    #[test]
    fn test_coverage_boost_no_terms() {
        let terms = vec!["auth".to_string()];
        let content = "database connection pool";
        let boost = compute_coverage_boost(&terms, content);
        assert!((boost - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_coverage_boost_empty_query() {
        let boost = compute_coverage_boost(&[], "some content");
        assert_eq!(boost, 1.0);
    }

    #[test]
    fn test_compute_boost_combined() {
        let result = compute_boost(
            "src/handler.rs",
            &["function", "method"],
            &["handler".to_string()],
            "pub fn handler() { process() }",
        );
        assert!(result.symbol_boost > 1.5);
        assert_eq!(result.category_boost, 1.0);
        assert!(result.coverage_boost > 1.0);
        assert!(result.combined > 1.5);
    }

    #[test]
    fn test_compute_boost_test_file_penalized() {
        let source = compute_boost(
            "src/handler.rs",
            &["function"],
            &["handler".to_string()],
            "fn handler() ",
        );
        let test = compute_boost(
            "tests/test_handler.rs",
            &["function"],
            &["handler".to_string()],
            "fn handler() {}",
        );
        assert!(source.combined > test.combined);
    }

    #[test]
    fn test_classify_role_import_heavy() {
        let kinds = vec!["import", "import", "import", "import", "function"];
        assert_eq!(classify_role(&kinds, ""), FileRole::Import);
    }

    #[test]
    fn test_classify_role_definition() {
        let kinds = vec!["struct", "enum", "function", "method"];
        assert_eq!(classify_role(&kinds, "let x = 1;"), FileRole::Definition);
    }

    #[test]
    fn test_classify_role_orchestration() {
        let kinds = vec!["function", "method", "variable"];
        let content = "if foo { for x in items { if bar && baz { match val { while true { if check || done { } } } } } }";
        assert_eq!(classify_role(&kinds, content), FileRole::Orchestration);
    }

    #[test]
    fn test_classify_role_implementation_default() {
        let kinds = vec!["function", "variable"];
        assert_eq!(classify_role(&kinds, "let x = 1;"), FileRole::Implementation);
    }

    #[test]
    fn test_role_boost_values() {
        assert!(compute_role_boost(FileRole::Orchestration) > compute_role_boost(FileRole::Definition));
        assert!(compute_role_boost(FileRole::Definition) > compute_role_boost(FileRole::Implementation));
        assert!(compute_role_boost(FileRole::Implementation) > compute_role_boost(FileRole::Import));
    }

    #[test]
    fn test_count_complexity_basic() {
        let code = "if x > 0 { for i in items { if y && z { match val { } } } }";
        let c = count_complexity(code);
        assert!(c >= 3);
    }

    #[test]
    fn test_count_complexity_empty() {
        assert_eq!(count_complexity(""), 0);
    }
}
