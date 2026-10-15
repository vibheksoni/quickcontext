use std::collections::HashMap;

use crate::types::{ExtractedSymbol, SymbolKind, SymbolRole};


const TEST_PREFIXES: &[&str] = &["test_", "test.", "spec_", "spec."];
const TEST_SUFFIXES: &[&str] = &["_test", "_spec", "_tests", "_specs"];
const TEST_EXACT: &[&str] = &["tests", "test", "spec", "specs"];

const CONFIG_PREFIXES: &[&str] = &[
    "config", "configure", "setup", "init", "initialize",
    "create_app", "create_engine", "build_config", "load_config",
    "register", "bootstrap",
];

const CONFIG_SUFFIXES: &[&str] = &[
    "_config", "_configuration", "_settings", "_options",
    "_setup", "_init", "_builder",
];

const ENTRYPOINT_NAMES: &[&str] = &[
    "main", "run", "start", "serve", "execute", "cli",
    "app", "handler", "entrypoint", "lambda_handler",
];

const ORCHESTRATION_NAMES: &[&str] = &[
    "dispatch", "route", "handle", "process", "orchestrate",
    "coordinate", "pipeline", "workflow", "schedule", "run_all",
    "execute_all", "batch",
];

const ORCHESTRATION_FAN_OUT_THRESHOLD: usize = 5;
const UTILITY_FAN_IN_THRESHOLD: usize = 4;


/// Optional call graph context for graph-aware classification.
///
/// fan_in: HashMap<String, usize> — Number of callers per symbol name (lowercase).
/// fan_out: HashMap<String, usize> — Number of callees per symbol name (lowercase).
pub struct CallGraphContext {
    pub fan_in: HashMap<String, usize>,
    pub fan_out: HashMap<String, usize>,
}


/// Classify roles for all symbols in a list, mutating in place.
///
/// symbols: &mut [ExtractedSymbol] — Symbols to classify.
/// graph: Option<&CallGraphContext> — Optional call graph for graph-aware heuristics.
pub fn classify_symbols(symbols: &mut [ExtractedSymbol], graph: Option<&CallGraphContext>) {
    for symbol in symbols.iter_mut() {
        symbol.role = Some(classify_one(symbol, graph));
    }
}


/// Classify a single symbol into a role using layered heuristics.
///
/// symbol: &ExtractedSymbol — Symbol to classify.
/// graph: Option<&CallGraphContext> — Optional call graph data.
fn classify_one(symbol: &ExtractedSymbol, graph: Option<&CallGraphContext>) -> SymbolRole {
    if is_definition_kind(symbol.kind) {
        return SymbolRole::Definition;
    }

    let name_lower = symbol.name.to_lowercase();

    if is_test_symbol(symbol, &name_lower) {
        return SymbolRole::Test;
    }

    if is_config_symbol(&name_lower, symbol) {
        return SymbolRole::Configuration;
    }

    if !is_callable_kind(symbol.kind) {
        return SymbolRole::Definition;
    }

    if let Some(role) = classify_by_graph(&name_lower, symbol, graph) {
        return role;
    }

    if is_entrypoint(symbol, &name_lower) {
        return SymbolRole::Entrypoint;
    }

    if is_orchestration_name(&name_lower) {
        return SymbolRole::Orchestration;
    }

    if is_utility(symbol) {
        return SymbolRole::Utility;
    }

    SymbolRole::Logic
}


/// Check if a SymbolKind is inherently a definition (no behavior).
///
/// kind: SymbolKind — The symbol kind to check.
fn is_definition_kind(kind: SymbolKind) -> bool {
    matches!(
        kind,
        SymbolKind::Struct
            | SymbolKind::Enum
            | SymbolKind::Interface
            | SymbolKind::Trait
            | SymbolKind::TypeAlias
            | SymbolKind::Constant
            | SymbolKind::HtmlTag
            | SymbolKind::CssSelector
            | SymbolKind::Heading
            | SymbolKind::DataKey
            | SymbolKind::Import
    )
}


/// Check if a SymbolKind represents callable behavior.
///
/// kind: SymbolKind — The symbol kind to check.
fn is_callable_kind(kind: SymbolKind) -> bool {
    matches!(
        kind,
        SymbolKind::Function
            | SymbolKind::Method
            | SymbolKind::Constructor
            | SymbolKind::Decorator
    )
}


/// Check if a symbol is a test function/class based on name and file path.
///
/// symbol: &ExtractedSymbol — Symbol to check.
/// name_lower: &str — Pre-lowercased symbol name.
fn is_test_symbol(symbol: &ExtractedSymbol, name_lower: &str) -> bool {
    if TEST_EXACT.contains(&name_lower) {
        return true;
    }

    for prefix in TEST_PREFIXES {
        if name_lower.starts_with(prefix) {
            return true;
        }
    }

    for suffix in TEST_SUFFIXES {
        if name_lower.ends_with(suffix) {
            return true;
        }
    }

    let path_lower = symbol.file_path.to_lowercase();
    let path_lower = path_lower.replace('\\', "/");
    if path_lower.contains("/test/")
        || path_lower.contains("/tests/")
        || path_lower.contains("/spec/")
        || path_lower.contains("/specs/")
        || path_lower.contains("/__tests__/")
    {
        return true;
    }

    false
}


/// Check if a symbol is configuration/setup based on name patterns.
///
/// name_lower: &str — Pre-lowercased symbol name.
/// symbol: &ExtractedSymbol — Symbol for additional context.
fn is_config_symbol(name_lower: &str, symbol: &ExtractedSymbol) -> bool {
    for prefix in CONFIG_PREFIXES {
        if name_lower.starts_with(prefix) {
            return true;
        }
    }

    for suffix in CONFIG_SUFFIXES {
        if name_lower.ends_with(suffix) {
            return true;
        }
    }

    let path_lower = symbol.file_path.to_lowercase();
    let path_lower = path_lower.replace('\\', "/");
    let file_name = path_lower.rsplit('/').next().unwrap_or("");
    matches!(
        file_name,
        "config.py"
            | "config.rs"
            | "config.ts"
            | "config.js"
            | "config.go"
            | "settings.py"
            | "setup.py"
            | "setup.cfg"
            | "conftest.py"
    )
}


/// Classify a symbol using call graph fan-in/fan-out data.
///
/// name_lower: &str — Pre-lowercased symbol name.
/// symbol: &ExtractedSymbol — Symbol for additional context.
/// graph: Option<&CallGraphContext> — Call graph data (None = skip graph heuristics).
fn classify_by_graph(
    name_lower: &str,
    _symbol: &ExtractedSymbol,
    graph: Option<&CallGraphContext>,
) -> Option<SymbolRole> {
    let graph = graph?;

    let fan_out = graph.fan_out.get(name_lower).copied().unwrap_or(0);
    let fan_in = graph.fan_in.get(name_lower).copied().unwrap_or(0);

    if fan_out >= ORCHESTRATION_FAN_OUT_THRESHOLD && fan_in <= 2 {
        return Some(SymbolRole::Orchestration);
    }

    if fan_in >= UTILITY_FAN_IN_THRESHOLD && fan_out <= 1 {
        return Some(SymbolRole::Utility);
    }

    if fan_in == 0 && fan_out > 0 {
        return Some(SymbolRole::Entrypoint);
    }

    None
}


/// Check if a symbol is an entrypoint (main, run, serve, etc.).
///
/// symbol: &ExtractedSymbol — Symbol for visibility/parent context.
/// name_lower: &str — Pre-lowercased symbol name.
fn is_entrypoint(symbol: &ExtractedSymbol, name_lower: &str) -> bool {
    if !ENTRYPOINT_NAMES.contains(&name_lower) {
        return false;
    }

    if symbol.parent.is_some() {
        return false;
    }

    symbol
        .visibility
        .as_deref()
        .is_none_or(|v| v.contains("pub") || v.contains("export"))
}


/// Check if a symbol name matches orchestration patterns.
///
/// name_lower: &str — Pre-lowercased symbol name.
fn is_orchestration_name(name_lower: &str) -> bool {
    for name in ORCHESTRATION_NAMES {
        if name_lower == *name || name_lower.starts_with(&format!("{name}_")) {
            return true;
        }
    }
    false
}


/// Check if a symbol is a utility (private, small, helper-like).
///
/// symbol: &ExtractedSymbol — Symbol to check.
fn is_utility(symbol: &ExtractedSymbol) -> bool {
    let is_private = symbol.visibility.as_deref().is_some_and(|v| {
        v.contains("private") || v.contains("protected")
    });

    if is_private {
        return true;
    }

    let name_lower = symbol.name.to_lowercase();
    if name_lower.starts_with('_') && !name_lower.starts_with("__") {
        return true;
    }

    let helper_suffixes = ["_helper", "_util", "_internal", "_impl", "_aux"];
    for suffix in &helper_suffixes {
        if name_lower.ends_with(suffix) {
            return true;
        }
    }

    false
}


#[cfg(test)]
mod tests {
    use super::*;

    fn make_symbol(name: &str, kind: SymbolKind) -> ExtractedSymbol {
        ExtractedSymbol {
            name: name.to_string(),
            kind,
            language: "python".to_string(),
            file_path: "src/main.py".to_string(),
            line_start: 0,
            line_end: 10,
            byte_start: 0,
            byte_end: 100,
            source: String::new(),
            signature: None,
            docstring: None,
            params: None,
            return_type: None,
            parent: None,
            visibility: None,
            role: None,
        }
    }

    #[test]
    fn test_struct_is_definition() {
        let sym = make_symbol("MyStruct", SymbolKind::Struct);
        assert_eq!(classify_one(&sym, None), SymbolRole::Definition);
    }

    #[test]
    fn test_enum_is_definition() {
        let sym = make_symbol("Color", SymbolKind::Enum);
        assert_eq!(classify_one(&sym, None), SymbolRole::Definition);
    }

    #[test]
    fn test_import_is_definition() {
        let sym = make_symbol("os", SymbolKind::Import);
        assert_eq!(classify_one(&sym, None), SymbolRole::Definition);
    }

    #[test]
    fn test_test_prefix() {
        let sym = make_symbol("test_login", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Test);
    }

    #[test]
    fn test_test_suffix() {
        let sym = make_symbol("login_test", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Test);
    }

    #[test]
    fn test_test_by_path() {
        let mut sym = make_symbol("helper", SymbolKind::Function);
        sym.file_path = "project/tests/test_auth.py".to_string();
        assert_eq!(classify_one(&sym, None), SymbolRole::Test);
    }

    #[test]
    fn test_config_prefix() {
        let sym = make_symbol("configure_logging", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Configuration);
    }

    #[test]
    fn test_config_suffix() {
        let sym = make_symbol("database_config", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Configuration);
    }

    #[test]
    fn test_config_by_filename() {
        let mut sym = make_symbol("get_value", SymbolKind::Function);
        sym.file_path = "src/config.py".to_string();
        assert_eq!(classify_one(&sym, None), SymbolRole::Configuration);
    }

    #[test]
    fn test_entrypoint_main() {
        let sym = make_symbol("main", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Entrypoint);
    }

    #[test]
    fn test_entrypoint_not_method() {
        let mut sym = make_symbol("run", SymbolKind::Method);
        sym.parent = Some("Server".to_string());
        assert_eq!(classify_one(&sym, None), SymbolRole::Logic);
    }

    #[test]
    fn test_orchestration_name() {
        let sym = make_symbol("dispatch_events", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Orchestration);
    }

    #[test]
    fn test_utility_private() {
        let mut sym = make_symbol("do_work", SymbolKind::Method);
        sym.visibility = Some("private".to_string());
        assert_eq!(classify_one(&sym, None), SymbolRole::Utility);
    }

    #[test]
    fn test_utility_underscore() {
        let sym = make_symbol("_parse_line", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Utility);
    }

    #[test]
    fn test_utility_helper_suffix() {
        let sym = make_symbol("format_helper", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Utility);
    }

    #[test]
    fn test_logic_default() {
        let sym = make_symbol("calculate_total", SymbolKind::Function);
        assert_eq!(classify_one(&sym, None), SymbolRole::Logic);
    }

    #[test]
    fn test_graph_orchestration() {
        let sym = make_symbol("build_pipeline", SymbolKind::Function);
        let mut fan_out = HashMap::new();
        fan_out.insert("build_pipeline".to_string(), 7);
        let graph = CallGraphContext {
            fan_in: HashMap::new(),
            fan_out,
        };
        assert_eq!(classify_one(&sym, Some(&graph)), SymbolRole::Orchestration);
    }

    #[test]
    fn test_graph_utility() {
        let sym = make_symbol("normalize", SymbolKind::Function);
        let mut fan_in = HashMap::new();
        fan_in.insert("normalize".to_string(), 6);
        let graph = CallGraphContext {
            fan_in,
            fan_out: HashMap::new(),
        };
        assert_eq!(classify_one(&sym, Some(&graph)), SymbolRole::Utility);
    }

    #[test]
    fn test_graph_entrypoint_zero_callers() {
        let sym = make_symbol("process_data", SymbolKind::Function);
        let mut fan_out = HashMap::new();
        fan_out.insert("process_data".to_string(), 2);
        let graph = CallGraphContext {
            fan_in: HashMap::new(),
            fan_out,
        };
        assert_eq!(classify_one(&sym, Some(&graph)), SymbolRole::Entrypoint);
    }

    #[test]
    fn test_classify_symbols_batch() {
        let mut symbols = vec![
            make_symbol("MyClass", SymbolKind::Struct),
            make_symbol("test_it", SymbolKind::Function),
            make_symbol("main", SymbolKind::Function),
            make_symbol("compute", SymbolKind::Function),
        ];
        classify_symbols(&mut symbols, None);
        assert_eq!(symbols[0].role, Some(SymbolRole::Definition));
        assert_eq!(symbols[1].role, Some(SymbolRole::Test));
        assert_eq!(symbols[2].role, Some(SymbolRole::Entrypoint));
        assert_eq!(symbols[3].role, Some(SymbolRole::Logic));
    }
}