use serde::{Deserialize, Serialize};


fn default_true() -> bool {
    true
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolRole {
    Definition,
    Entrypoint,
    Orchestration,
    Logic,
    Utility,
    Test,
    Configuration,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Trait,
    Module,
    Import,
    Constant,
    Variable,
    Property,
    Constructor,
    Decorator,
    TypeAlias,
    HtmlTag,
    CssSelector,
    Heading,
    DataKey,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub language: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docstring: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<SymbolRole>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub file_path: String,
    pub language: String,
    pub symbols: Vec<ExtractedSymbol>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_mtime: Option<u64>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileScanEntry {
    pub file_path: String,
    pub language: String,
    pub file_size: u64,
    pub file_mtime: u64,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepMatch {
    pub file_path: String,
    pub line_number: usize,
    pub column_start: usize,
    pub column_end: usize,
    pub line: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_before: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_after: Vec<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepResult {
    pub matches: Vec<GrepMatch>,
    pub searched_files: usize,
    pub duration_ms: u128,
    pub truncated: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolLookupRequest {
    pub query: String,
    pub path: Option<String>,
    pub respect_gitignore: bool,
    pub limit: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolLookupItem {
    pub name: String,
    pub kind: String,
    pub language: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub parent: Option<String>,
    pub signature: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolLookupResult {
    pub project_root: String,
    pub results: Vec<SymbolLookupItem>,
    pub indexed_files: usize,
    pub indexed_symbols: usize,
    pub from_cache: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallerLookupRequest {
    pub symbol: String,
    pub path: Option<String>,
    pub respect_gitignore: bool,
    pub limit: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallerItem {
    pub callee_name: String,
    pub caller_name: String,
    pub caller_kind: String,
    pub caller_language: String,
    pub caller_file_path: String,
    pub caller_line: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallerLookupResult {
    pub project_root: String,
    pub symbol: String,
    pub callers: Vec<CallerItem>,
    pub indexed_files: usize,
    pub indexed_symbols: usize,
    pub from_cache: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportEdge {
    pub source_file: String,
    pub target_file: String,
    pub import_statement: String,
    pub module_path: String,
    pub language: String,
    pub line: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportGraphResult {
    pub file_path: String,
    pub project_root: String,
    pub edges: Vec<ImportEdge>,
    pub total_files: usize,
    pub total_imports: usize,
    pub duration_ms: u128,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportNeighborsResult {
    pub file_path: String,
    pub project_root: String,
    pub imports: Vec<ImportEdge>,
    pub importers: Vec<ImportEdge>,
    pub total_files: usize,
    pub total_imports: usize,
    pub duration_ms: u128,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchMatch {
    pub file_path: String,
    pub score: f64,
    pub matched_terms: Vec<String>,
    pub snippet: String,
    pub snippet_line_start: usize,
    pub snippet_line_end: usize,
    pub language: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchResult {
    pub matches: Vec<TextSearchMatch>,
    pub searched_files: usize,
    pub query_expr: String,
    pub duration_ms: u128,
    pub truncated: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolInputField {
    pub name: String,
    pub required: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOutputField {
    pub path: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolEvidence {
    pub kind: String,
    pub line: usize,
    pub text: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolContract {
    pub file_path: String,
    pub operation: String,
    pub transport: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_name: Option<String>,
    pub input_fields: Vec<ProtocolInputField>,
    pub output_fields: Vec<ProtocolOutputField>,
    pub evidence: Vec<ProtocolEvidence>,
    pub matched_terms: Vec<String>,
    pub score: f64,
    pub confidence: f64,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolSearchResult {
    pub contracts: Vec<ProtocolContract>,
    pub searched_files: usize,
    pub duration_ms: u128,
    pub truncated: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchCapture {
    pub name: String,
    pub text: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchItem {
    pub file_path: String,
    pub language: String,
    pub matched_text: String,
    pub line_start: usize,
    pub column_start: usize,
    pub line_end: usize,
    pub column_end: usize,
    pub captures: Vec<PatternMatchCapture>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchResult {
    pub matches: Vec<PatternMatchItem>,
    pub searched_files: usize,
    pub duration_ms: u128,
    pub truncated: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteEdit {
    pub line_start: usize,
    pub column_start: usize,
    pub line_end: usize,
    pub column_end: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub original: String,
    pub replacement: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteFileResult {
    pub file_path: String,
    pub edits: Vec<RewriteEdit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewritten_source: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteResult {
    pub files: Vec<RewriteFileResult>,
    pub searched_files: usize,
    pub total_edits: usize,
    pub dry_run: bool,
    pub duration_ms: u128,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileReadLine {
    pub line_number: usize,
    pub text: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileReadResult {
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub total_lines: usize,
    pub truncated: bool,
    pub duration_ms: u128,
    pub lines: Vec<FileReadLine>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileLineEdit {
    pub start_line: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEditResult {
    pub file_path: String,
    pub applied: bool,
    pub dry_run: bool,
    pub before_hash: String,
    pub after_hash: String,
    pub duration_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edit_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_text: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEditRevertResult {
    pub file_path: String,
    pub reverted: bool,
    pub before_hash: String,
    pub after_hash: String,
    pub duration_ms: u128,
    pub edit_id: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEditRequest {
    pub file_path: String,
    pub symbol_name: String,
    pub new_source: String,
    #[serde(default)]
    pub dry_run: bool,
    pub expected_hash: Option<String>,
    #[serde(default = "default_true")]
    pub record_undo: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEditResult {
    pub file_path: String,
    pub symbol_name: String,
    pub applied: bool,
    pub dry_run: bool,
    pub before_hash: String,
    pub after_hash: String,
    pub line_start: usize,
    pub line_end: usize,
    pub duration_ms: u128,
    pub edit_id: Option<String>,
    pub updated_text: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEditUndoRecord {
    pub edit_id: String,
    pub file_path: String,
    pub before_hash: String,
    pub after_hash: String,
    pub original_text: String,
    pub updated_text: String,
    pub created_at_ms: u128,
    pub reverted: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub language: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<SymbolRole>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactExtractionResult {
    pub file_path: String,
    pub language: String,
    pub symbols: Vec<CompactSymbol>,
    pub symbol_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_mtime: Option<u64>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractStats {
    pub total_files: usize,
    pub total_symbols: usize,
    pub languages: std::collections::HashMap<String, usize>,
    pub duration_ms: u128,
}


impl CompactSymbol {
    pub fn from_extracted(s: &ExtractedSymbol) -> Self {
        Self {
            name: s.name.clone(),
            kind: s.kind,
            language: s.language.clone(),
            file_path: s.file_path.clone(),
            line_start: s.line_start,
            line_end: s.line_end,
            signature: s.signature.clone(),
            parent: s.parent.clone(),
            visibility: s.visibility.clone(),
            role: s.role,
        }
    }
}


impl CompactExtractionResult {
    pub fn from_full(r: &ExtractionResult) -> Self {
        let symbols: Vec<CompactSymbol> = r.symbols.iter().map(CompactSymbol::from_extracted).collect();
        let symbol_count = symbols.len();
        Self {
            file_path: r.file_path.clone(),
            language: r.language.clone(),
            symbols,
            symbol_count,
            file_hash: r.file_hash.clone(),
            file_size: r.file_size,
            file_mtime: r.file_mtime,
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceDirection {
    Upstream,
    Downstream,
    Both,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceNode {
    pub name: String,
    pub kind: String,
    pub language: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub depth: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEdge {
    pub from_name: String,
    pub to_name: String,
    pub call_line: usize,
    pub file_path: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraphTraceResult {
    pub root_symbol: String,
    pub direction: TraceDirection,
    pub max_depth: usize,
    pub nodes: Vec<TraceNode>,
    pub edges: Vec<TraceEdge>,
    pub cycles_detected: Vec<String>,
    pub indexed_files: usize,
    pub indexed_symbols: usize,
    pub from_cache: bool,
}
