use serde::{Deserialize, Serialize};

use crate::types::FileLineEdit;


#[derive(Debug, Deserialize)]
#[serde(tag = "method")]
pub enum Request {
    #[serde(rename = "extract")]
    Extract {
        path: String,
        #[serde(default)]
        compact: Option<bool>,
        #[serde(default)]
        stats_only: Option<bool>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
    },
    #[serde(rename = "extract_symbol")]
    ExtractSymbol {
        file: String,
        symbol: String,
    },
    #[serde(rename = "scan_files")]
    ScanFiles {
        path: String,
        #[serde(default)]
        respect_gitignore: Option<bool>,
    },
    #[serde(rename = "grep")]
    Grep {
        query: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        before_context: Option<usize>,
        #[serde(default)]
        after_context: Option<usize>,
    },
    #[serde(rename = "symbol_lookup")]
    SymbolLookup {
        query: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        intent_mode: Option<bool>,
        #[serde(default)]
        intent_level: Option<u8>,
    },
    #[serde(rename = "find_callers")]
    FindCallers {
        symbol: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        limit: Option<usize>,
    },
    #[serde(rename = "trace_call_graph")]
    TraceCallGraph {
        symbol: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        direction: Option<String>,
        #[serde(default)]
        max_depth: Option<usize>,
    },
    #[serde(rename = "skeleton")]
    Skeleton {
        path: String,
        #[serde(default)]
        max_depth: Option<usize>,
        #[serde(default)]
        include_signatures: Option<bool>,
        #[serde(default)]
        include_line_numbers: Option<bool>,
        #[serde(default)]
        collapse_threshold: Option<usize>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        format: Option<String>,
    },
    #[serde(rename = "import_graph")]
    ImportGraph {
        file: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
    },
    #[serde(rename = "import_neighbors")]
    ImportNeighbors {
        file: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
    },
    #[serde(rename = "find_importers")]
    FindImporters {
        file: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
    },
    #[serde(rename = "text_search")]
    TextSearch {
        query: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        intent_mode: Option<bool>,
        #[serde(default)]
        intent_level: Option<u8>,
    },
    #[serde(rename = "warm_project")]
    WarmProject {
        path: String,
        #[serde(default)]
        respect_gitignore: Option<bool>,
    },
    #[serde(rename = "protocol_search")]
    ProtocolSearch {
        query: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        context_radius: Option<usize>,
        #[serde(default)]
        min_score: Option<f64>,
        #[serde(default)]
        include_markers: Option<Vec<String>>,
        #[serde(default)]
        exclude_markers: Option<Vec<String>>,
        #[serde(default)]
        max_input_fields: Option<usize>,
        #[serde(default)]
        max_output_fields: Option<usize>,
    },
    #[serde(rename = "pattern_search")]
    PatternSearch {
        pattern: String,
        language: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        limit: Option<usize>,
    },
    #[serde(rename = "pattern_rewrite")]
    PatternRewrite {
        pattern: String,
        replacement: String,
        language: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        respect_gitignore: Option<bool>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        dry_run: Option<bool>,
    },
    #[serde(rename = "lsp_definition")]
    LspDefinition {
        file: String,
        line: u32,
        character: u32,
    },
    #[serde(rename = "lsp_references")]
    LspReferences {
        file: String,
        line: u32,
        character: u32,
        #[serde(default)]
        include_declaration: Option<bool>,
    },
    #[serde(rename = "lsp_hover")]
    LspHover {
        file: String,
        line: u32,
        character: u32,
    },
    #[serde(rename = "lsp_symbols")]
    LspSymbols {
        file: String,
    },
    #[serde(rename = "lsp_format")]
    LspFormat {
        file: String,
        #[serde(default)]
        tab_size: Option<u32>,
        #[serde(default)]
        insert_spaces: Option<bool>,
    },
    #[serde(rename = "lsp_diagnostics")]
    LspDiagnostics {
        file: String,
    },
    #[serde(rename = "lsp_completion")]
    LspCompletion {
        file: String,
        line: u32,
        character: u32,
    },
    #[serde(rename = "lsp_rename")]
    LspRename {
        file: String,
        line: u32,
        character: u32,
        new_name: String,
    },
    #[serde(rename = "lsp_prepare_rename")]
    LspPrepareRename {
        file: String,
        line: u32,
        character: u32,
    },
    #[serde(rename = "lsp_code_actions")]
    LspCodeActions {
        file: String,
        start_line: u32,
        start_character: u32,
        end_line: u32,
        end_character: u32,
        #[serde(default)]
        diagnostics: Option<Vec<serde_json::Value>>,
    },
    #[serde(rename = "lsp_signature_help")]
    LspSignatureHelp {
        file: String,
        line: u32,
        character: u32,
    },
    #[serde(rename = "lsp_workspace_symbols")]
    LspWorkspaceSymbols {
        query: String,
        #[serde(default)]
        file: Option<String>,
    },
    #[serde(rename = "file_read")]
    FileRead {
        file: String,
        #[serde(default)]
        start_line: Option<usize>,
        #[serde(default)]
        end_line: Option<usize>,
        #[serde(default)]
        max_bytes: Option<usize>,
    },
    #[serde(rename = "file_edit")]
    FileEdit {
        file: String,
        mode: String,
        #[serde(default)]
        edits: Option<Vec<FileLineEdit>>,
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        dry_run: Option<bool>,
        #[serde(default)]
        expected_hash: Option<String>,
        #[serde(default)]
        record_undo: Option<bool>,
    },
    #[serde(rename = "file_edit_revert")]
    FileEditRevert {
        edit_id: String,
        #[serde(default)]
        dry_run: Option<bool>,
        #[serde(default)]
        expected_hash: Option<String>,
    },
    #[serde(rename = "symbol_edit")]
    SymbolEdit {
        file: String,
        symbol_name: String,
        new_source: String,
        #[serde(default)]
        dry_run: Option<bool>,
        #[serde(default)]
        expected_hash: Option<String>,
        #[serde(default)]
        record_undo: Option<bool>,
    },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "shutdown")]
    Shutdown,
}


#[derive(Debug, Serialize)]
#[serde(tag = "status")]
pub enum Response {
    #[serde(rename = "ok")]
    Ok {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<serde_json::Value>,
    },
    #[serde(rename = "error")]
    Error { message: String },
}


impl Response {
    pub fn ok_data(data: serde_json::Value) -> Self {
        Self::Ok { data: Some(data) }
    }

    pub fn ok_empty() -> Self {
        Self::Ok { data: None }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error {
            message: msg.into(),
        }
    }
}
