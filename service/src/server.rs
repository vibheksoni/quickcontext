use std::path::Path;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
#[cfg(unix)]
use tokio::net::UnixListener;
#[cfg(windows)]
use tokio::net::windows::named_pipe::{PipeMode, ServerOptions};
use tokio_util::sync::CancellationToken;

use crate::extract::{
    extract_compact_streaming, extract_file_compact, extract_path, extract_path_with_options,
    scan_supported_files, sha256_hex, ExtractOptions,
};
use crate::file_ops;
use crate::symbol_edit;
use crate::grep::grep_path;
use crate::import_graph;
use crate::lang::{self, LanguageSpec};
use crate::lsp;
use crate::pattern_match;
use crate::pattern_rewrite;
use crate::protocol::{Request, Response};
use crate::protocol_search::{self, ProtocolSearchOptions};
use crate::skeleton::{self, SkeletonOptions};
use crate::symbol_index::{file_symbols, find_callers, symbol_lookup, trace_call_graph, warm_project as warm_symbol_project};
use crate::types::TraceDirection;
use crate::text_search;


#[cfg(windows)]
const PIPE_NAME: &str = r"\\.\pipe\quickcontext";
#[cfg(unix)]
const SOCKET_PATH_ENV_VAR: &str = "QC_SOCKET_PATH";
#[cfg(unix)]
const DEFAULT_SOCKET_BASENAME: &str = "quickcontext.sock";
#[cfg(windows)]
const MAX_INSTANCES: usize = 16;
#[cfg(windows)]
const BUFFER_SIZE: u32 = 65536;
const DEFAULT_GREP_LIMIT: usize = 200;
const DEFAULT_SYMBOL_LIMIT: usize = 50;
const DEFAULT_CALLER_LIMIT: usize = 100;
const DEFAULT_TEXT_SEARCH_LIMIT: usize = 20;
const DEFAULT_PATTERN_SEARCH_LIMIT: usize = 50;
const DEFAULT_PATTERN_REWRITE_LIMIT: usize = 50;
const DEFAULT_TRACE_DEPTH: usize = 5;
const DEFAULT_INTENT_MODE: bool = false;
const DEFAULT_INTENT_LEVEL: u8 = 2;


pub async fn run() -> std::io::Result<()> {
    #[cfg(windows)]
    {
        return run_windows().await;
    }

    #[cfg(unix)]
    {
        return run_unix().await;
    }

    #[allow(unreachable_code)]
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "unsupported platform for quickcontext service transport",
    ))
}


fn install_ctrl_c_handler(cancel: &CancellationToken) {
    let ctrl_c_cancel = cancel.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("[quickcontext] ctrl+c received, shutting down");
        ctrl_c_cancel.cancel();
    });
}


fn should_log_client_error(err: &std::io::Error) -> bool {
    !matches!(
        err.kind(),
        std::io::ErrorKind::UnexpectedEof
            | std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::ConnectionReset
    )
}


#[cfg(unix)]
fn default_socket_path() -> std::path::PathBuf {
    if let Ok(path) = std::env::var(SOCKET_PATH_ENV_VAR) {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return std::path::PathBuf::from(trimmed);
        }
    }

    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        let trimmed = runtime_dir.trim();
        if !trimmed.is_empty() {
            return std::path::PathBuf::from(trimmed).join(DEFAULT_SOCKET_BASENAME);
        }
    }

    let user = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "quickcontext".to_string());
    let safe_user: String = user
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();

    std::env::temp_dir().join(format!("quickcontext-{safe_user}.sock"))
}


#[cfg(unix)]
struct SocketCleanupGuard {
    path: std::path::PathBuf,
}


#[cfg(unix)]
impl Drop for SocketCleanupGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}


#[cfg(windows)]
async fn run_windows() -> std::io::Result<()> {
    let cancel = CancellationToken::new();
    let specs = lang::registry();

    let mut server = ServerOptions::new()
        .first_pipe_instance(true)
        .pipe_mode(PipeMode::Byte)
        .in_buffer_size(BUFFER_SIZE)
        .out_buffer_size(BUFFER_SIZE)
        .max_instances(MAX_INSTANCES)
        .create(PIPE_NAME)?;

    eprintln!("[quickcontext] listening on {PIPE_NAME}");
    install_ctrl_c_handler(&cancel);

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                eprintln!("[quickcontext] server stopped");
                break;
            }
            result = server.connect() => {
                result?;
                let client = server;

                server = ServerOptions::new()
                    .pipe_mode(PipeMode::Byte)
                    .in_buffer_size(BUFFER_SIZE)
                    .out_buffer_size(BUFFER_SIZE)
                    .max_instances(MAX_INSTANCES)
                    .create(PIPE_NAME)?;

                let cancel = cancel.clone();

                tokio::spawn(async move {
                    if let Err(e) = handle_client(client, specs, &cancel).await {
                        if should_log_client_error(&e) {
                            eprintln!("[quickcontext] client error: {e}");
                        }
                    }
                });
            }
        }
    }

    {
        let mgr = lsp::manager::global();
        let mut mgr = mgr.lock().await;
        mgr.shutdown_all().await;
    }

    Ok(())
}


#[cfg(unix)]
async fn run_unix() -> std::io::Result<()> {
    let cancel = CancellationToken::new();
    let specs = lang::registry();
    let socket_path = default_socket_path();

    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    let listener = UnixListener::bind(&socket_path)?;
    let _cleanup = SocketCleanupGuard {
        path: socket_path.clone(),
    };

    eprintln!("[quickcontext] listening on {}", socket_path.display());
    install_ctrl_c_handler(&cancel);

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                eprintln!("[quickcontext] server stopped");
                break;
            }
            result = listener.accept() => {
                let (stream, _) = result?;
                let cancel = cancel.clone();

                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, specs, &cancel).await {
                        if should_log_client_error(&e) {
                            eprintln!("[quickcontext] client error: {e}");
                        }
                    }
                });
            }
        }
    }

    {
        let mgr = lsp::manager::global();
        let mut mgr = mgr.lock().await;
        mgr.shutdown_all().await;
    }

    Ok(())
}


async fn handle_client(
    mut pipe: impl AsyncRead + AsyncWrite + Unpin,
    specs: &[LanguageSpec],
    cancel: &CancellationToken,
) -> std::io::Result<()> {
    loop {
        let request = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = read_frame(&mut pipe) => {
                match result {
                    Ok(bytes) => bytes,
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => break,
                    Err(e) => return Err(e),
                }
            }
        };

        let response = match serde_json::from_slice::<Request>(&request) {
            Ok(req) => dispatch(req, specs, cancel).await,
            Err(e) => Response::error(format!("invalid request: {e}")),
        };

        let payload = serde_json::to_vec(&response)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        write_frame(&mut pipe, &payload).await?;
    }

    Ok(())
}


async fn dispatch(req: Request, specs: &[LanguageSpec], cancel: &CancellationToken) -> Response {
    match req {
        Request::Extract {
            path,
            compact,
            stats_only,
            respect_gitignore,
        } => handle_extract(&path, compact, stats_only, respect_gitignore, specs),
        Request::ExtractSymbol { file, symbol } => handle_extract_symbol(&file, &symbol, specs),
        Request::ScanFiles { path, respect_gitignore } => {
            handle_scan_files(&path, respect_gitignore.unwrap_or(true), specs)
        }
        Request::Grep {
            query,
            path,
            respect_gitignore,
            limit,
            before_context,
            after_context,
        } => handle_grep(
            &query,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            limit.unwrap_or(DEFAULT_GREP_LIMIT),
            before_context.unwrap_or(0),
            after_context.unwrap_or(0),
        ),
        Request::SymbolLookup {
            query,
            path,
            respect_gitignore,
            limit,
            intent_mode,
            intent_level,
        } => handle_symbol_lookup(
            &query,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            limit.unwrap_or(DEFAULT_SYMBOL_LIMIT),
            intent_mode.unwrap_or(DEFAULT_INTENT_MODE),
            intent_level.unwrap_or(DEFAULT_INTENT_LEVEL),
            specs,
        ),
        Request::FileSymbols {
            file,
            path,
            respect_gitignore,
            limit,
        } => handle_file_symbols(
            &file,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            limit.unwrap_or(DEFAULT_SYMBOL_LIMIT),
            specs,
        ),
        Request::FindCallers {
            symbol,
            path,
            respect_gitignore,
            limit,
        } => handle_find_callers(
            &symbol,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            limit.unwrap_or(DEFAULT_CALLER_LIMIT),
            specs,
        ),
        Request::TraceCallGraph {
            symbol,
            path,
            respect_gitignore,
            direction,
            max_depth,
        } => handle_trace_call_graph(
            &symbol,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            direction.as_deref(),
            max_depth.unwrap_or(DEFAULT_TRACE_DEPTH),
            specs,
        ),
        Request::Skeleton {
            path,
            max_depth,
            include_signatures,
            include_line_numbers,
            collapse_threshold,
            respect_gitignore,
            format,
        } => handle_skeleton(
            &path,
            max_depth,
            include_signatures,
            include_line_numbers,
            collapse_threshold,
            respect_gitignore,
            format.as_deref(),
            specs,
        ),
        Request::ImportGraph {
            file,
            path,
            respect_gitignore,
        } => handle_import_graph(
            &file,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            specs,
        ),
        Request::ImportNeighbors {
            file,
            path,
            respect_gitignore,
        } => handle_import_neighbors(
            &file,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            specs,
        ),
        Request::FindImporters {
            file,
            path,
            respect_gitignore,
        } => handle_find_importers(
            &file,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            specs,
        ),
        Request::TextSearch {
            query,
            path,
            respect_gitignore,
            limit,
            intent_mode,
            intent_level,
        } => handle_text_search(
            &query,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            limit.unwrap_or(DEFAULT_TEXT_SEARCH_LIMIT),
            intent_mode.unwrap_or(DEFAULT_INTENT_MODE),
            intent_level.unwrap_or(DEFAULT_INTENT_LEVEL),
            specs,
        ),
        Request::WarmProject {
            path,
            respect_gitignore,
        } => handle_warm_project(
            &path,
            respect_gitignore.unwrap_or(true),
            specs,
        ),
        Request::ProtocolSearch {
            query,
            path,
            respect_gitignore,
            limit,
            context_radius,
            min_score,
            include_markers,
            exclude_markers,
            max_input_fields,
            max_output_fields,
        } => handle_protocol_search(
            &query,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            limit.unwrap_or(DEFAULT_TEXT_SEARCH_LIMIT),
            ProtocolSearchOptions {
                context_radius,
                min_score,
                include_markers,
                exclude_markers,
                max_input_fields,
                max_output_fields,
            },
            specs,
        ),
        Request::PatternSearch {
            pattern,
            language,
            path,
            respect_gitignore,
            limit,
        } => handle_pattern_search(
            &pattern,
            &language,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            limit.unwrap_or(DEFAULT_PATTERN_SEARCH_LIMIT),
            specs,
        ),
        Request::PatternRewrite {
            pattern,
            replacement,
            language,
            path,
            respect_gitignore,
            limit,
            dry_run,
        } => handle_pattern_rewrite(
            &pattern,
            &replacement,
            &language,
            path.as_deref(),
            respect_gitignore.unwrap_or(true),
            limit.unwrap_or(DEFAULT_PATTERN_REWRITE_LIMIT),
            dry_run.unwrap_or(true),
            specs,
        ),
        Request::LspDefinition { file, line, character } => {
            handle_lsp_definition(&file, line, character).await
        }
        Request::LspReferences { file, line, character, include_declaration } => {
            handle_lsp_references(&file, line, character, include_declaration.unwrap_or(true)).await
        }
        Request::LspHover { file, line, character } => {
            handle_lsp_hover(&file, line, character).await
        }
        Request::LspSymbols { file } => {
            handle_lsp_symbols(&file).await
        }
        Request::LspFormat { file, tab_size, insert_spaces } => {
            handle_lsp_format(&file, tab_size.unwrap_or(4), insert_spaces.unwrap_or(true)).await
        }
        Request::LspDiagnostics { file } => {
            handle_lsp_diagnostics(&file).await
        }
        Request::LspCompletion { file, line, character } => {
            handle_lsp_completion(&file, line, character).await
        }
        Request::LspRename { file, line, character, new_name } => {
            handle_lsp_rename(&file, line, character, &new_name).await
        }
        Request::LspPrepareRename { file, line, character } => {
            handle_lsp_prepare_rename(&file, line, character).await
        }
        Request::LspCodeActions { file, start_line, start_character, end_line, end_character, diagnostics } => {
            handle_lsp_code_actions(&file, start_line, start_character, end_line, end_character, diagnostics.as_deref()).await
        }
        Request::LspSignatureHelp { file, line, character } => {
            handle_lsp_signature_help(&file, line, character).await
        }
        Request::LspWorkspaceSymbols { query, file } => {
            handle_lsp_workspace_symbols(&query, file.as_deref()).await
        }
        Request::FileRead {
            file,
            start_line,
            end_line,
            max_bytes,
        } => handle_file_read(&file, start_line, end_line, max_bytes),
        Request::FileEdit {
            file,
            mode,
            edits,
            text,
            dry_run,
            expected_hash,
            record_undo,
        } => handle_file_edit(
            &file,
            &mode,
            edits,
            text,
            dry_run.unwrap_or(false),
            expected_hash.as_deref(),
            record_undo.unwrap_or(true),
        ),
        Request::FileEditRevert {
            edit_id,
            dry_run,
            expected_hash,
        } => handle_file_edit_revert(
            &edit_id,
            dry_run.unwrap_or(false),
            expected_hash.as_deref(),
        ),
        Request::SymbolEdit {
            file,
            symbol_name,
            new_source,
            dry_run,
            expected_hash,
            record_undo,
        } => handle_symbol_edit(
            &file,
            &symbol_name,
            &new_source,
            dry_run.unwrap_or(false),
            expected_hash.as_deref(),
            record_undo.unwrap_or(true),
        ),
        Request::Ping => Response::ok_data(serde_json::json!("pong")),
        Request::LspSessions => handle_lsp_sessions().await,
        Request::LspShutdownAll => handle_lsp_shutdown_all().await,
        Request::Shutdown => {
            eprintln!("[quickcontext] shutdown requested");
            cancel.cancel();
            Response::ok_empty()
        }
    }
}


async fn handle_lsp_definition(file: &str, line: u32, character: u32) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.goto_definition(file, line, character).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_references(
    file: &str,
    line: u32,
    character: u32,
    include_declaration: bool,
) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.find_references(file, line, character, include_declaration).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_hover(file: &str, line: u32, character: u32) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.hover(file, line, character).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_symbols(file: &str) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.document_symbols(file).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_format(file: &str, tab_size: u32, insert_spaces: bool) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.format_document(file, tab_size, insert_spaces).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_diagnostics(file: &str) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => return Response::error(format!("read file: {e}")),
    };

    match client.request_diagnostics(file, &text).await {
        Ok(diag) => Response::ok_data(diag),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_completion(file: &str, line: u32, character: u32) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.completion(file, line, character).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_rename(file: &str, line: u32, character: u32, new_name: &str) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.rename(file, line, character, new_name).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_prepare_rename(file: &str, line: u32, character: u32) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.prepare_rename(file, line, character).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_code_actions(
    file: &str,
    start_line: u32,
    start_character: u32,
    end_line: u32,
    end_character: u32,
    diagnostics: Option<&[serde_json::Value]>,
) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    let diags = diagnostics.unwrap_or(&[]);
    match client.code_actions(file, start_line, start_character, end_line, end_character, diags).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_signature_help(file: &str, line: u32, character: u32) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let client = match mgr.get_client(file).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    if client.spec().needs_did_open {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => return Response::error(format!("read file: {e}")),
        };
        let _ = client.did_open(file, &text).await;
    }

    match client.signature_help(file, line, character).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_workspace_symbols(query: &str, file: Option<&str>) -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;

    let target = match file {
        Some(f) => f,
        None => return Response::error("file parameter required for workspace symbols (used to identify the language server)"),
    };

    let client = match mgr.get_client(target).await {
        Ok(c) => c,
        Err(e) => return Response::error(e),
    };

    match client.workspace_symbols(query).await {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(e),
    }
}


async fn handle_lsp_sessions() -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;
    let sessions = mgr.active_sessions();
    let rows: Vec<serde_json::Value> = sessions
        .into_iter()
        .map(|session| serde_json::json!({
            "server_name": session.server_name,
            "language_id": session.language_id,
            "project_root": session.project_root,
            "pid": session.pid,
            "initialized": session.initialized,
            "alive": session.alive,
        }))
        .collect();
    Response::ok_data(serde_json::json!({ "sessions": rows }))
}


async fn handle_lsp_shutdown_all() -> Response {
    let mgr = lsp::manager::global();
    let mut mgr = mgr.lock().await;
    let session_count = mgr.active_sessions().len();
    mgr.shutdown_all().await;
    Response::ok_data(serde_json::json!({
        "stopped_sessions": session_count,
    }))
}


fn handle_extract(
    path_str: &str,
    compact: Option<bool>,
    stats_only: Option<bool>,
    respect_gitignore: Option<bool>,
    specs: &[LanguageSpec],
) -> Response {
    let path = Path::new(path_str);
    let options = ExtractOptions {
        respect_gitignore: respect_gitignore.unwrap_or(true),
    };
    let is_compact = compact.unwrap_or(false);
    let is_stats_only = stats_only.unwrap_or(false);

    if (is_compact || is_stats_only) && path.is_dir() {
        return handle_extract_compact(path, specs, options, is_stats_only);
    }

    if (is_compact || is_stats_only) && path.is_file() {
        let path_str = path.to_string_lossy();
        let spec = match lang::detect_language(&path_str, specs) {
            Some(spec) => spec,
            None => return Response::error(format!("unsupported file type: {}", path.display())),
        };
        let source = match std::fs::read_to_string(path) {
            Ok(source) => source,
            Err(e) => return Response::error(format!("failed to read file: {e}")),
        };
        let meta = std::fs::metadata(path).ok();
        let mut compact_result = crate::types::CompactExtractionResult::from_full(&extract_file_compact(
            &path_str,
            &source,
            spec,
        ));
        compact_result.file_hash = Some(sha256_hex(source.as_bytes()));
        compact_result.file_size = meta.as_ref().map(|m| m.len());
        compact_result.file_mtime = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        if is_stats_only {
            let stats = crate::types::ExtractStats {
                total_files: 1,
                total_symbols: compact_result.symbol_count,
                languages: std::collections::HashMap::from([(spec.name.to_string(), compact_result.symbol_count)]),
                duration_ms: 0,
            };
            return match serde_json::to_value(&stats) {
                Ok(val) => Response::ok_data(val),
                Err(e) => Response::error(format!("serialization failed: {e}")),
            };
        }

        return match serde_json::to_value(serde_json::json!({
            "results": [compact_result],
            "stats": {
                "total_files": 1,
                "total_symbols": compact_result.symbol_count,
                "languages": std::collections::HashMap::from([(spec.name.to_string(), compact_result.symbol_count)]),
                "duration_ms": 0,
            },
        })) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        };
    }

    match extract_path_with_options(path, specs, options) {
        Ok(results) => {
            if is_stats_only {
                let stats = build_extract_stats(&results);
                return match serde_json::to_value(&stats) {
                    Ok(val) => Response::ok_data(val),
                    Err(e) => Response::error(format!("serialization failed: {e}")),
                };
            }

            if is_compact {
                let compact_results: Vec<crate::types::CompactExtractionResult> =
                    results.iter().map(crate::types::CompactExtractionResult::from_full).collect();
                let stats = build_extract_stats(&results);
                return match serde_json::to_value(serde_json::json!({
                    "results": compact_results,
                    "stats": stats,
                })) {
                    Ok(val) => Response::ok_data(val),
                    Err(e) => Response::error(format!("serialization failed: {e}")),
                };
            }

            match serde_json::to_value(&results) {
                Ok(val) => Response::ok_data(val),
                Err(e) => Response::error(format!("serialization failed: {e}")),
            }
        }
        Err(e) => Response::error(e),
    }
}


fn build_extract_stats(results: &[crate::types::ExtractionResult]) -> crate::types::ExtractStats {
    let mut languages: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut total_symbols = 0usize;

    for result in results {
        total_symbols += result.symbols.len();
        let entry = languages.entry(result.language.clone()).or_insert(0);
        *entry += result.symbols.len();
    }

    crate::types::ExtractStats {
        total_files: results.len(),
        total_symbols,
        languages,
        duration_ms: 0,
    }
}


fn handle_extract_compact(
    path: &Path,
    specs: &[LanguageSpec],
    options: ExtractOptions,
    stats_only: bool,
) -> Response {
    if stats_only {
        let mut sink = std::io::sink();
        let stats = extract_compact_streaming(path, specs, options, &mut sink);
        match serde_json::to_value(&stats) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        }
    } else {
        let mut buf = Vec::new();
        let stats = extract_compact_streaming(path, specs, options, &mut buf);
        let jsonl = String::from_utf8_lossy(&buf);
        let results: Vec<serde_json::Value> = jsonl
            .lines()
            .filter(|l| !l.is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let output = serde_json::json!({
            "results": results,
            "stats": stats,
        });
        Response::ok_data(output)
    }
}


fn handle_extract_symbol(file_str: &str, symbol_query: &str, specs: &[LanguageSpec]) -> Response {
    let path = Path::new(file_str);
    if !path.is_file() {
        return Response::error(format!("not a file: {file_str}"));
    }

    let results = match extract_path(path, specs) {
        Ok(r) => r,
        Err(e) => return Response::error(e),
    };

    let extraction = match results.into_iter().next() {
        Some(r) => r,
        None => return Response::error(format!("no extraction result for: {file_str}")),
    };

    let (parent_filter, name_filter) = if let Some(dot_pos) = symbol_query.rfind('.') {
        (Some(&symbol_query[..dot_pos]), &symbol_query[dot_pos + 1..])
    } else {
        (None, symbol_query)
    };

    let name_lower = name_filter.to_lowercase();
    let parent_lower = parent_filter.map(|p| p.to_lowercase());

    let matched: Vec<_> = extraction
        .symbols
        .into_iter()
        .filter(|s| {
            if s.name.to_lowercase() != name_lower {
                return false;
            }
            if let Some(ref pf) = parent_lower {
                match &s.parent {
                    Some(p) => p.to_lowercase() == *pf,
                    None => false,
                }
            } else {
                true
            }
        })
        .collect();

    if matched.is_empty() {
        return Response::error(format!("symbol not found: {symbol_query} in {file_str}"));
    }

    match serde_json::to_value(&matched) {
        Ok(val) => Response::ok_data(serde_json::json!({
            "file_path": file_str,
            "language": extraction.language,
            "query": symbol_query,
            "symbols": val,
            "total_matches": matched.len(),
        })),
        Err(e) => Response::error(format!("serialization failed: {e}")),
    }
}

fn handle_scan_files(path_str: &str, respect_gitignore: bool, specs: &[LanguageSpec]) -> Response {
    let path = Path::new(path_str);
    let entries = scan_supported_files(
        path,
        specs,
        ExtractOptions {
            respect_gitignore,
        },
    );

    match serde_json::to_value(entries) {
        Ok(val) => Response::ok_data(val),
        Err(e) => Response::error(format!("serialization failed: {e}")),
    }
}


fn handle_grep(query: &str, path_str: Option<&str>, respect_gitignore: bool, limit: usize, before_context: usize, after_context: usize) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);

    match grep_path(query, path, respect_gitignore, limit, before_context, after_context) {
        Ok(results) => match serde_json::to_value(&results) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_symbol_lookup(
    query: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    limit: usize,
    intent_mode: bool,
    intent_level: u8,
    specs: &[LanguageSpec],
) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);

    match symbol_lookup(
        query,
        path,
        specs,
        respect_gitignore,
        limit,
        intent_mode,
        intent_level,
    ) {
        Ok(results) => match serde_json::to_value(&results) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_file_symbols(
    file: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    limit: usize,
    specs: &[LanguageSpec],
) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);
    let file_path = Path::new(file);

    match file_symbols(
        file_path,
        path,
        specs,
        respect_gitignore,
        limit,
    ) {
        Ok(results) => match serde_json::to_value(&results) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_find_callers(
    symbol: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    limit: usize,
    specs: &[LanguageSpec],
) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);

    match find_callers(symbol, path, specs, respect_gitignore, limit) {
        Ok(results) => match serde_json::to_value(&results) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_trace_call_graph(
    symbol: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    direction_str: Option<&str>,
    max_depth: usize,
    specs: &[LanguageSpec],
) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);

    let direction = match direction_str.unwrap_or("both") {
        "upstream" => TraceDirection::Upstream,
        "downstream" => TraceDirection::Downstream,
        "both" => TraceDirection::Both,
        other => return Response::error(format!("invalid direction: {other} (expected upstream, downstream, or both)")),
    };

    match trace_call_graph(symbol, path, specs, respect_gitignore, direction, max_depth) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_skeleton(
    path_str: &str,
    max_depth: Option<usize>,
    include_signatures: Option<bool>,
    include_line_numbers: Option<bool>,
    collapse_threshold: Option<usize>,
    respect_gitignore: Option<bool>,
    format: Option<&str>,
    specs: &[LanguageSpec],
) -> Response {
    let path = Path::new(path_str);

    let options = SkeletonOptions {
        max_depth: max_depth.unwrap_or(20),
        include_signatures: include_signatures.unwrap_or(true),
        include_line_numbers: include_line_numbers.unwrap_or(false),
        collapse_threshold: collapse_threshold.unwrap_or(0),
        respect_gitignore: respect_gitignore.unwrap_or(true),
    };

    match skeleton::build_skeleton(path, specs, &options) {
        Ok(result) => {
            let output_format = format.unwrap_or("json");
            if output_format == "markdown" {
                let md = skeleton::render_markdown(&result, &options);
                Response::ok_data(serde_json::json!({
                    "markdown": md,
                    "total_files": result.total_files,
                    "total_symbols": result.total_symbols,
                    "total_directories": result.total_directories,
                    "duration_ms": result.duration_ms,
                }))
            } else {
                match serde_json::to_value(&result) {
                    Ok(val) => Response::ok_data(val),
                    Err(e) => Response::error(format!("serialization failed: {e}")),
                }
            }
        }
        Err(e) => Response::error(e),
    }
}


fn handle_import_graph(
    file_str: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    specs: &[LanguageSpec],
) -> Response {
    let file = Path::new(file_str);
    let project_root = path_str.unwrap_or(".");
    let root = Path::new(project_root);

    match import_graph::import_graph(file, root, specs, respect_gitignore) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_import_neighbors(
    file_str: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    specs: &[LanguageSpec],
) -> Response {
    let file = Path::new(file_str);
    let project_root = path_str.unwrap_or(".");
    let root = Path::new(project_root);

    match import_graph::import_neighbors(file, root, specs, respect_gitignore) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_find_importers(
    file_str: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    specs: &[LanguageSpec],
) -> Response {
    let file = Path::new(file_str);
    let project_root = path_str.unwrap_or(".");
    let root = Path::new(project_root);

    match import_graph::find_importers(file, root, specs, respect_gitignore) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


/// Handle text_search IPC request using BM25 ranking.
///
/// query: &str — Raw query string with operators and filters.
/// path_str: Option<&str> — Directory to search (defaults to ".").
/// respect_gitignore: bool — Honor .gitignore rules.
/// limit: usize — Max results to return.
/// specs: &[LanguageSpec] — Language specs for file detection.
fn handle_text_search(
    query: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    limit: usize,
    intent_mode: bool,
    intent_level: u8,
    specs: &[LanguageSpec],
) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);

    match text_search::text_search(
        query,
        path,
        respect_gitignore,
        limit,
        specs,
        intent_mode,
        intent_level,
    ) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}

fn handle_warm_project(
    path_str: &str,
    respect_gitignore: bool,
    specs: &[LanguageSpec],
) -> Response {
    let path = Path::new(path_str);
    let warm_result: Result<(usize, usize), String> = std::thread::scope(|scope| {
        let symbol_job = scope.spawn(|| warm_symbol_project(path, specs, respect_gitignore));
        let text_job = scope.spawn(|| crate::text_index::warm_index(path, specs, respect_gitignore));

        let symbol_count = symbol_job
            .join()
            .map_err(|_| "symbol warm thread panicked".to_string())??;
        let text_doc_count = text_job
            .join()
            .map_err(|_| "text warm thread panicked".to_string())??;
        Ok((symbol_count, text_doc_count))
    });
    let (symbol_count, text_doc_count) = match warm_result {
        Ok(value) => value,
        Err(e) => return Response::error(e),
    };

    Response::ok_data(serde_json::json!({
        "path": path_str,
        "symbol_count": symbol_count,
        "text_doc_count": text_doc_count,
        "respect_gitignore": respect_gitignore,
    }))
}


fn handle_protocol_search(
    query: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    limit: usize,
    options: ProtocolSearchOptions,
    specs: &[LanguageSpec],
) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);

    match protocol_search::protocol_search(query, path, respect_gitignore, limit, specs, Some(&options)) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


/// Handle pattern_search IPC request using AST pattern matching.
///
/// pattern: &str — Code pattern with metavariables ($NAME, $$$, $_).
/// language: &str — Language name to match against.
/// path_str: Option<&str> — Directory or file to search (defaults to ".").
/// respect_gitignore: bool — Honor .gitignore rules.
/// limit: usize — Max matches to return.
/// specs: &[LanguageSpec] — Language specs for file detection and parsing.
fn handle_pattern_search(
    pattern: &str,
    language: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    limit: usize,
    specs: &[LanguageSpec],
) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);

    match pattern_match::pattern_search(pattern, language, path, respect_gitignore, limit, specs) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


/// Handle pattern_rewrite IPC request using AST pattern matching + template substitution.
///
/// pattern: &str — Code pattern with metavariables ($NAME, $$$, $_).
/// replacement: &str — Replacement template with metavariable substitution.
/// language: &str — Language name to match against.
/// path_str: Option<&str> — Directory or file to search (defaults to ".").
/// respect_gitignore: bool — Honor .gitignore rules.
/// limit: usize — Max files to rewrite.
/// dry_run: bool — When true, compute edits but do not write files.
/// specs: &[LanguageSpec] — Language specs for file detection and parsing.
fn handle_pattern_rewrite(
    pattern: &str,
    replacement: &str,
    language: &str,
    path_str: Option<&str>,
    respect_gitignore: bool,
    limit: usize,
    dry_run: bool,
    specs: &[LanguageSpec],
) -> Response {
    let target = path_str.unwrap_or(".");
    let path = Path::new(target);

    match pattern_rewrite::pattern_rewrite(
        pattern,
        replacement,
        language,
        path,
        respect_gitignore,
        limit,
        dry_run,
        specs,
    ) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_file_read(
    file: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
    max_bytes: Option<usize>,
) -> Response {
    let path = Path::new(file);
    match file_ops::file_read(path, start_line, end_line, max_bytes) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_file_edit(
    file: &str,
    mode: &str,
    edits: Option<Vec<crate::types::FileLineEdit>>,
    text: Option<String>,
    dry_run: bool,
    expected_hash: Option<&str>,
    record_undo: bool,
) -> Response {
    let path = Path::new(file);
    match file_ops::file_edit(path, mode, edits, text, dry_run, expected_hash, record_undo) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


fn handle_file_edit_revert(edit_id: &str, dry_run: bool, expected_hash: Option<&str>) -> Response {
    match file_ops::file_edit_revert(edit_id, dry_run, expected_hash) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}

fn handle_symbol_edit(
    file: &str,
    symbol_name: &str,
    new_source: &str,
    dry_run: bool,
    expected_hash: Option<&str>,
    record_undo: bool,
) -> Response {
    let path = Path::new(file);
    match symbol_edit::symbol_edit(path, symbol_name, new_source, dry_run, expected_hash, record_undo) {
        Ok(result) => match serde_json::to_value(&result) {
            Ok(val) => Response::ok_data(val),
            Err(e) => Response::error(format!("serialization failed: {e}")),
        },
        Err(e) => Response::error(e),
    }
}


async fn read_frame(pipe: &mut (impl AsyncRead + Unpin)) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    pipe.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;

    if len > 256 * 1024 * 1024 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {len} bytes"),
        ));
    }

    let mut buf = vec![0u8; len];
    pipe.read_exact(&mut buf).await?;
    Ok(buf)
}


async fn write_frame(pipe: &mut (impl AsyncWrite + Unpin), data: &[u8]) -> std::io::Result<()> {
    let len = data.len() as u32;
    pipe.write_all(&len.to_le_bytes()).await?;
    pipe.write_all(data).await?;
    pipe.flush().await?;
    Ok(())
}
