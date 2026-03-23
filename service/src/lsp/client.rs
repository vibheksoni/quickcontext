use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, oneshot};
use tokio::time::{timeout, Duration};

use super::jsonrpc::{Id, JsonRpcNotification, JsonRpcRequest};
use super::registry::LspServerSpec;
use super::transport::{self, DiagnosticsMap, PendingMap};
use super::types;


const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;


pub struct LspClient {
    spec: &'static LspServerSpec,
    project_root: String,
    stdin: Arc<Mutex<ChildStdin>>,
    pending: PendingMap,
    diagnostics: DiagnosticsMap,
    child: Child,
    initialized: bool,
}


impl LspClient {
    /// Spawn a language server process and start the stdout reader task.
    ///
    /// spec: &'static LspServerSpec — Server spec from registry.
    /// project_root: &str — Absolute path to the project root directory.
    pub async fn spawn(
        spec: &'static LspServerSpec,
        project_root: &str,
    ) -> Result<Self, String> {
        let command_path = super::registry::resolve_binary(spec.binary)
            .unwrap_or_else(|| spec.binary.to_string());
        let mut cmd = Command::new(&command_path);
        for arg in spec.args {
            cmd.arg(arg);
        }
        cmd.current_dir(project_root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);

        let mut child = cmd.spawn().map_err(|e| {
            format!("failed to spawn {} (resolved as {}): {e}", spec.binary, command_path)
        })?;

        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let diagnostics: DiagnosticsMap = Arc::new(Mutex::new(HashMap::new()));
        let stdin = Arc::new(Mutex::new(stdin));

        let reader_pending = Arc::clone(&pending);
        let reader_diagnostics = Arc::clone(&diagnostics);
        let server_name = spec.name.to_string();
        tokio::spawn(transport::stdout_reader_task(
            stdout,
            reader_pending,
            reader_diagnostics,
            server_name,
        ));

        eprintln!("[lsp:{}] spawned for {project_root}", spec.name);

        Ok(Self {
            spec,
            project_root: project_root.to_string(),
            stdin,
            pending,
            diagnostics,
            child,
            initialized: false,
        })
    }

    /// Send a JSON-RPC request and wait for the response.
    ///
    /// method: &str — LSP method name.
    /// params: Option<serde_json::Value> — Request parameters.
    pub async fn request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        let req = JsonRpcRequest::new(method, params);
        let id = match &req.id {
            Id::Number(n) => *n,
            _ => return Err("non-numeric request id".to_string()),
        };

        let (tx, rx) = oneshot::channel();
        {
            let mut map = self.pending.lock().await;
            map.insert(id, tx);
        }

        transport::send_request(&self.stdin, &req).await?;

        let msg = timeout(REQUEST_TIMEOUT, rx)
            .await
            .map_err(|_| format!("request timed out: {method}"))?
            .map_err(|_| "response channel dropped".to_string())?;

        if let Some(err) = msg.error {
            return Err(format!("{err}"));
        }

        Ok(msg.result.unwrap_or(serde_json::Value::Null))
    }

    /// Send a JSON-RPC notification (no response expected).
    ///
    /// method: &str — LSP notification method name.
    /// params: Option<serde_json::Value> — Notification parameters.
    pub async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), String> {
        let notif = JsonRpcNotification::new(method, params);
        transport::send_notification(&self.stdin, &notif).await
    }

    /// Perform the LSP initialize handshake.
    ///
    /// Sends `initialize` request with client capabilities and rootUri,
    /// waits for server capabilities response, then sends `initialized` notification.
    /// Returns the server capabilities JSON.
    pub async fn initialize(&mut self) -> Result<serde_json::Value, String> {
        if self.initialized {
            return Err("already initialized".to_string());
        }

        let root_uri = types::path_to_uri(&self.project_root);

        let params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "didOpen": true,
                        "didClose": true
                    },
                    "hover": {
                        "contentFormat": ["markdown", "plaintext"]
                    },
                    "definition": {
                        "linkSupport": false
                    },
                    "references": {},
                    "completion": {
                        "completionItem": {
                            "snippetSupport": false,
                            "documentationFormat": ["markdown", "plaintext"]
                        }
                    },
                    "rename": {
                        "prepareSupport": true
                    },
                    "codeAction": {
                        "codeActionLiteralSupport": {
                            "codeActionKind": {
                                "valueSet": [
                                    "quickfix",
                                    "refactor",
                                    "refactor.extract",
                                    "refactor.inline",
                                    "refactor.rewrite",
                                    "source",
                                    "source.organizeImports"
                                ]
                            }
                        }
                    },
                    "signatureHelp": {
                        "signatureInformation": {
                            "documentationFormat": ["markdown", "plaintext"]
                        }
                    },
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": true
                    },
                    "formatting": {},
                    "publishDiagnostics": {
                        "relatedInformation": false
                    }
                },
                "workspace": {
                    "workspaceFolders": true,
                    "symbol": {
                        "symbolKind": {
                            "valueSet": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
                        }
                    }
                }
            },
            "workspaceFolders": [{
                "uri": root_uri,
                "name": self.project_root
                    .rsplit(['/', '\\'])
                    .next()
                    .unwrap_or(&self.project_root)
            }]
        });

        let result = self.request("initialize", Some(params)).await?;

        self.notify("initialized", Some(json!({}))).await?;
        self.initialized = true;

        eprintln!("[lsp:{}] initialized for {}", self.spec.name, self.project_root);
        Ok(result)
    }

    /// Gracefully shut down the language server.
    ///
    /// Sends `shutdown` request, waits for acknowledgement, then sends `exit` notification.
    pub async fn shutdown(&mut self) -> Result<(), String> {
        if self.initialized {
            let _ = self.request("shutdown", None).await;
            let _ = self.notify("exit", None).await;
            self.initialized = false;

            eprintln!("[lsp:{}] shutdown", self.spec.name);
        }

        match timeout(SHUTDOWN_TIMEOUT, self.child.wait()).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(format!("failed waiting for {}: {e}", self.spec.name)),
            Err(_) => {
                let _ = self.child.start_kill();
                let _ = timeout(SHUTDOWN_TIMEOUT, self.child.wait()).await;
            }
        }
        Ok(())
    }

    /// Notify the server that a document was opened.
    ///
    /// file_path: &str — Absolute path to the file.
    /// text: &str — Full file contents.
    pub async fn did_open(&self, file_path: &str, text: &str) -> Result<(), String> {
        let language_id = self.spec.language_id;
        let uri = types::path_to_uri(file_path);

        let params = json!({
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": 1,
                "text": text
            }
        });

        self.notify("textDocument/didOpen", Some(params)).await
    }

    /// Notify the server that a document was closed.
    ///
    /// file_path: &str — Absolute path to the file.
    pub async fn did_close(&self, file_path: &str) -> Result<(), String> {
        let uri = types::path_to_uri(file_path);

        let params = json!({
            "textDocument": {
                "uri": uri
            }
        });

        self.notify("textDocument/didClose", Some(params)).await
    }

    /// Check if the language server process is still running.
    pub fn is_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(None) => true,
            _ => false,
        }
    }

    /// Get the language server spec.
    pub fn spec(&self) -> &'static LspServerSpec {
        self.spec
    }

    /// Get the project root path.
    pub fn project_root(&self) -> &str {
        &self.project_root
    }

    /// Whether the server has completed initialization.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the child process id for the language server.
    pub fn process_id(&self) -> Option<u32> {
        self.child.id()
    }

    /// Go to definition of the symbol at the given position.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// line: u32 — Zero-based line number.
    /// character: u32 — Zero-based character offset.
    pub async fn goto_definition(
        &self,
        file_path: &str,
        line: u32,
        character: u32,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character }
        });
        self.request("textDocument/definition", Some(params)).await
    }

    /// Find all references to the symbol at the given position.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// line: u32 — Zero-based line number.
    /// character: u32 — Zero-based character offset.
    /// include_declaration: bool — Whether to include the declaration itself.
    pub async fn find_references(
        &self,
        file_path: &str,
        line: u32,
        character: u32,
        include_declaration: bool,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character },
            "context": { "includeDeclaration": include_declaration }
        });
        self.request("textDocument/references", Some(params)).await
    }

    /// Get hover information for the symbol at the given position.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// line: u32 — Zero-based line number.
    /// character: u32 — Zero-based character offset.
    pub async fn hover(
        &self,
        file_path: &str,
        line: u32,
        character: u32,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character }
        });
        self.request("textDocument/hover", Some(params)).await
    }

    /// Get document symbols (outline) for a file.
    ///
    /// file_path: &str — Absolute path to the source file.
    pub async fn document_symbols(
        &self,
        file_path: &str,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri }
        });
        self.request("textDocument/documentSymbol", Some(params)).await
    }

    /// Format an entire document.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// tab_size: u32 — Number of spaces per tab.
    /// insert_spaces: bool — Use spaces instead of tabs.
    pub async fn format_document(
        &self,
        file_path: &str,
        tab_size: u32,
        insert_spaces: bool,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "options": {
                "tabSize": tab_size,
                "insertSpaces": insert_spaces
            }
        });
        self.request("textDocument/formatting", Some(params)).await
    }

    /// Request completions at the given position.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// line: u32 — Zero-based line number.
    /// character: u32 — Zero-based character offset.
    pub async fn completion(
        &self,
        file_path: &str,
        line: u32,
        character: u32,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character }
        });
        self.request("textDocument/completion", Some(params)).await
    }

    /// Rename the symbol at the given position.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// line: u32 — Zero-based line number.
    /// character: u32 — Zero-based character offset.
    /// new_name: &str — New name for the symbol.
    pub async fn rename(
        &self,
        file_path: &str,
        line: u32,
        character: u32,
        new_name: &str,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character },
            "newName": new_name
        });
        self.request("textDocument/rename", Some(params)).await
    }

    /// Prepare rename to validate and get the range of the symbol.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// line: u32 — Zero-based line number.
    /// character: u32 — Zero-based character offset.
    pub async fn prepare_rename(
        &self,
        file_path: &str,
        line: u32,
        character: u32,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character }
        });
        self.request("textDocument/prepareRename", Some(params)).await
    }

    /// Get code actions available at the given range.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// start_line: u32 — Zero-based start line.
    /// start_character: u32 — Zero-based start character.
    /// end_line: u32 — Zero-based end line.
    /// end_character: u32 — Zero-based end character.
    /// diagnostics: &[serde_json::Value] — Diagnostics to include in context.
    pub async fn code_actions(
        &self,
        file_path: &str,
        start_line: u32,
        start_character: u32,
        end_line: u32,
        end_character: u32,
        diagnostics: &[serde_json::Value],
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": start_line, "character": start_character },
                "end": { "line": end_line, "character": end_character }
            },
            "context": {
                "diagnostics": diagnostics
            }
        });
        self.request("textDocument/codeAction", Some(params)).await
    }

    /// Get signature help at the given position.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// line: u32 — Zero-based line number.
    /// character: u32 — Zero-based character offset.
    pub async fn signature_help(
        &self,
        file_path: &str,
        line: u32,
        character: u32,
    ) -> Result<serde_json::Value, String> {
        let uri = types::path_to_uri(file_path);
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character }
        });
        self.request("textDocument/signatureHelp", Some(params)).await
    }

    /// Search for symbols across the workspace.
    ///
    /// query: &str — Symbol search query string.
    pub async fn workspace_symbols(
        &self,
        query: &str,
    ) -> Result<serde_json::Value, String> {
        let params = json!({
            "query": query
        });
        self.request("workspace/symbol", Some(params)).await
    }

    /// Trigger diagnostic collection for a file.
    ///
    /// Opens the document to prompt the server to publish diagnostics,
    /// waits briefly for them to arrive, then returns whatever was collected.
    ///
    /// file_path: &str — Absolute path to the source file.
    /// text: &str — Full file contents.
    pub async fn request_diagnostics(
        &self,
        file_path: &str,
        text: &str,
    ) -> Result<serde_json::Value, String> {
        self.did_open(file_path, text).await?;
        tokio::time::sleep(Duration::from_millis(500)).await;
        let uri = types::path_to_uri(file_path);
        let map = self.diagnostics.lock().await;
        Ok(map.get(&uri).cloned().unwrap_or(serde_json::Value::Null))
    }

    /// Retrieve cached diagnostics for a file URI.
    ///
    /// Returns diagnostics previously received via publishDiagnostics
    /// notifications, or null if none exist.
    ///
    /// file_path: &str — Absolute path to the source file.
    pub async fn get_diagnostics(
        &self,
        file_path: &str,
    ) -> serde_json::Value {
        let uri = types::path_to_uri(file_path);
        let map = self.diagnostics.lock().await;
        map.get(&uri).cloned().unwrap_or(serde_json::Value::Null)
    }
}
