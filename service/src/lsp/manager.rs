use std::collections::HashMap;
use std::sync::OnceLock;

use tokio::sync::Mutex;

use super::client::LspClient;
use super::registry;


type ClientKey = (String, String);


#[derive(Debug, Clone)]
pub struct LspSessionInfo {
    pub server_name: String,
    pub language_id: String,
    pub project_root: String,
    pub pid: Option<u32>,
    pub initialized: bool,
    pub alive: bool,
}


pub struct LspManager {
    clients: HashMap<ClientKey, LspClient>,
}


static INSTANCE: OnceLock<Mutex<LspManager>> = OnceLock::new();


/// Get the global LSP manager singleton.
pub fn global() -> &'static Mutex<LspManager> {
    INSTANCE.get_or_init(|| Mutex::new(LspManager::new()))
}


impl LspManager {
    fn new() -> Self {
        Self {
            clients: HashMap::new(),
        }
    }

    /// Get or spawn a language server for the given file path.
    ///
    /// Detects language from extension, finds project root, checks if a server
    /// is already running for that (project, language) pair. If not, spawns one,
    /// initializes it, and caches it.
    ///
    /// file_path: &str — Absolute path to the source file.
    pub async fn get_client(&mut self, file_path: &str) -> Result<&mut LspClient, String> {
        let language_id = registry::detect_language_from_path(file_path)
            .ok_or_else(|| format!("unsupported file type: {file_path}"))?;

        let spec = registry::spec_for_language(language_id)
            .ok_or_else(|| format!("no LSP server for language: {language_id}"))?;

        let project_root = registry::find_project_root(
            std::path::Path::new(file_path),
            spec,
        ).ok_or_else(|| format!("no project root found for: {file_path}"))?;

        let key = (project_root.clone(), language_id.to_string());

        let mut needs_spawn = !self.clients.contains_key(&key);

        if !needs_spawn {
            let client = self.clients.get_mut(&key).unwrap();
            if !client.is_alive() {
                eprintln!("[lsp:{}] dead, respawning", spec.name);
                self.clients.remove(&key);
                needs_spawn = true;
            }
        }

        if needs_spawn {
            if !registry::find_binary(spec.binary) {
                return Err(format!("{} not found on PATH", spec.binary));
            }

            let mut client = LspClient::spawn(spec, &project_root).await?;
            client.initialize().await?;
            self.clients.insert(key.clone(), client);
        }

        Ok(self.clients.get_mut(&key).unwrap())
    }

    /// Ensure a file is opened in the language server (if server requires didOpen).
    ///
    /// file_path: &str — Absolute path to the source file.
    /// text: &str — Full file contents.
    pub async fn ensure_open(&mut self, file_path: &str, text: &str) -> Result<(), String> {
        let client = self.get_client(file_path).await?;
        if client.spec().needs_did_open {
            client.did_open(file_path, text).await?;
        }
        Ok(())
    }

    /// Shut down all running language servers.
    pub async fn shutdown_all(&mut self) {
        let keys: Vec<ClientKey> = self.clients.keys().cloned().collect();
        for key in keys {
            if let Some(mut client) = self.clients.remove(&key) {
                let _ = client.shutdown().await;
            }
        }
    }

    /// List all active language server sessions.
    pub fn active_sessions(&mut self) -> Vec<LspSessionInfo> {
        let mut out = Vec::with_capacity(self.clients.len());
        for ((project_root, language_id), client) in self.clients.iter_mut() {
            out.push(LspSessionInfo {
                server_name: client.spec().name.to_string(),
                language_id: language_id.clone(),
                project_root: project_root.clone(),
                pid: client.process_id(),
                initialized: client.is_initialized(),
                alive: client.is_alive(),
            });
        }
        out
    }
}
