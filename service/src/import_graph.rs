use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::extract::{collect_file_signatures_fast, extract_path_with_options, ExtractOptions};
use crate::lang::LanguageSpec;
use crate::types::{ImportEdge, ImportGraphResult, ImportNeighborsResult, SymbolKind};

const REFRESH_DEBOUNCE_MS: u128 = 5000;

#[derive(Debug, Clone)]
struct ProjectImportGraph {
    config_fingerprint: u64,
    respect_gitignore: bool,
    file_signatures: HashMap<String, u64>,
    outgoing: HashMap<String, Vec<ImportEdge>>,
    incoming: HashMap<String, Vec<ImportEdge>>,
    total_files: usize,
    total_imports: usize,
    last_refresh_check_ms: u128,
}

#[derive(Default)]
struct ImportGraphManager {
    by_project_root: HashMap<String, Arc<RwLock<ProjectImportGraph>>>,
}

static GLOBAL_IMPORT_GRAPH_MANAGER: OnceLock<Mutex<ImportGraphManager>> = OnceLock::new();


pub fn import_graph(
    file_path: &Path,
    project_root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
) -> Result<ImportGraphResult, String> {
    let start = std::time::Instant::now();
    let canonical_file = file_path
        .canonicalize()
        .map_err(|e| format!("failed to resolve file path: {e}"))?;
    let canonical_root = project_root
        .canonicalize()
        .map_err(|e| format!("failed to resolve project root: {e}"))?;
    let fingerprint = config_fingerprint(specs, respect_gitignore);

    let graph_lock = {
        let mut manager = manager()
            .lock()
            .map_err(|_| "import graph lock poisoned".to_string())?;
        manager.get_or_build(&canonical_root, specs, respect_gitignore, fingerprint)?
    };
    refresh_graph_if_needed(
        &graph_lock,
        &canonical_root,
        specs,
        respect_gitignore,
        fingerprint,
    )?;

    let graph = graph_lock
        .read()
        .map_err(|_| "import graph read lock poisoned".to_string())?;

    let file_key = normalize(&canonical_file);
    let imports = graph
        .outgoing
        .get(&file_key)
        .cloned()
        .unwrap_or_default();

    Ok(ImportGraphResult {
        file_path: canonical_file.to_string_lossy().to_string(),
        project_root: canonical_root.to_string_lossy().to_string(),
        edges: imports,
        total_files: graph.total_files,
        total_imports: graph.total_imports,
        duration_ms: start.elapsed().as_millis(),
    })
}


pub fn find_importers(
    file_path: &Path,
    project_root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
) -> Result<ImportGraphResult, String> {
    let start = std::time::Instant::now();
    let canonical_file = file_path
        .canonicalize()
        .map_err(|e| format!("failed to resolve file path: {e}"))?;
    let canonical_root = project_root
        .canonicalize()
        .map_err(|e| format!("failed to resolve project root: {e}"))?;
    let fingerprint = config_fingerprint(specs, respect_gitignore);

    let graph_lock = {
        let mut manager = manager()
            .lock()
            .map_err(|_| "import graph lock poisoned".to_string())?;
        manager.get_or_build(&canonical_root, specs, respect_gitignore, fingerprint)?
    };
    refresh_graph_if_needed(
        &graph_lock,
        &canonical_root,
        specs,
        respect_gitignore,
        fingerprint,
    )?;

    let graph = graph_lock
        .read()
        .map_err(|_| "import graph read lock poisoned".to_string())?;

    let file_key = normalize(&canonical_file);
    let importers = graph
        .incoming
        .get(&file_key)
        .cloned()
        .unwrap_or_default();

    Ok(ImportGraphResult {
        file_path: canonical_file.to_string_lossy().to_string(),
        project_root: canonical_root.to_string_lossy().to_string(),
        edges: importers,
        total_files: graph.total_files,
        total_imports: graph.total_imports,
        duration_ms: start.elapsed().as_millis(),
    })
}


fn manager() -> &'static Mutex<ImportGraphManager> {
    GLOBAL_IMPORT_GRAPH_MANAGER.get_or_init(|| Mutex::new(ImportGraphManager::default()))
}

impl ImportGraphManager {
    fn get_or_build(
        &mut self,
        project_root: &Path,
        specs: &[LanguageSpec],
        respect_gitignore: bool,
        config_fingerprint: u64,
    ) -> Result<Arc<RwLock<ProjectImportGraph>>, String> {
        let key = project_root.to_string_lossy().to_string();
        if let Some(existing) = self.by_project_root.get(&key) {
            return Ok(Arc::clone(existing));
        }

        let graph = build_graph(project_root, specs, respect_gitignore, config_fingerprint)?;
        let project_lock = Arc::new(RwLock::new(graph));
        self.by_project_root.insert(key, Arc::clone(&project_lock));
        Ok(project_lock)
    }
}

fn refresh_graph_if_needed(
    graph_lock: &Arc<RwLock<ProjectImportGraph>>,
    project_root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    config_fingerprint: u64,
) -> Result<(), String> {
    let mut graph = graph_lock
        .write()
        .map_err(|_| "import graph write lock poisoned".to_string())?;

    if now_ms().saturating_sub(graph.last_refresh_check_ms) < REFRESH_DEBOUNCE_MS {
        return Ok(());
    }
    graph.last_refresh_check_ms = now_ms();

    if graph.respect_gitignore != respect_gitignore || graph.config_fingerprint != config_fingerprint {
        *graph = build_graph(project_root, specs, respect_gitignore, config_fingerprint)?;
        return Ok(());
    }

    let current_signatures = collect_file_signatures_fast(
        project_root,
        specs,
        ExtractOptions { respect_gitignore },
    );
    if current_signatures != graph.file_signatures {
        *graph = build_graph(project_root, specs, respect_gitignore, config_fingerprint)?;
    }

    Ok(())
}


fn build_graph(
    root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    config_fingerprint: u64,
) -> Result<ProjectImportGraph, String> {
    let file_signatures = collect_file_signatures_fast(
        root,
        specs,
        ExtractOptions { respect_gitignore },
    );
    let extracted = extract_path_with_options(
        root,
        specs,
        ExtractOptions { respect_gitignore },
    )?;

    let all_files: Vec<String> = extracted
        .iter()
        .map(|r| r.file_path.clone())
        .collect();

    let file_index = build_file_index(&all_files);

    let mut outgoing: HashMap<String, Vec<ImportEdge>> = HashMap::new();
    let mut incoming: HashMap<String, Vec<ImportEdge>> = HashMap::new();
    let mut total_imports: usize = 0;

    for result in &extracted {
        let source_file = &result.file_path;
        let source_key = normalize(Path::new(source_file));
        let source_dir = Path::new(source_file).parent();

        for symbol in &result.symbols {
            if symbol.kind != SymbolKind::Import {
                continue;
            }

            let module_path = parse_import_source(&symbol.source, &result.language);
            if module_path.is_empty() {
                continue;
            }

            let resolved = resolve_import(
                &module_path,
                &result.language,
                source_dir,
                root,
                &file_index,
            );

            if let Some(target_file) = resolved {
                let target_key = normalize(Path::new(&target_file));
                if target_key == source_key {
                    continue;
                }

                let edge = ImportEdge {
                    source_file: source_file.clone(),
                    target_file: target_file.clone(),
                    import_statement: symbol.source.trim().to_string(),
                    module_path: module_path.clone(),
                    language: result.language.clone(),
                    line: symbol.line_start,
                };

                total_imports += 1;
                outgoing.entry(source_key.clone()).or_default().push(edge.clone());
                incoming.entry(target_key).or_default().push(edge);
            }
        }
    }

    Ok(ProjectImportGraph {
        config_fingerprint,
        respect_gitignore,
        file_signatures,
        outgoing,
        incoming,
        total_files: all_files.len(),
        total_imports,
        last_refresh_check_ms: now_ms(),
    })
}


pub fn import_neighbors(
    file_path: &Path,
    project_root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
) -> Result<ImportNeighborsResult, String> {
    let start = std::time::Instant::now();
    let canonical_file = file_path
        .canonicalize()
        .map_err(|e| format!("failed to resolve file path: {e}"))?;
    let canonical_root = project_root
        .canonicalize()
        .map_err(|e| format!("failed to resolve project root: {e}"))?;
    let fingerprint = config_fingerprint(specs, respect_gitignore);

    let graph_lock = {
        let mut manager = manager()
            .lock()
            .map_err(|_| "import graph lock poisoned".to_string())?;
        manager.get_or_build(&canonical_root, specs, respect_gitignore, fingerprint)?
    };
    refresh_graph_if_needed(
        &graph_lock,
        &canonical_root,
        specs,
        respect_gitignore,
        fingerprint,
    )?;

    let graph = graph_lock
        .read()
        .map_err(|_| "import graph read lock poisoned".to_string())?;

    let file_key = normalize(&canonical_file);
    let imports = graph
        .outgoing
        .get(&file_key)
        .cloned()
        .unwrap_or_default();
    let importers = graph
        .incoming
        .get(&file_key)
        .cloned()
        .unwrap_or_default();

    Ok(ImportNeighborsResult {
        file_path: canonical_file.to_string_lossy().to_string(),
        project_root: canonical_root.to_string_lossy().to_string(),
        imports,
        importers,
        total_files: graph.total_files,
        total_imports: graph.total_imports,
        duration_ms: start.elapsed().as_millis(),
    })
}

fn config_fingerprint(specs: &[LanguageSpec], respect_gitignore: bool) -> u64 {
    let mut hasher = DefaultHasher::new();
    respect_gitignore.hash(&mut hasher);

    let mut lang_names: Vec<&str> = specs.iter().map(|spec| spec.name).collect();
    lang_names.sort_unstable();
    for lang in lang_names {
        lang.hash(&mut hasher);
    }

    hasher.finish()
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}


fn build_file_index(files: &[String]) -> HashMap<String, String> {
    let mut index: HashMap<String, String> = HashMap::new();

    for file in files {
        let path = Path::new(file);

        let key = normalize(path);
        index.insert(key, file.clone());

        if let Some(stem) = path.file_stem() {
            let stem_str = stem.to_string_lossy().to_string();
            let parent = path.parent().unwrap_or(Path::new(""));
            let parent_key = normalize(parent);
            let stem_key = format!("{}/{}", parent_key, stem_str.to_ascii_lowercase());
            index.entry(stem_key).or_insert_with(|| file.clone());
        }
    }

    index
}


fn parse_import_source(source: &str, language: &str) -> String {
    let trimmed = source.trim();

    match language {
        "python" => parse_python_import(trimmed),
        "javascript" | "typescript" | "tsx" => parse_js_import(trimmed),
        "rust" => parse_rust_use(trimmed),
        "go" => parse_go_import(trimmed),
        "java" => parse_java_import(trimmed),
        "csharp" => parse_csharp_using(trimmed),
        "c" | "cpp" => parse_c_include(trimmed),
        _ => String::new(),
    }
}


fn parse_python_import(source: &str) -> String {
    if let Some(rest) = source.strip_prefix("from") {
        let rest = rest.trim_start();
        if let Some(idx) = rest.find("import") {
            let module = rest[..idx].trim();
            return module.to_string();
        }
    }

    if let Some(rest) = source.strip_prefix("import") {
        let rest = rest.trim_start();
        let module = rest.split([',', ' ', '\n']).next().unwrap_or("");
        return module.trim().to_string();
    }

    String::new()
}


fn parse_js_import(source: &str) -> String {
    if let Some(idx) = source.find("from") {
        let after = source[idx + 4..].trim();
        return extract_string_literal(after);
    }

    if let Some(idx) = source.find("require(") {
        let after = source[idx + 8..].trim();
        return extract_string_literal(after);
    }

    if source.starts_with("import") {
        let after = source[6..].trim();
        if after.starts_with('\'') || after.starts_with('"') {
            return extract_string_literal(after);
        }
    }

    String::new()
}


fn parse_rust_use(source: &str) -> String {
    if let Some(rest) = source.strip_prefix("use") {
        let rest = rest.trim_start();
        let path = rest.trim_end_matches(';').trim();
        let base = path.split("::").next().unwrap_or("");
        return base.to_string();
    }
    String::new()
}


fn parse_go_import(source: &str) -> String {
    let trimmed = source.trim();

    if trimmed.starts_with("import") {
        let after = trimmed[6..].trim();

        if after.starts_with('(') {
            let inner = after.trim_start_matches('(').trim_end_matches(')').trim();
            let first_line = inner.lines().next().unwrap_or("").trim();
            return extract_string_literal(first_line);
        }

        return extract_string_literal(after);
    }

    String::new()
}


fn parse_java_import(source: &str) -> String {
    if let Some(rest) = source.strip_prefix("import") {
        let rest = rest.trim_start();
        let rest = if let Some(r) = rest.strip_prefix("static") {
            r.trim_start()
        } else {
            rest
        };
        let path = rest.trim_end_matches(';').trim();
        return path.to_string();
    }
    String::new()
}


fn parse_csharp_using(source: &str) -> String {
    if let Some(rest) = source.strip_prefix("using") {
        let rest = rest.trim_start();
        let rest = if let Some(r) = rest.strip_prefix("static") {
            r.trim_start()
        } else {
            rest
        };
        let ns = rest.trim_end_matches(';').trim();
        if ns.contains('=') {
            let parts: Vec<&str> = ns.splitn(2, '=').collect();
            if parts.len() == 2 {
                return parts[1].trim().to_string();
            }
        }
        return ns.to_string();
    }
    String::new()
}


fn parse_c_include(source: &str) -> String {
    let trimmed = source.trim();

    if let Some(rest) = trimmed.strip_prefix("#include") {
        let rest = rest.trim_start();
        if rest.starts_with('"') {
            let end = rest[1..].find('"').unwrap_or(rest.len() - 1);
            return rest[1..1 + end].to_string();
        }
        if rest.starts_with('<') {
            let end = rest[1..].find('>').unwrap_or(rest.len() - 1);
            return rest[1..1 + end].to_string();
        }
    }

    String::new()
}


fn extract_string_literal(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.starts_with('"') {
        let end = trimmed[1..].find('"').unwrap_or(trimmed.len() - 1);
        return trimmed[1..1 + end].to_string();
    }
    if trimmed.starts_with('\'') {
        let end = trimmed[1..].find('\'').unwrap_or(trimmed.len() - 1);
        return trimmed[1..1 + end].to_string();
    }
    if trimmed.starts_with('`') {
        let end = trimmed[1..].find('`').unwrap_or(trimmed.len() - 1);
        return trimmed[1..1 + end].to_string();
    }
    String::new()
}


fn resolve_import(
    module_path: &str,
    language: &str,
    source_dir: Option<&Path>,
    project_root: &Path,
    file_index: &HashMap<String, String>,
) -> Option<String> {
    match language {
        "python" => resolve_python_import(module_path, source_dir, project_root, file_index),
        "javascript" | "typescript" | "tsx" => {
            resolve_js_import(module_path, source_dir, file_index)
        }
        "rust" => resolve_rust_import(module_path, project_root, file_index),
        "c" | "cpp" => resolve_c_include(module_path, source_dir, project_root, file_index),
        "go" | "java" | "csharp" => None,
        _ => None,
    }
}


fn resolve_python_import(
    module_path: &str,
    source_dir: Option<&Path>,
    project_root: &Path,
    file_index: &HashMap<String, String>,
) -> Option<String> {
    let is_relative = module_path.starts_with('.');

    if is_relative {
        let dots = module_path.chars().take_while(|c| *c == '.').count();
        let rest = &module_path[dots..];
        let base = source_dir?;
        let mut target = base.to_path_buf();
        for _ in 1..dots {
            target = target.parent()?.to_path_buf();
        }
        if !rest.is_empty() {
            for part in rest.split('.') {
                target = target.join(part);
            }
        }
        return try_python_file(&target, file_index);
    }

    let parts: Vec<&str> = module_path.split('.').collect();
    let mut target = project_root.to_path_buf();
    for part in &parts {
        target = target.join(part);
    }
    if let Some(found) = try_python_file(&target, file_index) {
        return Some(found);
    }

    if parts.len() > 1 {
        let mut target = project_root.to_path_buf();
        for part in &parts[..parts.len() - 1] {
            target = target.join(part);
        }
        if let Some(found) = try_python_file(&target, file_index) {
            return Some(found);
        }
    }

    None
}


fn try_python_file(base: &Path, file_index: &HashMap<String, String>) -> Option<String> {
    let py = base.with_extension("py");
    let key = normalize(&py);
    if let Some(f) = file_index.get(&key) {
        return Some(f.clone());
    }

    let init = base.join("__init__.py");
    let key = normalize(&init);
    if let Some(f) = file_index.get(&key) {
        return Some(f.clone());
    }

    let pyi = base.with_extension("pyi");
    let key = normalize(&pyi);
    if let Some(f) = file_index.get(&key) {
        return Some(f.clone());
    }

    None
}


fn resolve_js_import(
    module_path: &str,
    source_dir: Option<&Path>,
    file_index: &HashMap<String, String>,
) -> Option<String> {
    if !module_path.starts_with('.') && !module_path.starts_with('/') {
        return None;
    }

    let base = source_dir?;
    let target = base.join(module_path);

    let extensions = ["", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"];
    for ext in &extensions {
        let candidate = if ext.is_empty() {
            target.clone()
        } else {
            target.with_extension(ext.trim_start_matches('.'))
        };
        let key = normalize(&candidate);
        if let Some(f) = file_index.get(&key) {
            return Some(f.clone());
        }
    }

    let index_names = ["index.ts", "index.tsx", "index.js", "index.jsx"];
    for name in &index_names {
        let candidate = target.join(name);
        let key = normalize(&candidate);
        if let Some(f) = file_index.get(&key) {
            return Some(f.clone());
        }
    }

    None
}


fn resolve_rust_import(
    crate_name: &str,
    project_root: &Path,
    file_index: &HashMap<String, String>,
) -> Option<String> {
    if crate_name != "crate" && crate_name != "self" && crate_name != "super" {
        return None;
    }

    let lib_rs = project_root.join("src").join("lib.rs");
    let key = normalize(&lib_rs);
    if let Some(f) = file_index.get(&key) {
        return Some(f.clone());
    }

    let main_rs = project_root.join("src").join("main.rs");
    let key = normalize(&main_rs);
    if let Some(f) = file_index.get(&key) {
        return Some(f.clone());
    }

    None
}


fn resolve_c_include(
    include_path: &str,
    source_dir: Option<&Path>,
    project_root: &Path,
    file_index: &HashMap<String, String>,
) -> Option<String> {
    if let Some(dir) = source_dir {
        let candidate = dir.join(include_path);
        let key = normalize(&candidate);
        if let Some(f) = file_index.get(&key) {
            return Some(f.clone());
        }
    }

    let candidate = project_root.join(include_path);
    let key = normalize(&candidate);
    if let Some(f) = file_index.get(&key) {
        return Some(f.clone());
    }

    None
}


fn normalize(path: &Path) -> String {
    let s = path.to_string_lossy().to_string();
    s.replace('\\', "/").to_ascii_lowercase()
}
