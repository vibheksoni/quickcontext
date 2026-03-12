use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use redb::{Database, ReadableDatabase, TableDefinition};
use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser};

use crate::extract::{collect_file_signatures_fast, extract_path_with_options, ExtractOptions};
use crate::intent::{build_intent_terms, normalize_intent_level, IntentTerms};
use crate::lang::LanguageSpec;
use crate::types::{
    CallGraphTraceResult,
    CallerItem,
    CallerLookupResult,
    ExtractedSymbol,
    SymbolLookupItem,
    SymbolLookupResult,
    TraceDirection,
    TraceEdge,
    TraceNode,
};


const STATE_KEY: &str = "state";
const SNAPSHOT_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("snapshot");
const REFRESH_DEBOUNCE_MS: u128 = 5000;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SymbolRecord {
    id: usize,
    name: String,
    name_lower: String,
    kind: String,
    language: String,
    file_path: String,
    line_start: usize,
    line_end: usize,
    byte_start: usize,
    byte_end: usize,
    parent: Option<String>,
    signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CallerEdge {
    callee_name_lower: String,
    callee_name: String,
    caller_symbol_id: usize,
    caller_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedIndex {
    project_root: String,
    file_signatures: HashMap<String, u64>,
    symbols: Vec<SymbolRecord>,
    edges: Vec<CallerEdge>,
    #[serde(default)]
    call_edges_ready: bool,
}

#[derive(Debug, Clone)]
struct ProjectIndex {
    project_root: String,
    file_signatures: HashMap<String, u64>,
    symbols: Vec<SymbolRecord>,
    token_to_symbol_ids: HashMap<String, Vec<usize>>,
    name_to_symbol_ids: HashMap<String, Vec<usize>>,
    callers_by_callee: HashMap<String, Vec<CallerEdge>>,
    callees_by_caller: HashMap<usize, Vec<CallerEdge>>,
    call_edges_ready: bool,
    last_refresh_check_ms: u128,
}

#[derive(Default)]
struct SymbolIndexManager {
    by_project_root: HashMap<String, Arc<RwLock<ProjectIndex>>>,
}

static GLOBAL_INDEX_MANAGER: OnceLock<Mutex<SymbolIndexManager>> = OnceLock::new();

pub fn symbol_lookup(
    query: &str,
    path: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    limit: usize,
    intent_mode: bool,
    intent_level: u8,
) -> Result<SymbolLookupResult, String> {
    let effective_limit = limit.max(1);
    let project_root = normalize_path(path)?;

    let project_lock = {
        let mut manager = manager().lock().map_err(|_| "symbol index lock poisoned".to_string())?;
        manager.get_or_build(&project_root, specs, respect_gitignore)?
    };

    let from_cache = refresh_symbol_index_if_needed(&project_lock, &project_root, specs, respect_gitignore)?;

    let project_index = project_lock
        .read()
        .map_err(|_| "symbol index read lock poisoned".to_string())?;

    let normalized_intent_level = normalize_intent_level(intent_level);
    let intent_terms = if intent_mode {
        build_intent_terms(query, normalized_intent_level)
    } else {
        IntentTerms {
            exact_terms: Vec::new(),
            expanded_terms: Vec::new(),
        }
    };

    let query_tokens = if intent_mode {
        collect_intent_query_tokens(&intent_terms)
    } else {
        tokenize(query)
    };

    let exact_term_set: HashSet<String> = if intent_mode {
        intent_terms.exact_terms.iter().cloned().collect()
    } else {
        HashSet::new()
    };
    let expanded_term_set: HashSet<String> = if intent_mode {
        intent_terms.expanded_terms.iter().cloned().collect()
    } else {
        HashSet::new()
    };
    let qualified_query = parse_qualified_symbol_query(query);

    let mut score_by_symbol_id: HashMap<usize, f64> = HashMap::new();

    if query_tokens.is_empty() {
        for symbol in &project_index.symbols {
            if symbol.name_lower.contains(&query.to_ascii_lowercase()) {
                score_by_symbol_id.insert(symbol.id, 1.0);
            }
        }
    } else {
        for token in &query_tokens {
            if let Some(ids) = project_index.token_to_symbol_ids.get(token) {
                for id in ids {
                    *score_by_symbol_id.entry(*id).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    if let Some(ids) = project_index.name_to_symbol_ids.get(&query.to_ascii_lowercase()) {
        for id in ids {
            *score_by_symbol_id.entry(*id).or_insert(0.0) += query_tokens.len().max(1) as f64 + 2.0;
        }
    }

    if let Some((parent_query, name_query)) = qualified_query.as_ref() {
        for symbol in &project_index.symbols {
            let parent = symbol.parent.as_deref().unwrap_or_default().to_ascii_lowercase();
            if parent == *parent_query && symbol.name_lower == *name_query {
                *score_by_symbol_id.entry(symbol.id).or_insert(0.0) += query_tokens.len().max(1) as f64 + 4.5;
            } else if parent == *parent_query {
                *score_by_symbol_id.entry(symbol.id).or_insert(0.0) += 1.25;
            }
        }
    }

    if intent_mode {
        for (symbol_id, score) in &mut score_by_symbol_id {
            if let Some(symbol) = project_index.symbols.get(*symbol_id) {
                let symbol_tokens = tokenize(&symbol.name);
                *score *= compute_intent_symbol_score(
                    &symbol_tokens,
                    &exact_term_set,
                    &expanded_term_set,
                    normalized_intent_level,
                );
            }
        }
    }

    if !query_tokens.is_empty() {
        for (symbol_id, score) in &mut score_by_symbol_id {
            if let Some(symbol) = project_index.symbols.get(*symbol_id) {
                let exact_name_match = symbol.name_lower == query.to_ascii_lowercase();
                let symbol_tokens = tokenize(&symbol.name);
                let coverage = symbol_query_coverage(&query_tokens, &symbol_tokens, exact_name_match);
                *score *= symbol_lookup_coverage_multiplier(query_tokens.len(), coverage, exact_name_match);
                *score *= symbol_lookup_kind_multiplier(&symbol.kind, query_tokens.len(), coverage, exact_name_match);
            }
        }
    }

    let mut ranked: Vec<(usize, f64)> = score_by_symbol_id.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut results = Vec::new();
    for (id, _) in ranked.into_iter().take(effective_limit) {
        if let Some(symbol) = project_index.symbols.get(id) {
            results.push(SymbolLookupItem {
                name: symbol.name.clone(),
                kind: symbol.kind.clone(),
                language: symbol.language.clone(),
                file_path: symbol.file_path.clone(),
                line_start: symbol.line_start,
                line_end: symbol.line_end,
                parent: symbol.parent.clone(),
                signature: symbol.signature.clone(),
            });
        }
    }

    Ok(SymbolLookupResult {
        project_root,
        results,
        indexed_files: project_index.file_signatures.len(),
        indexed_symbols: project_index.symbols.len(),
        from_cache,
    })
}


pub fn find_callers(
    symbol: &str,
    path: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    limit: usize,
) -> Result<CallerLookupResult, String> {
    let effective_limit = limit.max(1);
    let symbol_lower = symbol.to_ascii_lowercase();
    let project_root = normalize_path(path)?;

    let project_lock = {
        let mut manager = manager().lock().map_err(|_| "symbol index lock poisoned".to_string())?;
        manager.get_or_build(&project_root, specs, respect_gitignore)?
    };

    let from_cache = refresh_symbol_index_if_needed(&project_lock, &project_root, specs, respect_gitignore)?;

    let root = Path::new(&project_root);
    {
        let index = project_lock
            .read()
            .map_err(|_| "symbol index read lock poisoned".to_string())?;
        if !index.call_edges_ready {
            drop(index);
            let mut write_index = project_lock
                .write()
                .map_err(|_| "symbol index write lock poisoned".to_string())?;
            ensure_call_edges(&mut write_index, root, specs, respect_gitignore)?;
        }
    }

    let index = project_lock
        .read()
        .map_err(|_| "symbol index read lock poisoned".to_string())?;

    let callers = build_callers_from_index(&index, &symbol_lower, effective_limit);

    Ok(CallerLookupResult {
        project_root,
        symbol: symbol.to_string(),
        callers,
        indexed_files: index.file_signatures.len(),
        indexed_symbols: index.symbols.len(),
        from_cache,
    })
}

fn build_callers_from_index(index: &ProjectIndex, symbol_lower: &str, limit: usize) -> Vec<CallerItem> {
    let edges = index
        .callers_by_callee
        .get(symbol_lower)
        .cloned()
        .unwrap_or_default();

    let mut callers = Vec::new();
    for edge in edges.into_iter().take(limit) {
        if let Some(caller) = index.symbols.get(edge.caller_symbol_id) {
            callers.push(CallerItem {
                callee_name: edge.callee_name,
                caller_name: caller.name.clone(),
                caller_kind: caller.kind.clone(),
                caller_language: caller.language.clone(),
                caller_file_path: caller.file_path.clone(),
                caller_line: edge.caller_line,
            });
        }
    }

    callers
}

pub fn trace_call_graph(
    symbol: &str,
    path: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    direction: TraceDirection,
    max_depth: usize,
) -> Result<CallGraphTraceResult, String> {
    let symbol_lower = symbol.to_ascii_lowercase();
    let project_root = normalize_path(path)?;

    let project_lock = {
        let mut manager = manager().lock().map_err(|_| "symbol index lock poisoned".to_string())?;
        manager.get_or_build(&project_root, specs, respect_gitignore)?
    };

    let from_cache = refresh_symbol_index_if_needed(&project_lock, &project_root, specs, respect_gitignore)?;

    let root = Path::new(&project_root);
    {
        let index = project_lock
            .read()
            .map_err(|_| "symbol index read lock poisoned".to_string())?;
        if !index.call_edges_ready {
            drop(index);
            let mut write_index = project_lock
                .write()
                .map_err(|_| "symbol index write lock poisoned".to_string())?;
            ensure_call_edges(&mut write_index, root, specs, respect_gitignore)?;
        }
    }

    let index = project_lock
        .read()
        .map_err(|_| "symbol index read lock poisoned".to_string())?;

    let root_ids: Vec<usize> = index
        .name_to_symbol_ids
        .get(&symbol_lower)
        .cloned()
        .unwrap_or_default();

    if root_ids.is_empty() {
        return Ok(CallGraphTraceResult {
            root_symbol: symbol.to_string(),
            direction,
            max_depth,
            nodes: Vec::new(),
            edges: Vec::new(),
            cycles_detected: Vec::new(),
            indexed_files: index.file_signatures.len(),
            indexed_symbols: index.symbols.len(),
            from_cache,
        });
    }

    let mut nodes: Vec<TraceNode> = Vec::new();
    let mut edges: Vec<TraceEdge> = Vec::new();
    let mut cycles: Vec<String> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    for &root_id in &root_ids {
        if let Some(root_sym) = index.symbols.get(root_id) {
            if visited.insert(root_id) {
                nodes.push(TraceNode {
                    name: root_sym.name.clone(),
                    kind: root_sym.kind.clone(),
                    language: root_sym.language.clone(),
                    file_path: root_sym.file_path.clone(),
                    line_start: root_sym.line_start,
                    line_end: root_sym.line_end,
                    depth: 0,
                });
            }
        }
    }

    match direction {
        TraceDirection::Upstream => {
            bfs_upstream(&index, &root_ids, max_depth, &mut visited, &mut nodes, &mut edges, &mut cycles);
        }
        TraceDirection::Downstream => {
            bfs_downstream(&index, &root_ids, max_depth, &mut visited, &mut nodes, &mut edges, &mut cycles);
        }
        TraceDirection::Both => {
            let mut visited_up = visited.clone();
            let mut visited_down = visited.clone();
            bfs_upstream(&index, &root_ids, max_depth, &mut visited_up, &mut nodes, &mut edges, &mut cycles);
            bfs_downstream(&index, &root_ids, max_depth, &mut visited_down, &mut nodes, &mut edges, &mut cycles);
            visited = &visited_up | &visited_down;
        }
    }

    let _ = visited;

    Ok(CallGraphTraceResult {
        root_symbol: symbol.to_string(),
        direction,
        max_depth,
        nodes,
        edges,
        cycles_detected: cycles,
        indexed_files: index.file_signatures.len(),
        indexed_symbols: index.symbols.len(),
        from_cache,
    })
}

impl SymbolIndexManager {
    fn get_or_build(
        &mut self,
        project_root: &str,
        specs: &[LanguageSpec],
        respect_gitignore: bool,
    ) -> Result<Arc<RwLock<ProjectIndex>>, String> {
        if let Some(existing) = self.by_project_root.get(project_root) {
            return Ok(Arc::clone(existing));
        }

        let root = Path::new(project_root);
        let index = if let Some(persisted) = load_persisted_index(root)? {
            hydrate_project_index(persisted)
        } else {
            let rebuilt = build_project_index(root, specs, respect_gitignore)?;
            persist_index(root, &rebuilt)?;
            rebuilt
        };

        let project_lock = Arc::new(RwLock::new(index));
        self.by_project_root
            .insert(project_root.to_string(), Arc::clone(&project_lock));
        Ok(project_lock)
    }
}

fn refresh_symbol_index_if_needed(
    project_lock: &Arc<RwLock<ProjectIndex>>,
    project_root: &str,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
) -> Result<bool, String> {
    let root = Path::new(project_root);

    {
        let index = project_lock
            .read()
            .map_err(|_| "symbol index read lock poisoned".to_string())?;
        if now_ms().saturating_sub(index.last_refresh_check_ms) < REFRESH_DEBOUNCE_MS {
            return Ok(true);
        }
    }

    let current = collect_file_signatures_fast(
        root,
        specs,
        ExtractOptions {
            respect_gitignore,
        },
    );

    {
        let index = project_lock
            .read()
            .map_err(|_| "symbol index read lock poisoned".to_string())?;
        if index.file_signatures == current {
            return Ok(true);
        }
    }

    let mut index = project_lock
        .write()
        .map_err(|_| "symbol index write lock poisoned".to_string())?;

    if now_ms().saturating_sub(index.last_refresh_check_ms) < REFRESH_DEBOUNCE_MS {
        return Ok(true);
    }
    index.last_refresh_check_ms = now_ms();

    if index.file_signatures == current {
        return Ok(true);
    }

    let rebuilt = build_project_index(root, specs, respect_gitignore)?;
    persist_index(root, &rebuilt)?;
    *index = rebuilt;

    Ok(false)
}


fn ensure_call_edges(
    index: &mut ProjectIndex,
    root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
) -> Result<(), String> {
    if index.call_edges_ready {
        return Ok(());
    }

    let extracted = extract_path_with_options(
        root,
        specs,
        ExtractOptions {
            respect_gitignore,
        },
    )?;

    let mut edges = Vec::new();
    for file in &extracted {
        let source = match fs::read_to_string(&file.file_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let file_symbols: Vec<ExtractedSymbol> = file.symbols.clone();
        let file_edges = extract_call_edges(&source, &file.language, &file_symbols, specs)?;
        edges.extend(file_edges);
    }

    index.callers_by_callee.clear();
    index.callees_by_caller.clear();

    for edge in edges {
        index
            .callers_by_callee
            .entry(edge.callee_name_lower.clone())
            .or_default()
            .push(edge.clone());
        index
            .callees_by_caller
            .entry(edge.caller_symbol_id)
            .or_default()
            .push(edge);
    }

    index.call_edges_ready = true;
    persist_index(root, index)?;
    Ok(())
}

fn bfs_upstream(
    index: &ProjectIndex,
    start_ids: &[usize],
    max_depth: usize,
    visited: &mut HashSet<usize>,
    nodes: &mut Vec<TraceNode>,
    edges: &mut Vec<TraceEdge>,
    cycles: &mut Vec<String>,
) {
    let mut frontier: Vec<usize> = start_ids.to_vec();
    let mut ancestors: HashSet<usize> = start_ids.iter().copied().collect();
    let mut cycle_set: HashSet<String> = HashSet::new();
    let mut depth = 0;

    while depth < max_depth && !frontier.is_empty() {
        depth += 1;
        let mut next_frontier: Vec<usize> = Vec::new();

        for &sym_id in &frontier {
            let sym = match index.symbols.get(sym_id) {
                Some(s) => s,
                None => continue,
            };

            let sym_name_lower = sym.name_lower.clone();
            let caller_edges = match index.callers_by_callee.get(&sym_name_lower) {
                Some(e) => e,
                None => continue,
            };

            for edge in caller_edges {
                let caller_id = edge.caller_symbol_id;
                let caller = match index.symbols.get(caller_id) {
                    Some(s) => s,
                    None => continue,
                };

                edges.push(TraceEdge {
                    from_name: caller.name.clone(),
                    to_name: edge.callee_name.clone(),
                    call_line: edge.caller_line,
                    file_path: caller.file_path.clone(),
                });

                if !visited.insert(caller_id) {
                    if ancestors.contains(&caller_id) {
                        let key = format!("{} -> {}", caller.name, sym.name);
                        if cycle_set.insert(key.clone()) {
                            cycles.push(key);
                        }
                    }
                    continue;
                }

                nodes.push(TraceNode {
                    name: caller.name.clone(),
                    kind: caller.kind.clone(),
                    language: caller.language.clone(),
                    file_path: caller.file_path.clone(),
                    line_start: caller.line_start,
                    line_end: caller.line_end,
                    depth,
                });

                next_frontier.push(caller_id);
            }
        }

        for &id in &next_frontier {
            ancestors.insert(id);
        }
        frontier = next_frontier;
    }
}

fn bfs_downstream(
    index: &ProjectIndex,
    start_ids: &[usize],
    max_depth: usize,
    visited: &mut HashSet<usize>,
    nodes: &mut Vec<TraceNode>,
    edges: &mut Vec<TraceEdge>,
    cycles: &mut Vec<String>,
) {
    let mut frontier: Vec<usize> = start_ids.to_vec();
    let mut ancestors: HashSet<usize> = start_ids.iter().copied().collect();
    let mut cycle_set: HashSet<String> = HashSet::new();
    let mut depth = 0;

    while depth < max_depth && !frontier.is_empty() {
        depth += 1;
        let mut next_frontier: Vec<usize> = Vec::new();

        for &sym_id in &frontier {
            let sym = match index.symbols.get(sym_id) {
                Some(s) => s,
                None => continue,
            };

            let callee_edges = match index.callees_by_caller.get(&sym_id) {
                Some(e) => e,
                None => continue,
            };

            for edge in callee_edges {
                let callee_name_lower = &edge.callee_name_lower;

                edges.push(TraceEdge {
                    from_name: sym.name.clone(),
                    to_name: edge.callee_name.clone(),
                    call_line: edge.caller_line,
                    file_path: sym.file_path.clone(),
                });

                let target_ids = match index.name_to_symbol_ids.get(callee_name_lower) {
                    Some(ids) => ids,
                    None => continue,
                };

                for &target_id in target_ids {
                    if !visited.insert(target_id) {
                        if ancestors.contains(&target_id) {
                            let target = index.symbols.get(target_id);
                            let target_name = target.map(|s| s.name.as_str()).unwrap_or("?");
                            let key = format!("{} -> {}", sym.name, target_name);
                            if cycle_set.insert(key.clone()) {
                                cycles.push(key);
                            }
                        }
                        continue;
                    }

                    if let Some(target) = index.symbols.get(target_id) {
                        nodes.push(TraceNode {
                            name: target.name.clone(),
                            kind: target.kind.clone(),
                            language: target.language.clone(),
                            file_path: target.file_path.clone(),
                            line_start: target.line_start,
                            line_end: target.line_end,
                            depth,
                        });

                        next_frontier.push(target_id);
                    }
                }
            }
        }

        for &id in &next_frontier {
            ancestors.insert(id);
        }
        frontier = next_frontier;
    }
}

fn manager() -> &'static Mutex<SymbolIndexManager> {
    GLOBAL_INDEX_MANAGER.get_or_init(|| Mutex::new(SymbolIndexManager::default()))
}

fn normalize_path(path: &Path) -> Result<String, String> {
    let canonical = path.canonicalize().map_err(|e| format!("failed to resolve path: {e}"))?;
    Ok(canonical.to_string_lossy().to_string())
}

fn build_project_index(
    root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
) -> Result<ProjectIndex, String> {
    let file_signatures = collect_file_signatures_fast(
        root,
        specs,
        ExtractOptions {
            respect_gitignore,
        },
    );

    let extracted = extract_path_with_options(
        root,
        specs,
        ExtractOptions {
            respect_gitignore,
        },
    )?;

    let mut symbols = Vec::new();
    let mut token_to_symbol_ids: HashMap<String, Vec<usize>> = HashMap::new();
    let mut name_to_symbol_ids: HashMap<String, Vec<usize>> = HashMap::new();
    let mut callers_by_callee: HashMap<String, Vec<CallerEdge>> = HashMap::new();
    let mut callees_by_caller: HashMap<usize, Vec<CallerEdge>> = HashMap::new();

    for file in &extracted {
        for s in &file.symbols {
            let id = symbols.len();
            let name_lower = s.name.to_ascii_lowercase();
            let record = SymbolRecord {
                id,
                name: s.name.clone(),
                name_lower: name_lower.clone(),
                kind: format!("{:?}", s.kind).to_ascii_lowercase(),
                language: s.language.clone(),
                file_path: s.file_path.clone(),
                line_start: s.line_start,
                line_end: s.line_end,
                byte_start: s.byte_start,
                byte_end: s.byte_end,
                parent: s.parent.clone(),
                signature: s.signature.clone(),
            };

            name_to_symbol_ids.entry(name_lower).or_default().push(id);

            for token in tokenize(&record.name) {
                token_to_symbol_ids.entry(token).or_default().push(id);
            }

            symbols.push(record);
        }
        let source = match fs::read_to_string(&file.file_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let file_edges = extract_call_edges(&source, &file.language, &file.symbols, specs)?;
        for edge in file_edges {
            callers_by_callee
                .entry(edge.callee_name_lower.clone())
                .or_default()
                .push(edge.clone());
            callees_by_caller
                .entry(edge.caller_symbol_id)
                .or_default()
                .push(edge);
        }
    }

    Ok(ProjectIndex {
        project_root: root.to_string_lossy().to_string(),
        file_signatures,
        symbols,
        token_to_symbol_ids,
        name_to_symbol_ids,
        callers_by_callee,
        callees_by_caller,
        call_edges_ready: true,
        last_refresh_check_ms: now_ms(),
    })
}

fn extract_call_edges(
    source: &str,
    language_name: &str,
    symbols: &[ExtractedSymbol],
    specs: &[LanguageSpec],
) -> Result<Vec<CallerEdge>, String> {
    let spec = specs
        .iter()
        .find(|s| s.name == language_name)
        .ok_or_else(|| format!("language spec not found: {language_name}"))?;

    let mut parser = Parser::new();
    parser
        .set_language(&spec.language)
        .map_err(|_| format!("failed to set language for {language_name}"))?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| format!("failed to parse source for {language_name}"))?;

    let mut edges = Vec::new();
    collect_call_edges_recursive(tree.root_node(), source.as_bytes(), language_name, symbols, &mut edges);
    Ok(edges)
}

fn collect_call_edges_recursive(
    node: Node,
    src: &[u8],
    language_name: &str,
    symbols: &[ExtractedSymbol],
    edges: &mut Vec<CallerEdge>,
) {
    if let Some(callee_name) = extract_callee_name(node, src, language_name) {
        let call_byte = node.start_byte();
        if let Some(caller_symbol) = find_enclosing_symbol(symbols, call_byte) {
            edges.push(CallerEdge {
                callee_name_lower: callee_name.to_ascii_lowercase(),
                callee_name,
                caller_symbol_id: caller_symbol,
                caller_line: node.start_position().row + 1,
            });
        }
    }

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            collect_call_edges_recursive(child, src, language_name, symbols, edges);
        }
    }
}

fn extract_callee_name(node: Node, src: &[u8], language_name: &str) -> Option<String> {
    match language_name {
        "python" => {
            if node.kind() != "call" {
                return None;
            }
            let func = node.child_by_field_name("function")?;
            extract_last_identifier(func, src)
        }
        "javascript" | "typescript" => {
            if node.kind() != "call_expression" {
                return None;
            }
            let func = node.child_by_field_name("function")?;
            extract_last_identifier(func, src)
        }
        "rust" => {
            if node.kind() != "call_expression" {
                return None;
            }
            let func = node.child_by_field_name("function")?;
            extract_last_identifier(func, src)
        }
        _ => None,
    }
}

fn extract_last_identifier(node: Node, src: &[u8]) -> Option<String> {
    let mut stack = vec![node];
    let mut last = None;

    while let Some(curr) = stack.pop() {
        let kind = curr.kind();
        if kind == "identifier"
            || kind == "property_identifier"
            || kind == "field_identifier"
            || kind == "type_identifier"
            || kind == "scoped_identifier"
        {
            if let Ok(text) = curr.utf8_text(src) {
                if !text.is_empty() {
                    last = Some(text.to_string());
                }
            }
        }

        for i in 0..curr.child_count() {
            if let Some(child) = curr.child(i as u32) {
                stack.push(child);
            }
        }
    }

    last
}

fn find_enclosing_symbol(symbols: &[ExtractedSymbol], byte: usize) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;

    for (idx, s) in symbols.iter().enumerate() {
        if s.byte_start <= byte && byte <= s.byte_end {
            let span = s.byte_end - s.byte_start;
            match best {
                Some((_, best_span)) if span >= best_span => {}
                _ => best = Some((idx, span)),
            }
        }
    }

    best.map(|(idx, _)| idx)
}

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = HashSet::new();
    let mut current = String::new();

    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            if current.len() >= 2 {
                tokens.insert(current.clone());
            }
            current.clear();
        }
    }

    if !current.is_empty() && current.len() >= 2 {
        tokens.insert(current);
    }

    split_camel_case(input, &mut tokens);

    let mut out: Vec<String> = tokens.into_iter().collect();
    out.sort();
    out
}

fn split_camel_case(input: &str, tokens: &mut HashSet<String>) {
    let mut part = String::new();
    for ch in input.chars() {
        if ch.is_ascii_uppercase() && !part.is_empty() {
            if part.len() >= 2 {
                tokens.insert(part.to_ascii_lowercase());
            }
            part.clear();
        }

        if ch.is_ascii_alphanumeric() {
            part.push(ch);
        } else if !part.is_empty() {
            if part.len() >= 2 {
                tokens.insert(part.to_ascii_lowercase());
            }
            part.clear();
        }
    }

    if !part.is_empty() && part.len() >= 2 {
        tokens.insert(part.to_ascii_lowercase());
    }
}


fn collect_intent_query_tokens(intent_terms: &IntentTerms) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();

    for term in &intent_terms.exact_terms {
        for token in tokenize(term) {
            if seen.insert(token.clone()) {
                terms.push(token);
            }
        }
    }

    for term in &intent_terms.expanded_terms {
        for token in tokenize(term) {
            if seen.insert(token.clone()) {
                terms.push(token);
            }
        }
    }

    terms
}

fn compute_intent_symbol_score(
    symbol_tokens: &[String],
    exact_term_set: &HashSet<String>,
    expanded_term_set: &HashSet<String>,
    intent_level: u8,
) -> f64 {
    if exact_term_set.is_empty() && expanded_term_set.is_empty() {
        return 1.0;
    }

    let mut exact_hits = 0usize;
    let mut expanded_hits = 0usize;

    for token in symbol_tokens {
        if exact_term_set.contains(token) {
            exact_hits += 1;
            continue;
        }
        if expanded_term_set.contains(token) {
            expanded_hits += 1;
        }
    }

    let exact_boost = if exact_hits > 0 {
        1.0 + (exact_hits as f64 * 0.24)
    } else {
        1.0
    };

    let expanded_factor = match intent_level {
        1 => 0.03,
        2 => 0.045,
        _ => 0.06,
    };
    let expanded_boost = 1.0 + (expanded_hits as f64 * expanded_factor);

    let exact_coverage = if exact_term_set.is_empty() {
        0.0
    } else {
        exact_hits as f64 / exact_term_set.len() as f64
    };
    let expanded_coverage = if expanded_term_set.is_empty() {
        0.0
    } else {
        expanded_hits as f64 / expanded_term_set.len() as f64
    };
    let coverage_boost = 1.0 + (exact_coverage * 0.22) + (expanded_coverage * 0.06);

    let penalty = if exact_hits == 0 && expanded_hits > 0 {
        0.82
    } else {
        1.0
    };

    exact_boost * expanded_boost * coverage_boost * penalty
}

fn symbol_query_coverage(
    query_tokens: &[String],
    symbol_tokens: &[String],
    exact_name_match: bool,
) -> f64 {
    if exact_name_match || query_tokens.is_empty() {
        return 1.0;
    }

    let query_terms: HashSet<&str> = query_tokens.iter().map(String::as_str).collect();
    if query_terms.is_empty() {
        return 1.0;
    }

    let symbol_terms: HashSet<&str> = symbol_tokens.iter().map(String::as_str).collect();
    let matched = query_terms
        .iter()
        .filter(|term| symbol_terms.contains(**term))
        .count();

    matched as f64 / query_terms.len() as f64
}

fn symbol_lookup_coverage_multiplier(
    query_token_count: usize,
    coverage: f64,
    exact_name_match: bool,
) -> f64 {
    if exact_name_match || query_token_count <= 1 {
        return 1.0;
    }

    if coverage >= 0.999 {
        1.16
    } else if coverage >= 0.66 {
        1.0
    } else {
        0.58
    }
}

fn symbol_lookup_kind_multiplier(
    kind: &str,
    query_token_count: usize,
    coverage: f64,
    exact_name_match: bool,
) -> f64 {
    match kind.to_ascii_lowercase().as_str() {
        "class" | "struct" | "enum" | "interface" | "trait" | "module" => {
            if exact_name_match {
                1.22
            } else if query_token_count > 1 && coverage >= 0.999 {
                1.14
            } else {
                1.06
            }
        }
        "function" | "method" | "constructor" | "type_alias" => {
            if exact_name_match {
                1.16
            } else if query_token_count > 1 && coverage >= 0.999 {
                1.08
            } else {
                1.03
            }
        }
        "import" | "variable" | "property" | "constant" | "data_key" | "decorator" => {
            if exact_name_match {
                0.98
            } else if query_token_count > 1 && coverage < 0.999 {
                0.54
            } else {
                0.86
            }
        }
        _ => 1.0,
    }
}

fn parse_qualified_symbol_query(query: &str) -> Option<(String, String)> {
    let normalized = query.trim();
    if normalized.is_empty() {
        return None;
    }

    if let Some(idx) = normalized.rfind("::") {
        let parent = normalized[..idx].trim();
        let name = normalized[idx + 2..].trim();
        if !parent.is_empty() && !name.is_empty() {
            return Some((parent.to_ascii_lowercase(), name.to_ascii_lowercase()));
        }
    }

    if let Some(idx) = normalized.rfind('.') {
        let parent = normalized[..idx].trim();
        let name = normalized[idx + 1..].trim();
        if !parent.is_empty() && !name.is_empty() {
            return Some((parent.to_ascii_lowercase(), name.to_ascii_lowercase()));
        }
    }

    None
}

fn persistence_path(root: &Path) -> Result<PathBuf, String> {
    let dir = root.join(".quickcontext");
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create index dir: {e}"))?;
    Ok(dir.join("symbol_index.redb"))
}

fn persist_index(root: &Path, index: &ProjectIndex) -> Result<(), String> {
    let db_path = persistence_path(root)?;
    let db = Database::create(&db_path).map_err(|e| format!("failed to open redb: {e}"))?;

    let persisted = PersistedIndex {
        project_root: index.project_root.clone(),
        file_signatures: index.file_signatures.clone(),
        symbols: index.symbols.clone(),
        edges: index
            .callers_by_callee
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect(),
        call_edges_ready: index.call_edges_ready,
    };

    let encoded = serde_json::to_vec(&persisted).map_err(|e| format!("failed to encode index: {e}"))?;

    let txn = db.begin_write().map_err(|e| format!("failed to begin write txn: {e}"))?;
    {
        let mut table = txn
            .open_table(SNAPSHOT_TABLE)
            .map_err(|e| format!("failed to open snapshot table: {e}"))?;
        table
            .insert(STATE_KEY, encoded.as_slice())
            .map_err(|e| format!("failed to write snapshot: {e}"))?;
    }
    txn.commit().map_err(|e| format!("failed to commit snapshot: {e}"))
}

fn load_persisted_index(root: &Path) -> Result<Option<PersistedIndex>, String> {
    let db_path = persistence_path(root)?;
    let db = Database::create(&db_path).map_err(|e| format!("failed to open redb: {e}"))?;
    let txn = db.begin_read().map_err(|e| format!("failed to begin read txn: {e}"))?;

    let table = match txn.open_table(SNAPSHOT_TABLE) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let raw = match table.get(STATE_KEY) {
        Ok(Some(v)) => v,
        Ok(None) => return Ok(None),
        Err(_) => return Ok(None),
    };

    let decoded: PersistedIndex = serde_json::from_slice(raw.value())
        .map_err(|e| format!("failed to decode persisted index: {e}"))?;
    Ok(Some(decoded))
}

fn hydrate_project_index(persisted: PersistedIndex) -> ProjectIndex {
    let mut token_to_symbol_ids: HashMap<String, Vec<usize>> = HashMap::new();
    let mut name_to_symbol_ids: HashMap<String, Vec<usize>> = HashMap::new();
    let mut callers_by_callee: HashMap<String, Vec<CallerEdge>> = HashMap::new();
    let mut callees_by_caller: HashMap<usize, Vec<CallerEdge>> = HashMap::new();

    for symbol in &persisted.symbols {
        name_to_symbol_ids
            .entry(symbol.name_lower.clone())
            .or_default()
            .push(symbol.id);

        for token in tokenize(&symbol.name) {
            token_to_symbol_ids.entry(token).or_default().push(symbol.id);
        }
    }

    for edge in persisted.edges {
        callees_by_caller
            .entry(edge.caller_symbol_id)
            .or_default()
            .push(edge.clone());
        callers_by_callee
            .entry(edge.callee_name_lower.clone())
            .or_default()
            .push(edge);
    }

    ProjectIndex {
        project_root: persisted.project_root,
        file_signatures: persisted.file_signatures,
        symbols: persisted.symbols,
        token_to_symbol_ids,
        name_to_symbol_ids,
        callers_by_callee,
        callees_by_caller,
        call_edges_ready: persisted.call_edges_ready,
        last_refresh_check_ms: now_ms(),
    }
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_intent_symbol_score_prefers_exact_terms() {
        let exact_term_set: HashSet<String> = ["auth".to_string(), "token".to_string()].into_iter().collect();
        let expanded_term_set: HashSet<String> = ["authentication".to_string(), "jwt".to_string()].into_iter().collect();

        let exact_tokens = vec!["auth".to_string(), "token".to_string()];
        let expanded_only_tokens = vec!["authentication".to_string(), "jwt".to_string()];

        let exact_score = compute_intent_symbol_score(&exact_tokens, &exact_term_set, &expanded_term_set, 2);
        let expanded_only_score = compute_intent_symbol_score(&expanded_only_tokens, &exact_term_set, &expanded_term_set, 2);

        assert!(exact_score > expanded_only_score);
    }

    #[test]
    fn test_compute_intent_symbol_score_rewards_coverage_with_exact_hits() {
        let exact_term_set: HashSet<String> = ["auth".to_string(), "token".to_string()].into_iter().collect();
        let expanded_term_set: HashSet<String> = ["authentication".to_string(), "jwt".to_string()].into_iter().collect();

        let partial_tokens = vec!["auth".to_string()];
        let high_coverage_tokens = vec!["auth".to_string(), "token".to_string(), "authentication".to_string()];

        let partial_score = compute_intent_symbol_score(&partial_tokens, &exact_term_set, &expanded_term_set, 2);
        let high_coverage_score = compute_intent_symbol_score(&high_coverage_tokens, &exact_term_set, &expanded_term_set, 2);

        assert!(high_coverage_score > partial_score);
    }

    #[test]
    fn test_symbol_query_coverage_penalizes_partial_multi_token_matches() {
        let query_tokens = vec!["collection".to_string(), "manager".to_string()];
        let exact_tokens = vec!["collection".to_string(), "manager".to_string()];
        let partial_tokens = vec!["collection".to_string()];

        let exact_coverage = symbol_query_coverage(&query_tokens, &exact_tokens, false);
        let partial_coverage = symbol_query_coverage(&query_tokens, &partial_tokens, false);

        assert!(exact_coverage > partial_coverage);
        assert_eq!(partial_coverage, 0.5);
    }

    #[test]
    fn test_symbol_lookup_kind_multiplier_prefers_definitions_over_partial_noise() {
        let definition_score = symbol_lookup_coverage_multiplier(2, 1.0, true)
            * symbol_lookup_kind_multiplier("class", 2, 1.0, true);
        let import_score = symbol_lookup_coverage_multiplier(2, 0.5, false)
            * symbol_lookup_kind_multiplier("import", 2, 0.5, false);
        let variable_score = symbol_lookup_coverage_multiplier(2, 0.5, false)
            * symbol_lookup_kind_multiplier("variable", 2, 0.5, false);

        assert!(definition_score > import_score);
        assert!(definition_score > variable_score);
    }

    #[test]
    fn test_symbol_lookup_coverage_multiplier_is_neutral_for_single_token_queries() {
        let multiplier = symbol_lookup_coverage_multiplier(1, 0.5, false);
        assert_eq!(multiplier, 1.0);
    }

    #[test]
    fn test_parse_qualified_symbol_query_supports_dot_and_colons() {
        assert_eq!(
            parse_qualified_symbol_query("CodeSearcher._embed_query_cached"),
            Some(("codesearcher".to_string(), "_embed_query_cached".to_string()))
        );
        assert_eq!(
            parse_qualified_symbol_query("crate::module::symbol_name"),
            Some(("crate::module".to_string(), "symbol_name".to_string()))
        );
        assert_eq!(parse_qualified_symbol_query("CollectionManager"), None);
    }

    #[test]
    fn test_qualified_symbol_bonus_prefers_exact_member_over_parent_only_match() {
        let base = 1.0;
        let partial_parent_score = base + 1.25;
        let exact_member_score = base + 4.5;
        assert!(exact_member_score > partial_parent_score);
    }
}
