use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use redb::{Database, ReadableDatabase, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::extract::{collect_file_signatures_fast, ExtractOptions};
use crate::lang::{self, LanguageSpec};
use crate::query_parser::SearchFilter;
use crate::tokenizer::Tokenizer;

const STATE_KEY: &str = "state";
const SNAPSHOT_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("snapshot");
const TEXT_INDEX_SCHEMA_VERSION: u32 = 2;
const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.5;
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024;
const REFRESH_DEBOUNCE_MS: u128 = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TextDocRecord {
    id: u32,
    file_path: String,
    language: String,
    signature: u64,
    term_freq: Vec<(String, u32)>,
    doc_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedTextIndex {
    schema_version: u32,
    config_fingerprint: u64,
    project_root: String,
    respect_gitignore: bool,
    file_signatures: HashMap<String, u64>,
    docs: Vec<TextDocRecord>,
    next_doc_id: u32,
}

#[derive(Debug, Clone)]
struct ProjectTextIndex {
    project_root: String,
    config_fingerprint: u64,
    respect_gitignore: bool,
    file_signatures: HashMap<String, u64>,
    docs: HashMap<u32, TextDocRecord>,
    path_to_doc_id: HashMap<String, u32>,
    postings: HashMap<String, HashMap<u32, u32>>,
    total_doc_len: usize,
    next_doc_id: u32,
    last_refresh_check_ms: u128,
}

#[derive(Default)]
struct TextIndexManager {
    by_project_root: HashMap<String, Arc<RwLock<ProjectTextIndex>>>,
}

#[derive(Debug, Clone)]
pub struct IndexedCandidate {
    pub file_path: String,
    pub language: String,
    pub score: f64,
    pub matched_terms: Vec<String>,
}

static GLOBAL_TEXT_INDEX_MANAGER: OnceLock<Mutex<TextIndexManager>> = OnceLock::new();

pub fn search_candidates(
    root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    filter: &SearchFilter,
    query: &str,
    max_candidates: usize,
) -> Result<(Vec<IndexedCandidate>, usize), String> {
    let project_root = root
        .canonicalize()
        .map_err(|e| format!("failed to resolve path: {e}"))?;
    let fingerprint = config_fingerprint(specs, respect_gitignore);

    let project_lock = {
        let mut manager = manager()
            .lock()
            .map_err(|_| "text index lock poisoned".to_string())?;
        manager.get_or_build(&project_root, specs, respect_gitignore, fingerprint)?
    };

    refresh_index_if_needed(
        &project_lock,
        &project_root,
        specs,
        respect_gitignore,
        fingerprint,
    )?;

    let index = project_lock
        .read()
        .map_err(|_| "text index read lock poisoned".to_string())?;

    let searched_files = count_filtered_docs(&index, filter);
    let candidates = score_candidates(&index, query, filter, max_candidates.max(1));

    Ok((candidates, searched_files))
}


fn refresh_index_if_needed(
    project_lock: &Arc<RwLock<ProjectTextIndex>>,
    project_root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    config_fingerprint: u64,
) -> Result<(), String> {
    {
        let index = project_lock
            .read()
            .map_err(|_| "text index read lock poisoned".to_string())?;
        if now_ms().saturating_sub(index.last_refresh_check_ms) < REFRESH_DEBOUNCE_MS {
            return Ok(());
        }
        if index.respect_gitignore == respect_gitignore && index.config_fingerprint == config_fingerprint {
            let current = collect_file_signatures_fast(
                project_root,
                specs,
                ExtractOptions {
                    respect_gitignore,
                },
            );
            if current == index.file_signatures {
                return Ok(());
            }
        }
    }

    let mut index = project_lock
        .write()
        .map_err(|_| "text index write lock poisoned".to_string())?;

    if now_ms().saturating_sub(index.last_refresh_check_ms) < REFRESH_DEBOUNCE_MS {
        return Ok(());
    }
    index.last_refresh_check_ms = now_ms();

    if index.respect_gitignore != respect_gitignore || index.config_fingerprint != config_fingerprint {
        let rebuilt = build_index(project_root, specs, respect_gitignore, config_fingerprint)?;
        persist_index(project_root, &rebuilt)?;
        *index = rebuilt;
        return Ok(());
    }

    let changed = reconcile_index(&mut index, project_root, specs, respect_gitignore)?;
    if changed {
        persist_index(project_root, &index)?;
    }

    Ok(())
}

fn manager() -> &'static Mutex<TextIndexManager> {
    GLOBAL_TEXT_INDEX_MANAGER.get_or_init(|| Mutex::new(TextIndexManager::default()))
}

impl TextIndexManager {
    fn get_or_build(
        &mut self,
        project_root: &Path,
        specs: &[LanguageSpec],
        respect_gitignore: bool,
        config_fingerprint: u64,
    ) -> Result<Arc<RwLock<ProjectTextIndex>>, String> {
        let key = project_root.to_string_lossy().to_string();

        if let Some(existing) = self.by_project_root.get(&key) {
            return Ok(Arc::clone(existing));
        }

        let loaded = load_or_build_index(project_root, specs, respect_gitignore, config_fingerprint)?;
        let project_lock = Arc::new(RwLock::new(loaded));
        self.by_project_root
            .insert(key, Arc::clone(&project_lock));

        Ok(project_lock)
    }
}

fn load_or_build_index(
    project_root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    config_fingerprint: u64,
) -> Result<ProjectTextIndex, String> {
    if let Some(persisted) = load_persisted_index(project_root)? {
        if persisted.schema_version == TEXT_INDEX_SCHEMA_VERSION
            && persisted.respect_gitignore == respect_gitignore
            && persisted.config_fingerprint == config_fingerprint
        {
            return Ok(hydrate_index(persisted));
        }
    }

    let rebuilt = build_index(project_root, specs, respect_gitignore, config_fingerprint)?;
    persist_index(project_root, &rebuilt)?;
    Ok(rebuilt)
}

fn build_index(
    project_root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
    config_fingerprint: u64,
) -> Result<ProjectTextIndex, String> {
    let file_signatures = collect_file_signatures_fast(
        project_root,
        specs,
        ExtractOptions {
            respect_gitignore,
        },
    );

    let mut index = ProjectTextIndex {
        project_root: project_root.to_string_lossy().to_string(),
        config_fingerprint,
        respect_gitignore,
        file_signatures: HashMap::new(),
        docs: HashMap::new(),
        path_to_doc_id: HashMap::new(),
        postings: HashMap::new(),
        total_doc_len: 0,
        next_doc_id: 0,
        last_refresh_check_ms: now_ms(),
    };

    for (path, sig) in file_signatures {
        if !should_index_file(&path, specs) {
            index.file_signatures.insert(path, sig);
            continue;
        }
        if upsert_doc(&mut index, &path, sig, specs, None) {
            index.file_signatures.insert(path, sig);
        }
    }

    Ok(index)
}

fn reconcile_index(
    index: &mut ProjectTextIndex,
    project_root: &Path,
    specs: &[LanguageSpec],
    respect_gitignore: bool,
) -> Result<bool, String> {
    let current = collect_file_signatures_fast(
        project_root,
        specs,
        ExtractOptions {
            respect_gitignore,
        },
    );

    if current == index.file_signatures {
        return Ok(false);
    }

    let previous = index.file_signatures.clone();

    for path in previous.keys() {
        if !current.contains_key(path) {
            remove_doc_by_path(index, path);
        }
    }

    let mut changed = false;
    let mut next_signatures = previous;

    for (path, sig) in &current {
        if !should_index_file(path, specs) {
            let had_doc = index.path_to_doc_id.contains_key(path);
            let old_sig = next_signatures.get(path).copied();
            mark_non_indexable_path(index, &mut next_signatures, path, *sig);
            if had_doc || old_sig != Some(*sig) {
                changed = true;
            }
            continue;
        }

        let old_sig = next_signatures.get(path).copied();
        if old_sig == Some(*sig) {
            continue;
        }

        let reuse_id = index.path_to_doc_id.get(path).copied();
        if upsert_doc(index, path, *sig, specs, reuse_id) {
            next_signatures.insert(path.clone(), *sig);
            changed = true;
        }
    }

    next_signatures.retain(|path, _| current.contains_key(path));
    if next_signatures != index.file_signatures {
        changed = true;
    }

    index.file_signatures = next_signatures;
    index.last_refresh_check_ms = now_ms();
    Ok(changed)
}

fn should_index_file(path: &str, specs: &[LanguageSpec]) -> bool {
    if lang::detect_language(path, specs).is_none() {
        return false;
    }

    let Ok(meta) = fs::metadata(path) else {
        return false;
    };

    meta.len() <= MAX_FILE_SIZE
}

fn mark_non_indexable_path(
    index: &mut ProjectTextIndex,
    next_signatures: &mut HashMap<String, u64>,
    path: &str,
    signature: u64,
) {
    remove_doc_by_path(index, path);
    next_signatures.insert(path.to_string(), signature);
}

fn upsert_doc(
    index: &mut ProjectTextIndex,
    file_path: &str,
    signature: u64,
    specs: &[LanguageSpec],
    reuse_id: Option<u32>,
) -> bool {
    let source = match fs::read_to_string(file_path) {
        Ok(s) => s,
        Err(_) => return false,
    };

    if source.contains('\0') {
        return false;
    }

    let spec = match lang::detect_language(file_path, specs) {
        Some(s) => s,
        None => return false,
    };

    let term_freq = tokenizer().tokenize_with_frequency(&source);
    let doc_len: usize = term_freq.iter().map(|(_, tf)| *tf as usize).sum();

    let id = reuse_id.unwrap_or_else(|| {
        let next = index.next_doc_id;
        index.next_doc_id = index.next_doc_id.saturating_add(1);
        next
    });

    remove_doc_by_id(index, id);

    let record = TextDocRecord {
        id,
        file_path: file_path.to_string(),
        language: spec.name.to_string(),
        signature,
        term_freq,
        doc_len,
    };

    add_doc(index, record);
    true
}


fn add_doc(index: &mut ProjectTextIndex, doc: TextDocRecord) {
    let doc_id = doc.id;

    for (term, tf) in &doc.term_freq {
        index
            .postings
            .entry(term.clone())
            .or_default()
            .insert(doc_id, *tf);
    }

    index.total_doc_len += doc.doc_len;
    index.path_to_doc_id.insert(doc.file_path.clone(), doc_id);
    index.docs.insert(doc_id, doc);
}

fn remove_doc_by_path(index: &mut ProjectTextIndex, file_path: &str) {
    if let Some(doc_id) = index.path_to_doc_id.get(file_path).copied() {
        remove_doc_by_id(index, doc_id);
    }
}

fn remove_doc_by_id(index: &mut ProjectTextIndex, doc_id: u32) {
    let Some(old_doc) = index.docs.remove(&doc_id) else {
        return;
    };

    index.path_to_doc_id.remove(&old_doc.file_path);
    index.total_doc_len = index.total_doc_len.saturating_sub(old_doc.doc_len);

    for (term, _) in &old_doc.term_freq {
        if let Some(posting) = index.postings.get_mut(term) {
            posting.remove(&doc_id);
            if posting.is_empty() {
                index.postings.remove(term);
            }
        }
    }
}

fn score_candidates(
    index: &ProjectTextIndex,
    query: &str,
    filter: &SearchFilter,
    max_candidates: usize,
) -> Vec<IndexedCandidate> {
    let query_terms = tokenizer().tokenize(query);
    if query_terms.is_empty() || index.docs.is_empty() {
        return Vec::new();
    }

    let num_docs = index.docs.len();
    let avgdl = if num_docs == 0 {
        1.0
    } else {
        index.total_doc_len.max(1) as f64 / num_docs as f64
    };

    let mut score_by_doc: HashMap<u32, f64> = HashMap::new();
    let mut terms_by_doc: HashMap<u32, Vec<String>> = HashMap::new();

    for term in &query_terms {
        let Some(posting) = index.postings.get(term) else {
            continue;
        };

        let df = posting.len();
        let idf = compute_idf(num_docs, df);

        for (doc_id, tf_u32) in posting {
            let Some(doc) = index.docs.get(doc_id) else {
                continue;
            };
            if !matches_filter(&doc.file_path, &doc.language, filter) {
                continue;
            }

            let tf = *tf_u32 as f64;
            let doc_len = doc.doc_len.max(1) as f64;
            let tf_norm = (tf * (BM25_K1 + 1.0))
                / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avgdl)));

            *score_by_doc.entry(*doc_id).or_insert(0.0) += idf * tf_norm;
            terms_by_doc
                .entry(*doc_id)
                .or_default()
                .push(term.clone());
        }
    }

    let mut scored: Vec<(u32, f64)> = score_by_doc.into_iter().filter(|(_, score)| *score > 0.0).collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut out = Vec::new();
    for (doc_id, score) in scored.into_iter().take(max_candidates.max(1)) {
        let Some(doc) = index.docs.get(&doc_id) else {
            continue;
        };
        let matched_terms = terms_by_doc.remove(&doc_id).unwrap_or_default();

        out.push(IndexedCandidate {
            file_path: doc.file_path.clone(),
            language: doc.language.clone(),
            score,
            matched_terms,
        });
    }

    out
}

fn compute_idf(num_docs: usize, doc_freq: usize) -> f64 {
    let n = num_docs as f64;
    let df = doc_freq as f64;
    let numerator = (n - df) + 0.5;
    let denominator = df + 0.5;
    (1.0 + (numerator / denominator)).ln()
}

fn count_filtered_docs(index: &ProjectTextIndex, filter: &SearchFilter) -> usize {
    index
        .docs
        .values()
        .filter(|doc| matches_filter(&doc.file_path, &doc.language, filter))
        .count()
}

fn matches_filter(path: &str, lang_name: &str, filter: &SearchFilter) -> bool {
    let path_lower = path.to_lowercase().replace('\\', "/");

    if let Some(ref pat) = filter.file_pattern {
        let pat_lower = pat.to_lowercase();
        if !path_lower.contains(&pat_lower) {
            return false;
        }
    }

    if let Some(ref ext) = filter.ext_filter {
        let ext_lower = ext.to_lowercase();
        let ext_with_dot = if ext_lower.starts_with('.') {
            ext_lower
        } else {
            format!(".{ext_lower}")
        };
        if !path_lower.ends_with(&ext_with_dot) {
            return false;
        }
    }

    if let Some(ref dir) = filter.dir_filter {
        let dir_lower = dir.to_lowercase().replace('\\', "/");
        if !path_lower.contains(&dir_lower) {
            return false;
        }
    }

    if let Some(ref lang) = filter.lang_filter {
        let lang_lower = lang.to_lowercase();
        if lang_name.to_lowercase() != lang_lower {
            return false;
        }
    }

    true
}

fn config_fingerprint(specs: &[LanguageSpec], respect_gitignore: bool) -> u64 {
    let mut hasher = DefaultHasher::new();
    TEXT_INDEX_SCHEMA_VERSION.hash(&mut hasher);
    respect_gitignore.hash(&mut hasher);

    let mut lang_names: Vec<&str> = specs.iter().map(|spec| spec.name).collect();
    lang_names.sort_unstable();
    for lang in lang_names {
        lang.hash(&mut hasher);
    }

    hasher.finish()
}

fn persistence_path(root: &Path) -> Result<PathBuf, String> {
    let dir = root.join(".quickcontext");
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create text index dir: {e}"))?;
    Ok(dir.join("text_search_index.redb"))
}

fn tokenizer() -> &'static Tokenizer {
    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(Tokenizer::new)
}

fn persist_index(root: &Path, index: &ProjectTextIndex) -> Result<(), String> {
    let db_path = persistence_path(root)?;
    let db = Database::create(&db_path).map_err(|e| format!("failed to open redb: {e}"))?;

    let mut docs: Vec<TextDocRecord> = index.docs.values().cloned().collect();
    docs.sort_by_key(|d| d.id);

    let persisted = PersistedTextIndex {
        schema_version: TEXT_INDEX_SCHEMA_VERSION,
        config_fingerprint: index.config_fingerprint,
        project_root: index.project_root.clone(),
        respect_gitignore: index.respect_gitignore,
        file_signatures: index.file_signatures.clone(),
        docs,
        next_doc_id: index.next_doc_id,
    };

    let encoded = serde_json::to_vec(&persisted)
        .map_err(|e| format!("failed to encode text index: {e}"))?;

    let txn = db
        .begin_write()
        .map_err(|e| format!("failed to begin write txn: {e}"))?;
    {
        let mut table = txn
            .open_table(SNAPSHOT_TABLE)
            .map_err(|e| format!("failed to open snapshot table: {e}"))?;
        table
            .insert(STATE_KEY, encoded.as_slice())
            .map_err(|e| format!("failed to write snapshot: {e}"))?;
    }
    txn.commit()
        .map_err(|e| format!("failed to commit snapshot: {e}"))
}

fn load_persisted_index(root: &Path) -> Result<Option<PersistedTextIndex>, String> {
    let db_path = persistence_path(root)?;
    let db = Database::create(&db_path).map_err(|e| format!("failed to open redb: {e}"))?;
    let txn = db
        .begin_read()
        .map_err(|e| format!("failed to begin read txn: {e}"))?;

    let table = match txn.open_table(SNAPSHOT_TABLE) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let raw = match table.get(STATE_KEY) {
        Ok(Some(v)) => v,
        Ok(None) => return Ok(None),
        Err(_) => return Ok(None),
    };

    let decoded: PersistedTextIndex = serde_json::from_slice(raw.value())
        .map_err(|e| format!("failed to decode text index: {e}"))?;
    Ok(Some(decoded))
}

fn hydrate_index(persisted: PersistedTextIndex) -> ProjectTextIndex {
    let mut docs = HashMap::new();
    let mut path_to_doc_id = HashMap::new();
    let mut postings: HashMap<String, HashMap<u32, u32>> = HashMap::new();
    let mut total_doc_len = 0usize;

    for doc in persisted.docs {
        total_doc_len += doc.doc_len;
        path_to_doc_id.insert(doc.file_path.clone(), doc.id);

        for (term, tf) in &doc.term_freq {
            postings
                .entry(term.clone())
                .or_default()
                .insert(doc.id, *tf);
        }

        docs.insert(doc.id, doc);
    }

    ProjectTextIndex {
        project_root: persisted.project_root,
        config_fingerprint: persisted.config_fingerprint,
        respect_gitignore: persisted.respect_gitignore,
        file_signatures: persisted.file_signatures,
        docs,
        path_to_doc_id,
        postings,
        total_doc_len,
        next_doc_id: persisted.next_doc_id,
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
    fn test_matches_filter_extension() {
        let mut filter = SearchFilter::default();
        filter.ext_filter = Some("rs".to_string());

        assert!(matches_filter("src/main.rs", "rust", &filter));
        assert!(!matches_filter("src/main.py", "python", &filter));
    }

    #[test]
    fn test_config_fingerprint_stable_and_sensitive() {
        let specs = crate::lang::registry();
        let fp_a = config_fingerprint(&specs, true);
        let fp_b = config_fingerprint(&specs, true);
        let fp_c = config_fingerprint(&specs, false);

        assert_eq!(fp_a, fp_b);
        assert_ne!(fp_a, fp_c);
    }

}
