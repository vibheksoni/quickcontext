use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::intent::{build_intent_terms, normalize_intent_level, IntentTerms};
use crate::lang::LanguageSpec;
use crate::query_parser::{ParsedQuery, QueryExpr, QueryParser};
use crate::ranking::Bm25Scorer;
use crate::structural_boost;
use crate::text_index::{search_candidates, IndexedCandidate};
use crate::types::{TextSearchMatch, TextSearchResult};

const SNIPPET_CONTEXT_LINES: usize = 3;
const TOOLING_FILE_PENALTY: f64 = 0.15;
const DOC_FILE_QUERY_PENALTY: f64 = 0.45;
const TEST_FILE_QUERY_PENALTY: f64 = 0.45;
const GENERATED_BUNDLE_FILE_PENALTY: f64 = 0.18;
const GENERIC_WRAPPER_FILE_PENALTY: f64 = 0.78;
const GENERIC_ENTRYPOINT_FILE_PENALTY: f64 = 0.72;
const PATH_KEYWORD_BASE_BOOST: f64 = 0.16;
const PATH_KEYWORD_BASENAME_BOOST: f64 = 0.10;
const PATH_KEYWORD_MAX_BOOST: f64 = 1.42;


/// Full-text search over files using persistent BM25 candidates and structural reranking.
///
/// query_str: &str — Raw query string with optional operators and filters.
/// root: &Path — Directory or file to search within.
/// respect_gitignore: bool — Whether to honor .gitignore rules.
/// limit: usize — Maximum number of results to return.
/// specs: &[LanguageSpec] — Language specs for file detection.
/// intent_mode: bool — When true, enable non-embedding intent expansion for simple queries.
/// intent_level: u8 — Intent expansion aggressiveness from 1..=3.
pub fn text_search(
    query_str: &str,
    root: &Path,
    respect_gitignore: bool,
    limit: usize,
    specs: &[LanguageSpec],
    intent_mode: bool,
    intent_level: u8,
) -> Result<TextSearchResult, String> {
    if query_str.trim().is_empty() {
        return Err("query cannot be empty".to_string());
    }
    if !root.exists() {
        return Err(format!("path does not exist: {}", root.display()));
    }

    let start = Instant::now();
    let effective_limit = limit.max(1);

    let parser = QueryParser::new();
    let parsed = parser.parse(query_str);

    if parsed.expr.is_none() {
        return Err("query produced no searchable terms".to_string());
    }

    let normalized_intent_level = normalize_intent_level(intent_level);
    let intent_terms = if intent_mode {
        build_intent_terms(query_str, normalized_intent_level)
    } else {
        IntentTerms {
            exact_terms: Vec::new(),
            expanded_terms: Vec::new(),
        }
    };

    let search_query = if intent_mode {
        build_bm25_query(&parsed.raw, &intent_terms)
    } else {
        parsed.raw.clone()
    };

    let max_candidates = effective_limit.saturating_mul(4).max(effective_limit + 2);
    let (candidates, searched_files) = search_candidates(
        root,
        specs,
        respect_gitignore,
        &parsed.filter,
        &search_query,
        max_candidates,
    )?;

    if candidates.is_empty() {
        return Ok(TextSearchResult {
            matches: Vec::new(),
            searched_files,
            query_expr: format!("{:?}", parsed.expr),
            duration_ms: start.elapsed().as_millis(),
            truncated: false,
        });
    }

    let mut results = score_and_rank(
        &parsed,
        &candidates,
        effective_limit,
        intent_mode,
        normalized_intent_level,
        &intent_terms,
    );
    let truncated = results.len() > effective_limit;
    results.truncate(effective_limit);

    Ok(TextSearchResult {
        matches: results,
        searched_files,
        query_expr: format!("{:?}", parsed.expr),
        duration_ms: start.elapsed().as_millis(),
        truncated,
    })
}


/// Score indexed candidates and rerank by structural and intent-aware boosts.
///
/// parsed: &ParsedQuery — Parsed query with expression tree and filters.
/// candidates: &[IndexedCandidate] — Candidate files from persistent index.
/// limit: usize — Max results to return (fetch extra for truncation detection).
/// intent_mode: bool — Enable intent-aware expansion/reranking.
/// intent_level: u8 — Intent expansion aggressiveness from 1..=3.
/// intent_terms: &IntentTerms — Precomputed exact and expanded terms for this query.
fn score_and_rank(
    parsed: &ParsedQuery,
    candidates: &[IndexedCandidate],
    limit: usize,
    intent_mode: bool,
    intent_level: u8,
    intent_terms: &IntentTerms,
) -> Vec<TextSearchMatch> {
    let expr = match &parsed.expr {
        Some(e) => e,
        None => return Vec::new(),
    };

    let scorer = Bm25Scorer::with_defaults();
    let query_terms: Vec<String> = if intent_mode {
        collect_query_terms(intent_terms, &scorer)
    } else {
        scorer.tokenizer().tokenize(&parsed.raw)
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

    if is_fast_path_expr(expr) {
        return score_and_rank_fast(
            candidates,
            limit,
            intent_mode,
            intent_level,
            &exact_term_set,
            &expanded_term_set,
            &query_terms,
        );
    }

    let mut matches = Vec::new();

    for candidate in candidates {
        let content = match read_candidate_content(&candidate.file_path) {
            Some(c) => c,
            None => continue,
        };

        if !evaluate_expr(expr, &content) {
            continue;
        }

        let category_boost =
            structural_boost::compute_category_boost(structural_boost::classify_file(&candidate.file_path));
        let coverage_boost = structural_boost::compute_coverage_boost(&query_terms, &content);
        let role_boost =
            structural_boost::compute_role_boost(structural_boost::classify_role_from_content(&content));
        let tooling_penalty = compute_tooling_path_penalty(&candidate.file_path, &query_terms);
        let path_boost = compute_path_keyword_boost(&candidate.file_path, &query_terms);

        let intent_boost = if intent_mode {
            compute_intent_score(
                &candidate.matched_terms,
                &exact_term_set,
                &expanded_term_set,
                intent_level,
            )
        } else {
            1.0
        };

        let boosted_score = candidate.score
            * category_boost
            * coverage_boost
            * role_boost
            * intent_boost
            * tooling_penalty
            * path_boost;
        let (snippet, line_start, line_end) = extract_snippet(&content, &candidate.matched_terms);

        matches.push(TextSearchMatch {
            file_path: candidate.file_path.clone(),
            score: boosted_score,
            matched_terms: candidate.matched_terms.clone(),
            snippet,
            snippet_line_start: line_start,
            snippet_line_end: line_end,
            language: candidate.language.clone(),
        });
    }

    matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    matches.truncate(limit + 1);
    matches
}

fn score_and_rank_fast(
    candidates: &[IndexedCandidate],
    limit: usize,
    intent_mode: bool,
    intent_level: u8,
    exact_term_set: &HashSet<String>,
    expanded_term_set: &HashSet<String>,
    query_terms: &[String],
) -> Vec<TextSearchMatch> {
    let mut scored: Vec<(usize, f64)> = Vec::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        let category_boost =
            structural_boost::compute_category_boost(structural_boost::classify_file(&candidate.file_path));
        let coverage_boost = compute_matched_term_coverage_boost(&candidate.matched_terms, query_terms);
        let tooling_penalty = compute_tooling_path_penalty(&candidate.file_path, query_terms);
        let path_boost = compute_path_keyword_boost(&candidate.file_path, query_terms);
        let intent_boost = if intent_mode {
            compute_intent_score(
                &candidate.matched_terms,
                exact_term_set,
                expanded_term_set,
                intent_level,
            )
        } else {
            1.0
        };

        scored.push((
            idx,
            candidate.score * category_boost * coverage_boost * intent_boost * tooling_penalty * path_boost,
        ));
    }

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut matches = Vec::new();
    for (idx, score) in scored.into_iter().take(limit + 1) {
        let candidate = &candidates[idx];
        let content = match read_candidate_content(&candidate.file_path) {
            Some(c) => c,
            None => continue,
        };

        let (snippet, line_start, line_end) = extract_snippet(&content, &candidate.matched_terms);

        matches.push(TextSearchMatch {
            file_path: candidate.file_path.clone(),
            score,
            matched_terms: candidate.matched_terms.clone(),
            snippet,
            snippet_line_start: line_start,
            snippet_line_end: line_end,
            language: candidate.language.clone(),
        });
    }

    matches
}

fn is_fast_path_expr(expr: &QueryExpr) -> bool {
    match expr {
        QueryExpr::Term { excluded, exact, .. } => !*excluded && !*exact,
        QueryExpr::And(_, _) => false,
        QueryExpr::Or(left, right) => is_fast_path_expr(left) && is_fast_path_expr(right),
    }
}

fn compute_matched_term_coverage_boost(matched_terms: &[String], query_terms: &[String]) -> f64 {
    if query_terms.is_empty() {
        return 1.0;
    }

    let query: HashSet<String> = query_terms.iter().map(|t| t.to_ascii_lowercase()).collect();
    let matched: HashSet<String> = matched_terms.iter().map(|t| t.to_ascii_lowercase()).collect();

    let overlap = query.iter().filter(|term| matched.contains(*term)).count();
    if overlap == 0 {
        return 1.0;
    }

    let ratio = overlap as f64 / query.len() as f64;
    1.0 + (ratio * 0.22)
}

fn read_candidate_content(path: &str) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    if content.contains('\0') {
        return None;
    }
    Some(content)
}

fn collect_query_terms(intent_terms: &IntentTerms, scorer: &Bm25Scorer) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();

    for term in &intent_terms.exact_terms {
        for token in scorer.tokenizer().tokenize(term) {
            if seen.insert(token.clone()) {
                terms.push(token);
            }
        }
    }

    for term in &intent_terms.expanded_terms {
        for token in scorer.tokenizer().tokenize(term) {
            if seen.insert(token.clone()) {
                terms.push(token);
            }
        }
    }

    terms
}

fn build_bm25_query(raw_query: &str, intent_terms: &IntentTerms) -> String {
    let mut query = raw_query.trim().to_string();
    for term in &intent_terms.expanded_terms {
        if !query.is_empty() {
            query.push(' ');
        }
        query.push_str(term);
    }
    query
}

fn compute_intent_score(
    matched_terms: &[String],
    exact_term_set: &HashSet<String>,
    expanded_term_set: &HashSet<String>,
    intent_level: u8,
) -> f64 {
    if exact_term_set.is_empty() && expanded_term_set.is_empty() {
        return 1.0;
    }

    let mut exact_hits = 0usize;
    let mut expanded_hits = 0usize;

    for term in matched_terms {
        let lower = term.to_ascii_lowercase();
        if exact_term_set.contains(&lower) {
            exact_hits += 1;
            continue;
        }
        if expanded_term_set.contains(&lower) {
            expanded_hits += 1;
        }
    }

    let exact_boost = if exact_hits > 0 {
        1.0 + (exact_hits as f64 * 0.22)
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
    let coverage_boost = 1.0 + (exact_coverage * 0.2) + (expanded_coverage * 0.08);

    let penalty = if exact_hits == 0 && expanded_hits > 0 {
        0.85
    } else {
        1.0
    };

    exact_boost * expanded_boost * coverage_boost * penalty
}


/// Evaluate a query expression tree against file content.
///
/// expr: &QueryExpr — Expression tree node to evaluate.
/// content: &str — File content to match against.
fn evaluate_expr(expr: &QueryExpr, content: &str) -> bool {
    let content_lower = content.to_lowercase();

    match expr {
        QueryExpr::Term {
            keywords,
            required: _,
            excluded,
            exact,
        } => {
            let found = if *exact {
                let phrase = keywords.join(" ");
                content_lower.contains(&phrase)
            } else {
                keywords.iter().any(|kw| content_lower.contains(kw))
            };

            if *excluded {
                !found
            } else {
                found
            }
        }
        QueryExpr::And(left, right) => evaluate_expr(left, content) && evaluate_expr(right, content),
        QueryExpr::Or(left, right) => evaluate_expr(left, content) || evaluate_expr(right, content),
    }
}


/// Extract a context snippet around the first matched term in the file.
///
/// content: &str — File content.
/// matched_terms: &[String] — Terms that matched in this file.
fn extract_snippet(content: &str, matched_terms: &[String]) -> (String, usize, usize) {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() || matched_terms.is_empty() {
        return (String::new(), 0, 0);
    }

    let mut best_line: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        let line_lower = line.to_lowercase();
        for term in matched_terms {
            if line_lower.contains(term) {
                best_line = Some(i);
                break;
            }
        }
        if best_line.is_some() {
            break;
        }
    }

    let center = best_line.unwrap_or(0);
    let start = center.saturating_sub(SNIPPET_CONTEXT_LINES);
    let end = (center + SNIPPET_CONTEXT_LINES + 1).min(lines.len());

    let snippet = lines[start..end].join("\n");
    (snippet, start + 1, end)
}

fn compute_tooling_path_penalty(path: &str, query_terms: &[String]) -> f64 {
    let normalized_path = path.to_ascii_lowercase().replace('\\', "/");
    let mut penalty = 1.0;
    let tooling_terms = [
        "benchmark",
        "latency",
        "coverage",
        "phase",
        "tooling",
        "script",
        "scripts",
        "instrumentation",
        "timing",
    ];
    let query_is_tooling = query_terms
        .iter()
        .any(|term| tooling_terms.iter().any(|tool| term == tool));
    let is_tooling_path = normalized_path.contains("/scripts/")
        || normalized_path.starts_with("scripts/")
        || normalized_path.contains("benchmark")
        || normalized_path.ends_with("_cases.json")
        || normalized_path.ends_with(".bench.rs");
    if is_tooling_path && !query_is_tooling {
        penalty *= TOOLING_FILE_PENALTY;
    }

    let doc_terms = ["doc", "docs", "documentation", "readme", "guide", "guides", "manual"];
    let query_is_doc_focused = query_terms
        .iter()
        .any(|term| doc_terms.iter().any(|doc| term == doc));
    let is_doc_path = normalized_path.ends_with(".md")
        || normalized_path.ends_with(".rst")
        || normalized_path.contains("/docs/")
        || normalized_path.contains("/doc/")
        || normalized_path.ends_with("readme")
        || normalized_path.ends_with("readme.md")
        || normalized_path.ends_with("ai_docs.md");
    if is_doc_path && !query_is_doc_focused {
        penalty *= DOC_FILE_QUERY_PENALTY;
    }

    let test_terms = ["test", "tests", "spec", "specs", "regression", "unittest", "pytest"];
    let query_is_test_focused = query_terms
        .iter()
        .any(|term| test_terms.iter().any(|test| term == test));
    let is_test_path = normalized_path.contains("/tests/")
        || normalized_path.contains("/test/")
        || normalized_path.contains("/__tests__/")
        || normalized_path.ends_with(".test.ts")
        || normalized_path.ends_with(".test.js")
        || normalized_path.ends_with("_test.py")
        || normalized_path.ends_with("_spec.py")
        || normalized_path.contains("test_regressions.py");
    if is_test_path && !query_is_test_focused {
        penalty *= TEST_FILE_QUERY_PENALTY;
    }

    if is_generated_bundle_path(&normalized_path) && !query_is_renderer_focused(query_terms) {
        penalty *= GENERATED_BUNDLE_FILE_PENALTY;
    }

    penalty * compute_wrapper_path_penalty(&normalized_path, query_terms)
}

fn compute_path_keyword_boost(path: &str, query_terms: &[String]) -> f64 {
    if query_terms.is_empty() {
        return 1.0;
    }

    let normalized_path = path.to_ascii_lowercase().replace('\\', "/");
    let query_set: HashSet<String> = query_terms
        .iter()
        .map(|term| term.to_ascii_lowercase())
        .filter(|term| term.len() >= 3)
        .collect();
    if query_set.is_empty() {
        return 1.0;
    }

    let path_tokens = tokenize_path_tokens(&normalized_path);
    if path_tokens.is_empty() {
        return 1.0;
    }

    let path_overlap = query_set.intersection(&path_tokens).count();
    if path_overlap == 0 {
        return 1.0;
    }

    let filename = Path::new(&normalized_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let basename_tokens = tokenize_path_tokens(filename);
    let basename_overlap = query_set.intersection(&basename_tokens).count();

    let mut boost = 1.0 + (path_overlap as f64 * PATH_KEYWORD_BASE_BOOST);
    boost += basename_overlap as f64 * PATH_KEYWORD_BASENAME_BOOST;
    boost.min(PATH_KEYWORD_MAX_BOOST)
}

fn tokenize_path_tokens(value: &str) -> HashSet<String> {
    let mut tokens = HashSet::new();
    let normalized = value
        .replace('\\', "/")
        .replace(['.', '-', '_', ':', '(', ')', '[', ']', ','], " ");

    for token in normalized.split(|c: char| c == '/' || c.is_whitespace()) {
        let lowered = token.trim().to_ascii_lowercase();
        if lowered.len() < 3 || is_low_signal_path_token(&lowered) {
            continue;
        }
        tokens.insert(lowered);
    }

    tokens
}

fn is_low_signal_path_token(token: &str) -> bool {
    matches!(
        token,
        "app"
            | "apps"
            | "dist"
            | "feature"
            | "features"
            | "lib"
            | "main"
            | "node"
            | "nodes"
            | "preload"
            | "renderer"
            | "shared"
            | "src"
            | "utils"
            | "immutable"
            | "chunk"
            | "chunks"
            | "worker"
            | "workers"
            | "index"
            | "file"
            | "files"
            | "service"
            | "manager"
            | "repository"
            | "module"
            | "js"
            | "jsx"
            | "ts"
            | "tsx"
            | "mjs"
            | "cjs"
            | "cts"
            | "map"
    )
}

fn is_generated_bundle_path(path: &str) -> bool {
    path.contains("/app/immutable/")
        || path.contains("/immutable/chunks/")
        || path.contains("/immutable/nodes/")
        || path.contains("/immutable/workers/")
}

fn query_is_renderer_focused(query_terms: &[String]) -> bool {
    let renderer_terms = [
        "asset",
        "assets",
        "chunk",
        "chunks",
        "component",
        "components",
        "css",
        "dom",
        "frontend",
        "front-end",
        "html",
        "renderer",
        "route",
        "routes",
        "svelte",
        "style",
        "styles",
        "ui",
        "worker",
        "workers",
    ];
    query_terms
        .iter()
        .any(|term| renderer_terms.iter().any(|candidate| term == candidate))
}

fn compute_wrapper_path_penalty(path: &str, query_terms: &[String]) -> f64 {
    if query_terms.len() < 3 {
        return 1.0;
    }

    let filename = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let basename = filename
        .split('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();

    let mentions = |terms: &[&str]| -> bool {
        query_terms
            .iter()
            .any(|term| terms.iter().any(|candidate| term == candidate))
    };

    let mut penalty = 1.0;

    let is_shallow_entry = filename == "index.js"
        && (path == "main/index.js"
            || path == "preload/index.js"
            || path == "shared/index.js"
            || path.ends_with("/main/index.js")
            || path.ends_with("/preload/index.js")
            || path.ends_with("/shared/index.js"));
    if is_shallow_entry && !mentions(&["index", "entry", "entrypoint", "bootstrap", "startup", "init", "initialize"]) {
        penalty *= GENERIC_ENTRYPOINT_FILE_PENALTY;
    }

    if filename.ends_with(".ipc.js") && !mentions(&["ipc", "channel", "channels", "invoke", "message", "messages"]) {
        penalty *= GENERIC_WRAPPER_FILE_PENALTY;
    }
    if basename.ends_with("tools") && !mentions(&["tool", "tools"]) {
        penalty *= GENERIC_WRAPPER_FILE_PENALTY;
    }
    if basename.ends_with("bridge") && !mentions(&["bridge", "bridges"]) {
        penalty *= GENERIC_WRAPPER_FILE_PENALTY;
    }
    if basename.ends_with("adapter") && !mentions(&["adapter", "adapters", "protocol", "protocols"]) {
        penalty *= GENERIC_WRAPPER_FILE_PENALTY;
    }
    if basename.ends_with("provider") && !mentions(&["provider", "providers", "model", "models"]) {
        penalty *= GENERIC_WRAPPER_FILE_PENALTY;
    }
    if matches!(basename.as_str(), "types" | "constants" | "schemas")
        && !mentions(&["type", "types", "constant", "constants", "schema", "schemas"])
    {
        penalty *= GENERIC_WRAPPER_FILE_PENALTY;
    }

    penalty
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_intent_score_prefers_exact_terms() {
        let exact_term_set: HashSet<String> = ["auth".to_string(), "token".to_string()].into_iter().collect();
        let expanded_term_set: HashSet<String> = ["authentication".to_string(), "jwt".to_string()].into_iter().collect();

        let exact_terms = vec!["auth".to_string(), "token".to_string()];
        let expanded_only_terms = vec!["authentication".to_string(), "jwt".to_string()];

        let exact_score = compute_intent_score(&exact_terms, &exact_term_set, &expanded_term_set, 2);
        let expanded_only_score = compute_intent_score(&expanded_only_terms, &exact_term_set, &expanded_term_set, 2);

        assert!(exact_score > expanded_only_score);
    }

    #[test]
    fn test_compute_intent_score_rewards_coverage_with_exact_hits() {
        let exact_term_set: HashSet<String> = ["auth".to_string(), "token".to_string()].into_iter().collect();
        let expanded_term_set: HashSet<String> = ["authentication".to_string(), "jwt".to_string()].into_iter().collect();

        let partial_terms = vec!["auth".to_string()];
        let high_coverage_terms = vec!["auth".to_string(), "token".to_string(), "authentication".to_string()];

        let partial_score = compute_intent_score(&partial_terms, &exact_term_set, &expanded_term_set, 2);
        let high_coverage_score = compute_intent_score(&high_coverage_terms, &exact_term_set, &expanded_term_set, 2);

        assert!(high_coverage_score > partial_score);
    }

    #[test]
    fn test_compute_tooling_path_penalty_penalizes_scripts_for_normal_queries() {
        let query_terms = vec!["python".to_string(), "connect".to_string(), "linux".to_string()];
        let penalty = compute_tooling_path_penalty("scripts/retrieval_benchmark.py", &query_terms);
        assert!(penalty < 1.0);
    }

    #[test]
    fn test_compute_tooling_path_penalty_keeps_scripts_for_tooling_queries() {
        let query_terms = vec!["benchmark".to_string(), "latency".to_string()];
        let penalty = compute_tooling_path_penalty("scripts/retrieval_benchmark.py", &query_terms);
        assert_eq!(penalty, 1.0);
    }

    #[test]
    fn test_compute_tooling_path_penalty_penalizes_docs_for_non_doc_queries() {
        let query_terms = vec!["python".to_string(), "connect".to_string(), "linux".to_string()];
        let penalty = compute_tooling_path_penalty("AI_DOCS.md", &query_terms);
        assert!(penalty < 1.0);
    }

    #[test]
    fn test_compute_tooling_path_penalty_keeps_docs_for_doc_queries() {
        let query_terms = vec!["docs".to_string(), "readme".to_string()];
        let penalty = compute_tooling_path_penalty("AI_DOCS.md", &query_terms);
        assert_eq!(penalty, 1.0);
    }

    #[test]
    fn test_compute_tooling_path_penalty_penalizes_tests_for_non_test_queries() {
        let query_terms = vec!["python".to_string(), "connect".to_string(), "linux".to_string()];
        let penalty = compute_tooling_path_penalty("engine/tests/test_regressions.py", &query_terms);
        assert!(penalty < 1.0);
    }

    #[test]
    fn test_compute_tooling_path_penalty_keeps_tests_for_test_queries() {
        let query_terms = vec!["test".to_string(), "regression".to_string()];
        let penalty = compute_tooling_path_penalty("engine/tests/test_regressions.py", &query_terms);
        assert_eq!(penalty, 1.0);
    }

    #[test]
    fn test_compute_tooling_path_penalty_penalizes_generated_renderer_chunks_for_non_ui_queries() {
        let query_terms = vec!["event".to_string(), "storage".to_string(), "notification".to_string()];
        let penalty = compute_tooling_path_penalty(
            "renderer/app/immutable/chunks/BLk-YuGw.js",
            &query_terms,
        );
        assert!(penalty < 1.0);
    }

    #[test]
    fn test_compute_tooling_path_penalty_keeps_generated_renderer_chunks_for_renderer_queries() {
        let query_terms = vec!["renderer".to_string(), "component".to_string(), "chunk".to_string()];
        let penalty = compute_tooling_path_penalty(
            "renderer/app/immutable/chunks/BLk-YuGw.js",
            &query_terms,
        );
        assert_eq!(penalty, 1.0);
    }

    #[test]
    fn test_compute_tooling_path_penalty_penalizes_generic_wrapper_files_for_non_wrapper_queries() {
        let query_terms = vec!["mcp".to_string(), "hub".to_string(), "restart".to_string(), "server".to_string()];
        let penalty = compute_tooling_path_penalty("main/index.js", &query_terms);
        assert!(penalty < 1.0);
    }

    #[test]
    fn test_compute_path_keyword_boost_rewards_descriptive_file_paths() {
        let query_terms = vec!["mcp".to_string(), "hub".to_string(), "server".to_string()];
        let boost = compute_path_keyword_boost(
            "features/mcp/main/hub/mcp-hub.js",
            &query_terms,
        );
        assert!(boost > 1.0);
    }

    #[test]
    fn test_evaluate_expr_simple_term() {
        let expr = QueryExpr::Term {
            keywords: vec!["authentication".to_string()],
            required: false,
            excluded: false,
            exact: false,
        };
        assert!(evaluate_expr(&expr, "user authentication handler"));
        assert!(!evaluate_expr(&expr, "database connection pool"));
    }

    #[test]
    fn test_evaluate_expr_excluded() {
        let expr = QueryExpr::Term {
            keywords: vec!["test".to_string()],
            required: false,
            excluded: true,
            exact: false,
        };
        assert!(!evaluate_expr(&expr, "this is a test file"));
        assert!(evaluate_expr(&expr, "production handler code"));
    }

    #[test]
    fn test_evaluate_expr_exact_phrase() {
        let expr = QueryExpr::Term {
            keywords: vec!["exact".to_string(), "match".to_string()],
            required: false,
            excluded: false,
            exact: true,
        };
        assert!(evaluate_expr(&expr, "find the exact match here"));
        assert!(!evaluate_expr(&expr, "match the exact value"));
    }

    #[test]
    fn test_evaluate_expr_and() {
        let expr = QueryExpr::And(
            Box::new(QueryExpr::Term {
                keywords: vec!["auth".to_string()],
                required: false,
                excluded: false,
                exact: false,
            }),
            Box::new(QueryExpr::Term {
                keywords: vec!["token".to_string()],
                required: false,
                excluded: false,
                exact: false,
            }),
        );
        assert!(evaluate_expr(&expr, "auth token validation"));
        assert!(!evaluate_expr(&expr, "auth handler only"));
    }

    #[test]
    fn test_evaluate_expr_or() {
        let expr = QueryExpr::Or(
            Box::new(QueryExpr::Term {
                keywords: vec!["auth".to_string()],
                required: false,
                excluded: false,
                exact: false,
            }),
            Box::new(QueryExpr::Term {
                keywords: vec!["token".to_string()],
                required: false,
                excluded: false,
                exact: false,
            }),
        );
        assert!(evaluate_expr(&expr, "auth handler"));
        assert!(evaluate_expr(&expr, "token validator"));
        assert!(!evaluate_expr(&expr, "database pool"));
    }

    #[test]
    fn test_extract_snippet_basic() {
        let lines = [
            "line 1",
            "line 2",
            "line 3 with auth",
            "line 4",
            "line 5",
        ];
        let content = lines.join("\n");
        let terms = vec!["auth".to_string()];
        let (snippet, start, end) = extract_snippet(&content, &terms);
        assert!(snippet.contains("auth"));
        assert!(start >= 1);
        assert!(end <= 5);
    }

    #[test]
    fn test_extract_snippet_empty() {
        let content = String::new();
        let terms = vec!["auth".to_string()];
        let (snippet, start, end) = extract_snippet(&content, &terms);
        assert!(snippet.is_empty());
        assert_eq!(start, 0);
        assert_eq!(end, 0);
    }
}
