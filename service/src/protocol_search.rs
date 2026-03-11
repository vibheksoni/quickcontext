use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use ignore::WalkBuilder;

use crate::lang::{detect_language, LanguageSpec};
use crate::types::{
    ProtocolContract,
    ProtocolEvidence,
    ProtocolInputField,
    ProtocolOutputField,
    ProtocolSearchResult,
};

const QUICK_IGNORE_FILENAME: &str = ".quick-ignore";
const DEFAULT_CONTEXT_RADIUS: usize = 28;
const DEFAULT_MIN_SCORE: f64 = 8.0;
const DEFAULT_MAX_INPUT_FIELDS: usize = 16;
const DEFAULT_MAX_OUTPUT_FIELDS: usize = 12;

const BASE_MARKERS: &[&str] = &[
    "operationname",
    "operation_name",
    "graphql",
    "queryid",
    "endpoint",
    "\"method\"",
    "#[serde(rename",
    "http://",
    "https://",
    "api/",
    "post(",
    "get(",
    "put(",
    "delete(",
    "patch(",
];

#[derive(Debug, Clone, Default)]
pub struct ProtocolSearchOptions {
    pub context_radius: Option<usize>,
    pub min_score: Option<f64>,
    pub include_markers: Option<Vec<String>>,
    pub exclude_markers: Option<Vec<String>>,
    pub max_input_fields: Option<usize>,
    pub max_output_fields: Option<usize>,
}

#[derive(Debug, Clone)]
struct ResolvedOptions {
    context_radius: usize,
    min_score: f64,
    include_markers: Vec<String>,
    exclude_markers: Vec<String>,
    max_input_fields: usize,
    max_output_fields: usize,
}

pub fn protocol_search(
    query: &str,
    root: &Path,
    respect_gitignore: bool,
    limit: usize,
    specs: &[LanguageSpec],
    options: Option<&ProtocolSearchOptions>,
) -> Result<ProtocolSearchResult, String> {
    if query.trim().is_empty() {
        return Err("query cannot be empty".to_string());
    }
    if !root.exists() {
        return Err(format!("path does not exist: {}", root.display()));
    }

    let cfg = resolve_options(options);
    let started = Instant::now();
    let mut searched_files = 0usize;
    let mut contracts: Vec<ProtocolContract> = Vec::new();

    if root.is_file() {
        let language = detect_language(&root.to_string_lossy(), specs);
        if is_supported_source(language) {
            searched_files = 1;
            let source = std::fs::read_to_string(root).unwrap_or_default();
            contracts.extend(extract_contracts(&source, &root.to_string_lossy(), query, &cfg));
        }
    } else {
        let mut builder = WalkBuilder::new(root);
        builder
            .hidden(true)
            .add_custom_ignore_filename(QUICK_IGNORE_FILENAME)
            .threads(std::thread::available_parallelism().map_or(4, usize::from));

        if !respect_gitignore {
            builder.git_ignore(false).git_global(false).git_exclude(false);
        }

        for dent in builder.build() {
            let dent = match dent {
                Ok(d) => d,
                Err(_) => continue,
            };
            let path = dent.path();
            if !path.is_file() {
                continue;
            }

            let language = detect_language(&path.to_string_lossy(), specs);
            if !is_supported_source(language) {
                continue;
            }

            searched_files += 1;
            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            contracts.extend(extract_contracts(&source, &path.to_string_lossy(), query, &cfg));
        }
    }

    contracts.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let effective_limit = limit.max(1);
    let truncated = contracts.len() > effective_limit;
    if contracts.len() > effective_limit {
        contracts.truncate(effective_limit);
    }

    Ok(ProtocolSearchResult {
        contracts,
        searched_files,
        duration_ms: started.elapsed().as_millis(),
        truncated,
    })
}

fn resolve_options(options: Option<&ProtocolSearchOptions>) -> ResolvedOptions {
    let opts = options.cloned().unwrap_or_default();

    ResolvedOptions {
        context_radius: opts
            .context_radius
            .unwrap_or(DEFAULT_CONTEXT_RADIUS)
            .clamp(4, 120),
        min_score: opts.min_score.unwrap_or(DEFAULT_MIN_SCORE).max(0.0),
        include_markers: normalize_markers(opts.include_markers.as_deref()),
        exclude_markers: normalize_markers(opts.exclude_markers.as_deref()),
        max_input_fields: opts
            .max_input_fields
            .unwrap_or(DEFAULT_MAX_INPUT_FIELDS)
            .clamp(1, 64),
        max_output_fields: opts
            .max_output_fields
            .unwrap_or(DEFAULT_MAX_OUTPUT_FIELDS)
            .clamp(1, 64),
    }
}

fn normalize_markers(markers: Option<&[String]>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    if let Some(values) = markers {
        for marker in values {
            let trimmed = marker.trim().to_ascii_lowercase();
            if trimmed.len() < 2 {
                continue;
            }
            if seen.insert(trimmed.clone()) {
                out.push(trimmed);
            }
        }
    }

    out
}

fn is_supported_source(language: Option<&LanguageSpec>) -> bool {
    language.is_some()
}

fn extract_contracts(
    source: &str,
    file_path: &str,
    query: &str,
    cfg: &ResolvedOptions,
) -> Vec<ProtocolContract> {
    if !contains_protocol_markers(source, cfg) {
        return Vec::new();
    }

    let mut out: Vec<ProtocolContract> = Vec::new();
    let mut seen: HashSet<(String, String, Option<String>)> = HashSet::new();

    let query_terms = extract_query_terms(query);
    let lines: Vec<&str> = source.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        let line_no = idx + 1;
        let lower = line.to_ascii_lowercase();

        if lower.contains("operationname") || lower.contains("operation_name") {
            if let Some(operation_name) = parse_operation_name(line) {
                let window = collect_window(&lines, idx, cfg.context_radius);
                let input_fields = collect_generic_input_fields(&window, cfg.max_input_fields);
                let output_fields = collect_generic_output_fields(&window, cfg.max_output_fields);
                let has_outputs = !output_fields.is_empty();
                let matched = matched_terms(&query_terms, &window);
                if !query_terms.is_empty() && matched.is_empty() {
                    continue;
                }

                let mut score = 8.0 + matched.len() as f64;
                if !input_fields.is_empty() {
                    score += 2.0;
                }
                if has_outputs {
                    score += 1.0;
                }
                if line.len() > 1000 && matched.len() <= 1 {
                    score -= 3.0;
                }

                let operation_lower = operation_name.to_ascii_lowercase();
                let key = (
                    "graphql".to_string(),
                    operation_lower.clone(),
                    Some(operation_name.clone()),
                );
                if score >= cfg.min_score && seen.insert(key) {
                    out.push(ProtocolContract {
                        file_path: file_path.to_string(),
                        operation: operation_lower,
                        transport: "graphql".to_string(),
                        endpoint: None,
                        operation_name: Some(operation_name),
                        input_fields,
                        output_fields,
                        evidence: vec![ProtocolEvidence {
                            kind: "graphql_operation".to_string(),
                            line: line_no,
                            text: trim_line(line),
                        }],
                        matched_terms: matched,
                        score,
                        confidence: confidence_score("graphql", true, has_outputs),
                    });
                }
            }
        }

        if lower.contains("#[serde(rename") {
            if let Some(method) = parse_serde_rename(line) {
                let window = collect_window(&lines, idx, cfg.context_radius.min(24));
                let input_fields = collect_generic_input_fields(&window, cfg.max_input_fields);
                let output_fields = collect_generic_output_fields(&window, cfg.max_output_fields);
                let has_outputs = !output_fields.is_empty();
                let matched = matched_terms(&query_terms, &window);
                if !query_terms.is_empty() && matched.is_empty() {
                    continue;
                }

                let score = 7.0
                    + matched.len() as f64
                    + if !input_fields.is_empty() { 2.0 } else { 0.0 }
                    + if has_outputs { 1.0 } else { 0.0 };

                let method_lower = method.to_ascii_lowercase();
                let key = (
                    "ipc".to_string(),
                    method_lower.clone(),
                    Some(method.clone()),
                );
                if score >= cfg.min_score && seen.insert(key) {
                    out.push(ProtocolContract {
                        file_path: file_path.to_string(),
                        operation: method_lower,
                        transport: "ipc".to_string(),
                        endpoint: None,
                        operation_name: Some(method),
                        input_fields,
                        output_fields,
                        evidence: vec![ProtocolEvidence {
                            kind: "ipc_method".to_string(),
                            line: line_no,
                            text: trim_line(line),
                        }],
                        matched_terms: matched,
                        score,
                        confidence: confidence_score("ipc", true, has_outputs),
                    });
                }
            }
        }

        if lower.contains("\"method\":") {
            if let Some(method) = parse_json_method(line) {
                let window = collect_window(&lines, idx, cfg.context_radius.min(24));
                let input_fields = collect_generic_input_fields(&window, cfg.max_input_fields);
                let output_fields = collect_generic_output_fields(&window, cfg.max_output_fields);
                let has_outputs = !output_fields.is_empty();
                let matched = matched_terms(&query_terms, &window);
                if !query_terms.is_empty() && matched.is_empty() {
                    continue;
                }

                let score = 7.0
                    + matched.len() as f64
                    + if !input_fields.is_empty() { 2.0 } else { 0.0 }
                    + if has_outputs { 1.0 } else { 0.0 };

                let method_lower = method.to_ascii_lowercase();
                let key = (
                    "ipc".to_string(),
                    method_lower.clone(),
                    Some(method.clone()),
                );
                if score >= cfg.min_score && seen.insert(key) {
                    out.push(ProtocolContract {
                        file_path: file_path.to_string(),
                        operation: method_lower,
                        transport: "ipc".to_string(),
                        endpoint: None,
                        operation_name: Some(method),
                        input_fields,
                        output_fields,
                        evidence: vec![ProtocolEvidence {
                            kind: "payload_method".to_string(),
                            line: line_no,
                            text: trim_line(line),
                        }],
                        matched_terms: matched,
                        score,
                        confidence: confidence_score("ipc", true, has_outputs),
                    });
                }
            }
        }

        if is_http_signal(&lower) {
            let window = collect_window(&lines, idx, cfg.context_radius);
            let endpoint = parse_endpoint(line).or_else(|| find_endpoint_in_window(&window));
            if let Some(ep) = endpoint {
                let input_fields = collect_generic_input_fields(&window, cfg.max_input_fields);
                let output_fields = collect_generic_output_fields(&window, cfg.max_output_fields);
                let has_outputs = !output_fields.is_empty();
                let matched = matched_terms(&query_terms, &window);
                if !query_terms.is_empty() && matched.is_empty() {
                    continue;
                }

                let transport = detect_transport(&lower, &ep);
                let operation = operation_from_endpoint(&ep);
                let mut score = 9.0 + matched.len() as f64;
                if !input_fields.is_empty() {
                    score += 2.0;
                }
                if has_outputs {
                    score += 1.0;
                }
                if line.len() > 1000 && matched.len() <= 1 {
                    score -= 3.0;
                }

                let key = (transport.clone(), operation.clone(), None);
                if score >= cfg.min_score && seen.insert(key) {
                    out.push(ProtocolContract {
                        file_path: file_path.to_string(),
                        operation,
                        transport: transport.clone(),
                        endpoint: Some(ep),
                        operation_name: None,
                        input_fields,
                        output_fields,
                        evidence: vec![ProtocolEvidence {
                            kind: "transport_call".to_string(),
                            line: line_no,
                            text: trim_line(line),
                        }],
                        matched_terms: matched,
                        score,
                        confidence: confidence_score(&transport, true, has_outputs),
                    });
                }
            }
        }
    }

    out
}

fn contains_protocol_markers(source: &str, cfg: &ResolvedOptions) -> bool {
    let lower = source.to_ascii_lowercase();

    if cfg
        .exclude_markers
        .iter()
        .any(|marker| lower.contains(marker.as_str()))
    {
        return false;
    }

    if BASE_MARKERS.iter().any(|marker| lower.contains(marker)) {
        return true;
    }

    cfg.include_markers
        .iter()
        .any(|marker| lower.contains(marker.as_str()))
}

fn is_http_signal(lower_line: &str) -> bool {
    lower_line.contains("http://")
        || lower_line.contains("https://")
        || lower_line.contains("api/")
        || lower_line.contains("post(")
        || lower_line.contains("get(")
        || lower_line.contains("put(")
        || lower_line.contains("delete(")
        || lower_line.contains("patch(")
}

fn collect_window(lines: &[&str], center_idx: usize, radius: usize) -> String {
    let start = center_idx.saturating_sub(radius);
    let end = (center_idx + radius + 1).min(lines.len());
    lines[start..end].join("\n")
}

fn collect_generic_input_fields(window: &str, max_fields: usize) -> Vec<ProtocolInputField> {
    let mut out: Vec<ProtocolInputField> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for line in window.lines() {
        if let Some(name) = extract_candidate_field_name(line) {
            if is_likely_input_field(&name) && seen.insert(name.clone()) {
                out.push(ProtocolInputField {
                    name,
                    required: true,
                });
                if out.len() >= max_fields {
                    break;
                }
            }
        }
    }

    out
}

fn collect_generic_output_fields(window: &str, max_fields: usize) -> Vec<ProtocolOutputField> {
    let mut out: Vec<ProtocolOutputField> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for line in window.lines() {
        let lower = line.to_ascii_lowercase();

        if lower.contains("response") || lower.contains("result") || lower.contains("data") {
            for path in quoted_paths(line) {
                if looks_like_output_path(&path) && seen.insert(path.clone()) {
                    out.push(ProtocolOutputField { path });
                    if out.len() >= max_fields {
                        return out;
                    }
                }
            }

            if let Some(name) = extract_candidate_field_name(line) {
                if is_likely_output_field(&name) && seen.insert(name.clone()) {
                    out.push(ProtocolOutputField { path: name });
                    if out.len() >= max_fields {
                        return out;
                    }
                }
            }
        }
    }

    out
}

fn extract_candidate_field_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
        return None;
    }

    let key_part = if let Some((left, _)) = trimmed.split_once(':') {
        left
    } else if let Some((left, _)) = trimmed.split_once('=') {
        left
    } else {
        return None;
    };

    let mut key = key_part
        .trim()
        .trim_matches(',')
        .trim_matches('{')
        .trim_matches('}')
        .trim_matches('(')
        .trim_matches(')')
        .trim_matches('"')
        .trim_matches('\'')
        .to_string();

    for prefix in ["pub ", "let ", "const ", "var ", "mut "] {
        if key.starts_with(prefix) {
            key = key[prefix.len()..].trim().to_string();
        }
    }

    if key.len() < 2 || key.len() > 64 {
        return None;
    }
    if !key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        return None;
    }

    let lower = key.to_ascii_lowercase();
    if [
        "if",
        "for",
        "while",
        "match",
        "return",
        "query",
        "mutation",
        "method",
        "path",
        "url",
        "endpoint",
    ]
    .contains(&lower.as_str())
    {
        return None;
    }

    Some(key)
}

fn is_likely_input_field(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    !(lower.contains("response")
        || lower.contains("result")
        || lower.contains("output")
        || lower.contains("error"))
}

fn is_likely_output_field(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("response")
        || lower.contains("result")
        || lower.contains("output")
        || lower.contains("data")
        || lower.contains("payload")
}

fn looks_like_output_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.contains('.')
        || lower.contains("response")
        || lower.contains("result")
        || lower.contains("data")
}

fn parse_operation_name(line: &str) -> Option<String> {
    for marker in ["operationName", "operation_name"] {
        if let Some(value) = parse_quoted_value_after(line, marker) {
            return Some(value);
        }
    }
    None
}

fn parse_serde_rename(line: &str) -> Option<String> {
    parse_quoted_value_after(line, "rename")
}

fn parse_json_method(line: &str) -> Option<String> {
    parse_quoted_value_after(line, "\"method\"")
}

fn parse_quoted_value_after(line: &str, marker: &str) -> Option<String> {
    let marker_idx = line.find(marker)?;
    let tail = &line[marker_idx + marker.len()..];
    let mut quote_pos: Option<(usize, char)> = None;

    for (idx, ch) in tail.char_indices() {
        if ch == '"' || ch == '\'' {
            quote_pos = Some((idx, ch));
            break;
        }
    }

    let (start_idx, quote) = quote_pos?;
    let quoted = &tail[start_idx + 1..];
    let end_idx = quoted.find(quote)?;
    let value = quoted[..end_idx].trim();

    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn parse_endpoint(line: &str) -> Option<String> {
    for candidate in quoted_paths(line) {
        if candidate.starts_with("http://") || candidate.starts_with("https://") {
            if candidate.len() > 10 {
                return Some(candidate);
            }
        }
    }

    for candidate in quoted_paths(line) {
        if candidate == "http://" || candidate == "https://" {
            continue;
        }
        if candidate.contains('/') && !candidate.contains(' ') && candidate.len() > 3 {
            return Some(candidate);
        }
    }

    None
}

fn find_endpoint_in_window(window: &str) -> Option<String> {
    for line in window.lines() {
        if let Some(endpoint) = parse_endpoint(line) {
            return Some(endpoint);
        }
    }
    None
}

fn quoted_paths(line: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    for quote in ['"', '\''] {
        let mut cursor = 0usize;
        while let Some(start_rel) = line[cursor..].find(quote) {
            let start = cursor + start_rel + 1;
            let remaining = &line[start..];
            if let Some(end_rel) = remaining.find(quote) {
                let end = start + end_rel;
                let val = line[start..end].trim();
                if !val.is_empty() {
                    out.push(val.to_string());
                }
                cursor = end + 1;
            } else {
                break;
            }
        }
    }

    out
}

fn detect_transport(lower_line: &str, endpoint: &str) -> String {
    let endpoint_lower = endpoint.to_ascii_lowercase();
    if lower_line.contains("graphql") || endpoint_lower.contains("graphql") {
        return "graphql".to_string();
    }
    if lower_line.contains("post(") {
        return "rest_post".to_string();
    }
    if lower_line.contains("put(") {
        return "rest_put".to_string();
    }
    if lower_line.contains("patch(") {
        return "rest_patch".to_string();
    }
    if lower_line.contains("delete(") {
        return "rest_delete".to_string();
    }
    if lower_line.contains("get(") {
        return "rest_get".to_string();
    }
    "rest".to_string()
}

fn operation_from_endpoint(endpoint: &str) -> String {
    let cleaned = endpoint.trim().trim_matches('/');
    if cleaned.is_empty() {
        return "request".to_string();
    }

    let segment = cleaned
        .split('/')
        .rev()
        .find(|s| !s.is_empty())
        .unwrap_or("request");

    let mut op = segment
        .trim_end_matches(".json")
        .trim_end_matches(".xml")
        .trim_end_matches(".js")
        .to_ascii_lowercase();

    op = op
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();

    while op.contains("__") {
        op = op.replace("__", "_");
    }

    op.trim_matches('_').chars().take(80).collect()
}

fn extract_query_terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

fn matched_terms(query_terms: &[String], haystack: &str) -> Vec<String> {
    let lower = haystack.to_ascii_lowercase();
    let mut out: Vec<String> = Vec::new();
    for term in query_terms {
        if lower.contains(term) {
            out.push(term.clone());
        }
    }
    out
}

fn confidence_score(transport: &str, has_inputs: bool, has_outputs: bool) -> f64 {
    let mut score: f64 = match transport {
        "graphql" => 0.84,
        "ipc" => 0.82,
        _ => 0.8,
    };

    if has_inputs {
        score += 0.08;
    }
    if has_outputs {
        score += 0.07;
    }
    score.min(0.99)
}

fn trim_line(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.len() > 220 {
        format!("{}...", &trimmed[..220])
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{extract_contracts, ProtocolSearchOptions, ResolvedOptions};

    fn cfg() -> ResolvedOptions {
        super::resolve_options(Some(&ProtocolSearchOptions::default()))
    }

    #[test]
    fn extracts_graphql_operation_generically() {
        let source = r#"
const req = {
  operationName: "GetUsers",
  variables: { userIds: ["1"] },
  data: "users.result.items"
};
"#;

        let contracts = extract_contracts(source, "fixture.ts", "get users", &cfg());
        assert!(contracts
            .iter()
            .any(|c| c.transport == "graphql" && c.operation_name.as_deref() == Some("GetUsers")));
    }

    #[test]
    fn extracts_ipc_method_from_serde_and_payload() {
        let source = r#"
#[serde(rename = "symbol_lookup")]
enum Request {
  SymbolLookup { query: String, path: Option<String> }
}

let payload = { "method": "symbol_lookup", "query": "handler" };
"#;

        let contracts = extract_contracts(source, "fixture.rs", "symbol lookup", &cfg());
        assert!(contracts
            .iter()
            .any(|c| c.transport == "ipc" && c.operation == "symbol_lookup"));
    }

    #[test]
    fn extracts_http_endpoint_generically() {
        let source = r#"
client.post("/api/v1/messages/send", { recipient_id: uid, body: text });
let response = result.data.message;
"#;

        let contracts = extract_contracts(source, "fixture.js", "send message", &cfg());
        assert!(contracts.iter().any(|c| c.transport == "rest_post"
            && c.endpoint.as_deref() == Some("/api/v1/messages/send")));
    }

    #[test]
    fn respects_min_score_option() {
        let source = r#"
let payload = { "method": "ping" };
"#;

        let options = ProtocolSearchOptions {
            min_score: Some(20.0),
            ..ProtocolSearchOptions::default()
        };
        let contracts = extract_contracts(
            source,
            "fixture.rs",
            "ping",
            &super::resolve_options(Some(&options)),
        );

        assert!(contracts.is_empty());
    }
}
