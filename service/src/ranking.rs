use std::collections::HashMap;
use std::sync::OnceLock;

use crate::tokenizer::Tokenizer;

pub struct Bm25Config {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.5 }
    }
}

pub struct Bm25Scorer {
    config: Bm25Config,
}

pub struct ScoredDocument {
    pub doc_index: usize,
    pub score: f64,
    pub matched_terms: Vec<String>,
}

struct IdfEntry {
    term: String,
    idf: f64,
}

impl Bm25Scorer {
    pub fn new(config: Bm25Config) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(Bm25Config::default())
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        tokenizer()
    }

    fn compute_idf(num_docs: usize, doc_freq: usize) -> f64 {
        let n = num_docs as f64;
        let df = doc_freq as f64;
        let numerator = (n - df) + 0.5;
        let denominator = df + 0.5;
        (1.0 + (numerator / denominator)).ln()
    }

    fn compute_avgdl(doc_lengths: &[usize]) -> f64 {
        if doc_lengths.is_empty() {
            return 1.0;
        }
        let total: usize = doc_lengths.iter().sum();
        total as f64 / doc_lengths.len() as f64
    }

    pub fn rank(
        &self,
        query: &str,
        documents: &[&str],
    ) -> Vec<ScoredDocument> {
        if documents.is_empty() {
            return Vec::new();
        }

        let query_tokens = tokenizer().tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let doc_token_freqs: Vec<Vec<(String, u32)>> = documents
            .iter()
            .map(|doc| tokenizer().tokenize_with_frequency(doc))
            .collect();

        let doc_lengths: Vec<usize> = doc_token_freqs
            .iter()
            .map(|freqs| freqs.iter().map(|(_, f)| *f as usize).sum())
            .collect();

        let avgdl = Self::compute_avgdl(&doc_lengths);
        let num_docs = documents.len();

        let mut doc_freq: HashMap<&str, usize> = HashMap::new();
        for freqs in &doc_token_freqs {
            let unique_terms: std::collections::HashSet<&str> =
                freqs.iter().map(|(t, _)| t.as_str()).collect();
            for term in unique_terms {
                *doc_freq.entry(term).or_insert(0) += 1;
            }
        }

        let idf_values: Vec<IdfEntry> = query_tokens
            .iter()
            .map(|qt| {
                let df = doc_freq.get(qt.as_str()).copied().unwrap_or(0);
                IdfEntry {
                    term: qt.clone(),
                    idf: Self::compute_idf(num_docs, df),
                }
            })
            .collect();

        let k1 = self.config.k1;
        let b = self.config.b;

        let mut results: Vec<ScoredDocument> = doc_token_freqs
            .iter()
            .enumerate()
            .map(|(idx, freqs)| {
                let freq_map: HashMap<&str, u32> =
                    freqs.iter().map(|(t, f)| (t.as_str(), *f)).collect();
                let doc_len = doc_lengths[idx] as f64;

                let mut score = 0.0;
                let mut matched = Vec::new();

                for entry in &idf_values {
                    let tf = freq_map.get(entry.term.as_str()).copied().unwrap_or(0) as f64;
                    if tf > 0.0 {
                        let tf_norm = (tf * (k1 + 1.0))
                            / (tf + k1 * (1.0 - b + b * (doc_len / avgdl)));
                        score += entry.idf * tf_norm;
                        matched.push(entry.term.clone());
                    }
                }

                ScoredDocument {
                    doc_index: idx,
                    score,
                    matched_terms: matched,
                }
            })
            .filter(|d| d.score > 0.0)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

fn tokenizer() -> &'static Tokenizer {
    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(Tokenizer::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic_ranking() {
        let scorer = Bm25Scorer::with_defaults();
        let docs = vec![
            "authentication jwt token validation",
            "database connection pool management",
            "user authentication with oauth2 tokens",
        ];
        let results = scorer.rank("authentication token", &docs);
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_index, 0);
    }

    #[test]
    fn test_bm25_no_match() {
        let scorer = Bm25Scorer::with_defaults();
        let docs = vec!["hello world", "foo bar baz"];
        let results = scorer.rank("authentication", &docs);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_camel_case_matching() {
        let scorer = Bm25Scorer::with_defaults();
        let docs = vec![
            "fn getUserById(id: u64) -> User",
            "fn deleteAllRecords() -> Result",
            "fn updateUserProfile(user: User)",
        ];
        let results = scorer.rank("user", &docs);
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_bm25_empty_query() {
        let scorer = Bm25Scorer::with_defaults();
        let docs = vec!["some content"];
        let results = scorer.rank("", &docs);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_empty_docs() {
        let scorer = Bm25Scorer::with_defaults();
        let results = scorer.rank("query", &[]);
        assert!(results.is_empty());
    }
}
