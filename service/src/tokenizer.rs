use std::collections::HashSet;

static ENGLISH_STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
    "such", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "will", "with", "from", "has", "have", "had",
    "been", "being", "do", "does", "did", "can", "could", "would",
    "should", "may", "might", "shall", "must", "need", "about", "above",
    "after", "again", "all", "also", "am", "any", "because", "before",
    "between", "both", "each", "few", "get", "got", "her", "here",
    "him", "his", "how", "its", "just", "like", "more", "most", "my",
    "new", "now", "only", "other", "our", "out", "over", "own", "same",
    "she", "so", "some", "still", "than", "them", "too", "under", "up",
    "very", "we", "what", "when", "where", "which", "while", "who",
    "why", "you", "your",
];

static PROGRAMMING_STOP_WORDS: &[&str] = &[
    "func", "function", "fn", "def", "class", "struct", "enum", "trait",
    "impl", "interface", "type", "var", "let", "const", "val", "mut",
    "pub", "public", "private", "protected", "internal", "static",
    "final", "abstract", "virtual", "override", "async", "await",
    "return", "yield", "break", "continue", "match", "switch", "case",
    "default", "try", "catch", "throw", "throws", "finally", "raise",
    "except", "import", "export", "from", "use", "require", "include",
    "module", "package", "namespace", "void", "null", "none", "nil",
    "true", "false", "self", "this", "super", "new", "delete",
    "sizeof", "typeof", "instanceof", "else", "elif", "elsif",
    "unless", "until", "loop", "while", "for", "foreach", "do",
    "end", "begin", "then", "where", "when", "with", "as",
];

static PRESERVED_TERMS: &[&str] = &[
    "oauth", "oauth2", "jwt", "api", "url", "uri", "http", "https",
    "tcp", "udp", "dns", "ssh", "ssl", "tls", "ftp", "smtp", "imap",
    "ipv4", "ipv6", "graphql", "grpc", "rest", "sql", "nosql",
    "json", "xml", "yaml", "toml", "csv", "html", "css",
    "aws", "gcp", "azure", "docker", "k8s", "kubernetes",
    "redis", "kafka", "rabbitmq", "postgresql", "mysql", "mongodb",
    "sqlite", "qdrant", "elasticsearch", "nginx", "apache",
    "git", "svn", "npm", "pip", "cargo", "maven", "gradle",
    "webpack", "vite", "rollup", "esbuild", "babel",
    "react", "vue", "angular", "svelte", "nextjs", "nuxt",
    "fastapi", "flask", "django", "express", "actix", "axum",
    "tokio", "async", "rayon", "serde",
];

pub struct Tokenizer {
    english_stops: HashSet<&'static str>,
    programming_stops: HashSet<&'static str>,
    preserved: HashSet<&'static str>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            english_stops: ENGLISH_STOP_WORDS.iter().copied().collect(),
            programming_stops: PROGRAMMING_STOP_WORDS.iter().copied().collect(),
            preserved: PRESERVED_TERMS.iter().copied().collect(),
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut seen = HashSet::new();

        for word in split_on_boundaries(text) {
            if word.is_empty() {
                continue;
            }

            let lower = word.to_lowercase();

            if self.preserved.contains(lower.as_str()) {
                if seen.insert(lower.clone()) {
                    tokens.push(lower);
                }
                continue;
            }

            let parts = split_camel_case(word);
            for part in parts {
                if part.len() < 2 {
                    continue;
                }
                if self.english_stops.contains(part.as_str()) {
                    continue;
                }
                if self.programming_stops.contains(part.as_str()) {
                    continue;
                }
                if seen.insert(part.clone()) {
                    tokens.push(part);
                }
            }
        }

        tokens
    }

    pub fn tokenize_with_frequency(&self, text: &str) -> Vec<(String, u32)> {
        let mut freq: Vec<(String, u32)> = Vec::new();
        let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for word in split_on_boundaries(text) {
            if word.is_empty() {
                continue;
            }

            let lower = word.to_lowercase();

            if self.preserved.contains(lower.as_str()) {
                if let Some(&idx) = index.get(&lower) {
                    freq[idx].1 += 1;
                } else {
                    index.insert(lower.clone(), freq.len());
                    freq.push((lower, 1));
                }
                continue;
            }

            let parts = split_camel_case(word);
            for part in parts {
                if part.len() < 2 {
                    continue;
                }
                if self.english_stops.contains(part.as_str()) {
                    continue;
                }
                if self.programming_stops.contains(part.as_str()) {
                    continue;
                }
                if let Some(&idx) = index.get(&part) {
                    freq[idx].1 += 1;
                } else {
                    index.insert(part.clone(), freq.len());
                    freq.push((part, 1));
                }
            }
        }

        freq
    }
}

fn split_on_boundaries(text: &str) -> impl Iterator<Item = &str> {
    text.split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
        .filter(|word| !word.is_empty())
}

fn split_camel_case(input: &str) -> Vec<String> {
    if input.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();

    if len <= 2 {
        return vec![input.to_string()];
    }

    let has_upper = chars.iter().any(|c| c.is_uppercase());
    let has_underscore = chars.contains(&'_');

    if has_underscore {
        return input
            .split('_')
            .filter(|s| !s.is_empty())
            .flat_map(split_camel_case)
            .collect();
    }

    if !has_upper && !chars.iter().any(|c| c.is_ascii_digit()) {
        return vec![input.to_string()];
    }

    let mut parts = Vec::new();
    let mut start = 0;

    for i in 1..len {
        let prev = chars[i - 1];
        let curr = chars[i];

        let split = (prev.is_lowercase() && curr.is_uppercase())
            || (prev.is_ascii_digit() && curr.is_alphabetic())
            || (prev.is_alphabetic() && curr.is_ascii_digit())
            || (i + 1 < len && prev.is_uppercase() && curr.is_uppercase() && chars[i + 1].is_lowercase());

        if split {
            let part: String = chars[start..i].iter().collect();
            if !part.is_empty() {
                parts.push(part.to_lowercase());
            }
            start = i;
        }
    }

    let remaining: String = chars[start..].iter().collect();
    if !remaining.is_empty() {
        parts.push(remaining.to_lowercase());
    }

    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camel_case_splitting() {
        assert_eq!(split_camel_case("camelCase"), vec!["camel", "case"]);
        assert_eq!(split_camel_case("PascalCase"), vec!["pascal", "case"]);
        assert_eq!(split_camel_case("APIDefinition"), vec!["api", "definition"]);
        assert_eq!(split_camel_case("getUserById"), vec!["get", "user", "by", "id"]);
        assert_eq!(split_camel_case("parseJSONToHTML"), vec!["parse", "json", "to", "html"]);
        assert_eq!(split_camel_case("oauth2"), vec!["oauth", "2"]);
        assert_eq!(split_camel_case("snake_case_name"), vec!["snake", "case", "name"]);
    }

    #[test]
    fn test_tokenizer_basic() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize("getUserById authentication");
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"authentication".to_string()));
    }

    #[test]
    fn test_tokenizer_stops_filtered() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize("the function is not working");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"not".to_string()));
        assert!(tokens.contains(&"working".to_string()));
    }

    #[test]
    fn test_preserved_terms() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize("OAuth2 JWT GraphQL");
        assert!(tokens.contains(&"oauth2".to_string()));
        assert!(tokens.contains(&"jwt".to_string()));
        assert!(tokens.contains(&"graphql".to_string()));
    }

    #[test]
    fn test_frequency() {
        let tok = Tokenizer::new();
        let freq = tok.tokenize_with_frequency("user user user admin");
        let user_freq = freq.iter().find(|(t, _)| t == "user").unwrap();
        assert_eq!(user_freq.1, 3);
    }
}
