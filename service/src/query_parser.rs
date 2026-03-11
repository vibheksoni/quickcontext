use crate::tokenizer::Tokenizer;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub enum QueryExpr {
    Term {
        keywords: Vec<String>,
        required: bool,
        excluded: bool,
        exact: bool,
    },
    And(Box<QueryExpr>, Box<QueryExpr>),
    Or(Box<QueryExpr>, Box<QueryExpr>),
}

#[derive(Debug, Clone)]
pub struct SearchFilter {
    pub file_pattern: Option<String>,
    pub ext_filter: Option<String>,
    pub dir_filter: Option<String>,
    pub lang_filter: Option<String>,
    pub symbol_type: Option<String>,
}

impl Default for SearchFilter {
    fn default() -> Self {
        Self {
            file_pattern: None,
            ext_filter: None,
            dir_filter: None,
            lang_filter: None,
            symbol_type: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedQuery {
    pub expr: Option<QueryExpr>,
    pub filter: SearchFilter,
    pub raw: String,
}

#[derive(Debug, Clone)]
enum QueryToken {
    Word(String),
    Quoted(String),
    And,
    Or,
    Plus,
    Minus,
    LParen,
    RParen,
}

pub struct QueryParser;

impl QueryParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, input: &str) -> ParsedQuery {
        let raw = input.to_string();
        let mut filter = SearchFilter::default();

        let cleaned = self.extract_filters(input, &mut filter);
        let tokens = tokenize_query(&cleaned);
        let expr = if tokens.is_empty() {
            None
        } else {
            let mut pos = 0;
            self.parse_or(&tokens, &mut pos)
        };

        ParsedQuery { expr, filter, raw }
    }

    fn extract_filters(&self, input: &str, filter: &mut SearchFilter) -> String {
        let mut remaining = Vec::new();

        let mut i = 0;
        let chars: Vec<char> = input.chars().collect();
        let len = chars.len();

        while i < len {
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }

            if chars[i] == '"' {
                let start = i;
                i += 1;
                while i < len && chars[i] != '"' {
                    i += 1;
                }
                if i < len {
                    i += 1;
                }
                let chunk: String = chars[start..i].iter().collect();
                remaining.push(chunk);
                continue;
            }

            let start = i;
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();

            if let Some(val) = word.strip_prefix("file:") {
                filter.file_pattern = Some(val.to_string());
            } else if let Some(val) = word.strip_prefix("ext:") {
                filter.ext_filter = Some(val.to_string());
            } else if let Some(val) = word.strip_prefix("dir:") {
                filter.dir_filter = Some(val.to_string());
            } else if let Some(val) = word.strip_prefix("lang:") {
                filter.lang_filter = Some(val.to_string());
            } else if let Some(val) = word.strip_prefix("type:") {
                filter.symbol_type = Some(val.to_string());
            } else {
                remaining.push(word);
            }
        }

        remaining.join(" ")
    }

    fn parse_or(&self, tokens: &[QueryToken], pos: &mut usize) -> Option<QueryExpr> {
        let mut left = self.parse_and(tokens, pos)?;

        while *pos < tokens.len() {
            if matches!(tokens.get(*pos), Some(QueryToken::Or)) {
                *pos += 1;
                if let Some(right) = self.parse_and(tokens, pos) {
                    left = QueryExpr::Or(Box::new(left), Box::new(right));
                }
            } else {
                break;
            }
        }

        Some(left)
    }

    fn parse_and(&self, tokens: &[QueryToken], pos: &mut usize) -> Option<QueryExpr> {
        let mut left = self.parse_unary(tokens, pos)?;

        while *pos < tokens.len() {
            if matches!(tokens.get(*pos), Some(QueryToken::And)) {
                *pos += 1;
                if let Some(right) = self.parse_unary(tokens, pos) {
                    left = QueryExpr::And(Box::new(left), Box::new(right));
                }
            } else if matches!(
                tokens.get(*pos),
                Some(QueryToken::Word(_))
                    | Some(QueryToken::Quoted(_))
                    | Some(QueryToken::Plus)
                    | Some(QueryToken::Minus)
                    | Some(QueryToken::LParen)
            ) {
                if let Some(right) = self.parse_unary(tokens, pos) {
                    left = QueryExpr::Or(Box::new(left), Box::new(right));
                }
            } else {
                break;
            }
        }

        Some(left)
    }

    fn parse_unary(&self, tokens: &[QueryToken], pos: &mut usize) -> Option<QueryExpr> {
        if *pos >= tokens.len() {
            return None;
        }

        match &tokens[*pos] {
            QueryToken::Plus => {
                *pos += 1;
                let mut expr = self.parse_primary(tokens, pos)?;
                if let QueryExpr::Term { ref mut required, .. } = expr {
                    *required = true;
                }
                Some(expr)
            }
            QueryToken::Minus => {
                *pos += 1;
                let mut expr = self.parse_primary(tokens, pos)?;
                if let QueryExpr::Term { ref mut excluded, .. } = expr {
                    *excluded = true;
                }
                Some(expr)
            }
            _ => self.parse_primary(tokens, pos),
        }
    }

    fn parse_primary(&self, tokens: &[QueryToken], pos: &mut usize) -> Option<QueryExpr> {
        if *pos >= tokens.len() {
            return None;
        }

        match &tokens[*pos] {
            QueryToken::LParen => {
                *pos += 1;
                let expr = self.parse_or(tokens, pos);
                if matches!(tokens.get(*pos), Some(QueryToken::RParen)) {
                    *pos += 1;
                }
                expr
            }
            QueryToken::Quoted(phrase) => {
                let keywords = phrase
                    .split_whitespace()
                    .map(|s| s.to_lowercase())
                    .collect();
                *pos += 1;
                Some(QueryExpr::Term {
                    keywords,
                    required: false,
                    excluded: false,
                    exact: true,
                })
            }
            QueryToken::Word(word) => {
                let keywords = tokenizer().tokenize(word);
                *pos += 1;
                if keywords.is_empty() {
                    if *pos < tokens.len() {
                        return self.parse_unary(tokens, pos);
                    }
                    return None;
                }
                Some(QueryExpr::Term {
                    keywords,
                    required: false,
                    excluded: false,
                    exact: false,
                })
            }
            _ => {
                *pos += 1;
                None
            }
        }
    }
}

fn tokenizer() -> &'static Tokenizer {
    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(Tokenizer::new)
}

fn tokenize_query(input: &str) -> Vec<QueryToken> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        if ch == '(' {
            tokens.push(QueryToken::LParen);
            i += 1;
            continue;
        }

        if ch == ')' {
            tokens.push(QueryToken::RParen);
            i += 1;
            continue;
        }

        if ch == '+' && (i + 1 < len) && !chars[i + 1].is_whitespace() {
            tokens.push(QueryToken::Plus);
            i += 1;
            continue;
        }

        if ch == '-' && (i + 1 < len) && !chars[i + 1].is_whitespace() {
            tokens.push(QueryToken::Minus);
            i += 1;
            continue;
        }

        if ch == '"' {
            i += 1;
            let start = i;
            while i < len && chars[i] != '"' {
                i += 1;
            }
            let phrase: String = chars[start..i].iter().collect();
            tokens.push(QueryToken::Quoted(phrase));
            if i < len {
                i += 1;
            }
            continue;
        }

        let start = i;
        while i < len && !chars[i].is_whitespace() && chars[i] != '(' && chars[i] != ')' {
            i += 1;
        }
        let word: String = chars[start..i].iter().collect();

        match word.to_uppercase().as_str() {
            "AND" => tokens.push(QueryToken::And),
            "OR" => tokens.push(QueryToken::Or),
            _ => tokens.push(QueryToken::Word(word)),
        }
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_query() {
        let parser = QueryParser::new();
        let parsed = parser.parse("authentication");
        assert!(parsed.expr.is_some());
    }

    #[test]
    fn test_and_query() {
        let parser = QueryParser::new();
        let parsed = parser.parse("auth AND token");
        assert!(matches!(parsed.expr, Some(QueryExpr::And(_, _))));
    }

    #[test]
    fn test_or_query() {
        let parser = QueryParser::new();
        let parsed = parser.parse("auth OR token");
        assert!(matches!(parsed.expr, Some(QueryExpr::Or(_, _))));
    }

    #[test]
    fn test_required_term() {
        let parser = QueryParser::new();
        let parsed = parser.parse("+authentication");
        match parsed.expr {
            Some(QueryExpr::Term { required, .. }) => assert!(required),
            _ => panic!("expected required Term"),
        }
    }

    #[test]
    fn test_excluded_term() {
        let parser = QueryParser::new();
        let parsed = parser.parse("-test");
        match parsed.expr {
            Some(QueryExpr::Term { excluded, .. }) => assert!(excluded),
            _ => panic!("expected excluded Term"),
        }
    }

    #[test]
    fn test_quoted_phrase() {
        let parser = QueryParser::new();
        let parsed = parser.parse("\"exact match\"");
        match parsed.expr {
            Some(QueryExpr::Term { exact, ref keywords, .. }) => {
                assert!(exact);
                assert_eq!(keywords, &vec!["exact", "match"]);
            }
            _ => panic!("expected exact Term"),
        }
    }

    #[test]
    fn test_filter_extraction() {
        let parser = QueryParser::new();
        let parsed = parser.parse("auth file:main.rs lang:rust");
        assert_eq!(parsed.filter.file_pattern, Some("main.rs".to_string()));
        assert_eq!(parsed.filter.lang_filter, Some("rust".to_string()));
        assert!(parsed.expr.is_some());
    }

    #[test]
    fn test_parentheses() {
        let parser = QueryParser::new();
        let parsed = parser.parse("(auth OR token) AND handler");
        assert!(matches!(parsed.expr, Some(QueryExpr::And(_, _))));
    }

    #[test]
    fn test_implicit_or() {
        let parser = QueryParser::new();
        let parsed = parser.parse("authentication handler");
        assert!(matches!(parsed.expr, Some(QueryExpr::Or(_, _))));
    }

    #[test]
    fn test_empty_query() {
        let parser = QueryParser::new();
        let parsed = parser.parse("");
        assert!(parsed.expr.is_none());
    }
}
