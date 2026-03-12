pub mod python;
pub mod javascript;
pub mod typescript;
pub mod rust_lang;
pub mod go;
pub mod java;
pub mod c_lang;
pub mod cpp;
pub mod csharp;
pub mod ruby;
pub mod php;
pub mod bash;
pub mod html;
pub mod css;
pub mod markdown;
pub mod data;
pub mod swift;
pub mod scala;
pub mod lua;
pub mod elixir;
pub mod hcl;
pub mod sql;
pub mod protobuf;
pub mod xml;
pub mod zig;
pub mod haskell;
pub mod ocaml;
pub mod r_lang;

use std::sync::OnceLock;

use tree_sitter::Language;
use crate::types::SymbolKind;


pub struct LanguageSpec {
    pub name: &'static str,
    pub language: Language,
    pub extensions: &'static [&'static str],
    pub query: &'static str,
    pub pattern_kinds: &'static [SymbolKind],
}


pub fn registry() -> &'static [LanguageSpec] {
    static REGISTRY: OnceLock<Vec<LanguageSpec>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        vec![
            python::spec(),
            javascript::spec(),
            typescript::spec(),
            typescript::spec_tsx(),
            rust_lang::spec(),
            go::spec(),
            java::spec(),
            c_lang::spec(),
            cpp::spec(),
            csharp::spec(),
            ruby::spec(),
            php::spec(),
            bash::spec(),
            html::spec(),
            css::spec(),
            markdown::spec(),
            data::spec_json(),
            data::spec_yaml(),
            data::spec_toml(),
            swift::spec(),
            scala::spec(),
            lua::spec(),
            elixir::spec(),
            hcl::spec(),
            sql::spec(),
            protobuf::spec(),
            xml::spec(),
            zig::spec(),
            haskell::spec(),
            ocaml::spec(),
            r_lang::spec(),
        ]
    })
}


pub fn detect_language<'a>(path: &str, specs: &'a [LanguageSpec]) -> Option<&'a LanguageSpec> {
    let ext = path.rsplit('.').next()?;
    let ext_lower = ext.to_ascii_lowercase();
    specs.iter().find(|s| s.extensions.contains(&ext_lower.as_str()))
}
