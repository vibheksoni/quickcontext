use std::path::Path;
use std::process::Command;


#[derive(Debug, Clone)]
pub struct LspServerSpec {
    pub name: &'static str,
    pub language_id: &'static str,
    pub binary: &'static str,
    pub args: &'static [&'static str],
    pub extensions: &'static [&'static str],
    pub root_markers: &'static [&'static str],
    pub needs_did_open: bool,
}


pub const SPECS: &[LspServerSpec] = &[
    LspServerSpec {
        name: "rust-analyzer",
        language_id: "rust",
        binary: "rust-analyzer",
        args: &[],
        extensions: &[".rs"],
        root_markers: &["Cargo.toml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "pyright",
        language_id: "python",
        binary: "pyright-langserver",
        args: &["--stdio"],
        extensions: &[".py", ".pyi"],
        root_markers: &["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "typescript-language-server",
        language_id: "typescript",
        binary: "typescript-language-server",
        args: &["--stdio"],
        extensions: &[".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"],
        root_markers: &["tsconfig.json", "jsconfig.json", "package.json"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "gopls",
        language_id: "go",
        binary: "gopls",
        args: &["serve"],
        extensions: &[".go"],
        root_markers: &["go.mod"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "clangd",
        language_id: "c",
        binary: "clangd",
        args: &[],
        extensions: &[".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"],
        root_markers: &["compile_commands.json", "CMakeLists.txt", ".clangd"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "jdtls",
        language_id: "java",
        binary: "jdtls",
        args: &[],
        extensions: &[".java"],
        root_markers: &["pom.xml", "build.gradle", "build.gradle.kts"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "omnisharp",
        language_id: "csharp",
        binary: "OmniSharp",
        args: &["--languageserver"],
        extensions: &[".cs"],
        root_markers: &[".csproj", ".sln"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "lua-language-server",
        language_id: "lua",
        binary: "lua-language-server",
        args: &[],
        extensions: &[".lua"],
        root_markers: &[".luarc.json", ".luarc.jsonc"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "zls",
        language_id: "zig",
        binary: "zls",
        args: &[],
        extensions: &[".zig"],
        root_markers: &["build.zig"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "ruby-lsp",
        language_id: "ruby",
        binary: "ruby-lsp",
        args: &[],
        extensions: &[".rb", ".rake", ".gemspec"],
        root_markers: &["Gemfile", ".ruby-version", "Rakefile"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "phpactor",
        language_id: "php",
        binary: "phpactor",
        args: &["language-server"],
        extensions: &[".php"],
        root_markers: &["composer.json", ".phpactor.json", ".phpactor.yml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "kotlin-language-server",
        language_id: "kotlin",
        binary: "kotlin-language-server",
        args: &[],
        extensions: &[".kt", ".kts"],
        root_markers: &["build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "pom.xml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "sourcekit-lsp",
        language_id: "swift",
        binary: "sourcekit-lsp",
        args: &[],
        extensions: &[".swift"],
        root_markers: &["Package.swift", ".xcodeproj", ".xcworkspace"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "elixir-ls",
        language_id: "elixir",
        binary: "elixir-ls",
        args: &[],
        extensions: &[".ex", ".exs"],
        root_markers: &["mix.exs"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "haskell-language-server",
        language_id: "haskell",
        binary: "haskell-language-server-wrapper",
        args: &["--lsp"],
        extensions: &[".hs", ".lhs"],
        root_markers: &["hie.yaml", "cabal.project", "stack.yaml", ".cabal"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "metals",
        language_id: "scala",
        binary: "metals",
        args: &[],
        extensions: &[".scala", ".sc", ".sbt"],
        root_markers: &["build.sbt", "build.sc", ".scala-build"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "dart-language-server",
        language_id: "dart",
        binary: "dart",
        args: &["language-server", "--protocol=lsp"],
        extensions: &[".dart"],
        root_markers: &["pubspec.yaml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "r-languageserver",
        language_id: "r",
        binary: "R",
        args: &["--slave", "-e", "languageserver::run()"],
        extensions: &[".r", ".R", ".rmd", ".Rmd"],
        root_markers: &["DESCRIPTION", ".Rproj"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "bash-language-server",
        language_id: "bash",
        binary: "bash-language-server",
        args: &["start"],
        extensions: &[".sh", ".bash", ".zsh"],
        root_markers: &[".bashrc", ".bash_profile"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "yaml-language-server",
        language_id: "yaml",
        binary: "yaml-language-server",
        args: &["--stdio"],
        extensions: &[".yaml", ".yml"],
        root_markers: &[".yamllint", ".yamllint.yaml", ".yamllint.yml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "vscode-json-languageserver",
        language_id: "json",
        binary: "vscode-json-language-server",
        args: &["--stdio"],
        extensions: &[".json", ".jsonc"],
        root_markers: &[".json"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "vscode-html-languageserver",
        language_id: "html",
        binary: "vscode-html-language-server",
        args: &["--stdio"],
        extensions: &[".html", ".htm"],
        root_markers: &["index.html"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "vscode-css-languageserver",
        language_id: "css",
        binary: "vscode-css-language-server",
        args: &["--stdio"],
        extensions: &[".css", ".scss", ".less"],
        root_markers: &["package.json"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "terraform-ls",
        language_id: "terraform",
        binary: "terraform-ls",
        args: &["serve"],
        extensions: &[".tf", ".tfvars"],
        root_markers: &[".terraform", "main.tf", "terraform.tfstate"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "dockerfile-language-server",
        language_id: "dockerfile",
        binary: "docker-langserver",
        args: &["--stdio"],
        extensions: &[".dockerfile"],
        root_markers: &["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "taplo",
        language_id: "toml",
        binary: "taplo",
        args: &["lsp", "stdio"],
        extensions: &[".toml"],
        root_markers: &["Cargo.toml", "pyproject.toml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "marksman",
        language_id: "markdown",
        binary: "marksman",
        args: &["server"],
        extensions: &[".md", ".markdown"],
        root_markers: &[".marksman.toml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "svelte-language-server",
        language_id: "svelte",
        binary: "svelteserver",
        args: &["--stdio"],
        extensions: &[".svelte"],
        root_markers: &["svelte.config.js", "svelte.config.ts"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "vue-language-server",
        language_id: "vue",
        binary: "vue-language-server",
        args: &["--stdio"],
        extensions: &[".vue"],
        root_markers: &["vue.config.js", "nuxt.config.js", "nuxt.config.ts"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "ocamllsp",
        language_id: "ocaml",
        binary: "ocamllsp",
        args: &[],
        extensions: &[".ml", ".mli"],
        root_markers: &["dune-project", ".merlin"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "erlang-ls",
        language_id: "erlang",
        binary: "erlang_ls",
        args: &[],
        extensions: &[".erl", ".hrl"],
        root_markers: &["rebar.config", "erlang.mk"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "clojure-lsp",
        language_id: "clojure",
        binary: "clojure-lsp",
        args: &[],
        extensions: &[".clj", ".cljs", ".cljc", ".edn"],
        root_markers: &["deps.edn", "project.clj", "shadow-cljs.edn"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "solargraph",
        language_id: "ruby_solargraph",
        binary: "solargraph",
        args: &["stdio"],
        extensions: &[],
        root_markers: &[".solargraph.yml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "cmake-language-server",
        language_id: "cmake",
        binary: "cmake-language-server",
        args: &[],
        extensions: &[".cmake"],
        root_markers: &["CMakeLists.txt"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "perl-navigator",
        language_id: "perl",
        binary: "perlnavigator",
        args: &["--stdio"],
        extensions: &[".pl", ".pm", ".t"],
        root_markers: &["Makefile.PL", "Build.PL", "cpanfile"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "nim-langserver",
        language_id: "nim",
        binary: "nimlangserver",
        args: &[],
        extensions: &[".nim", ".nims", ".nimble"],
        root_markers: &[".nimble"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "vhdl-ls",
        language_id: "vhdl",
        binary: "vhdl_ls",
        args: &[],
        extensions: &[".vhd", ".vhdl"],
        root_markers: &["vhdl_ls.toml"],
        needs_did_open: true,
    },
    LspServerSpec {
        name: "verible",
        language_id: "verilog",
        binary: "verible-verilog-ls",
        args: &[],
        extensions: &[".v", ".sv", ".svh"],
        root_markers: &[".verible"],
        needs_did_open: true,
    },
];


/// Find the LSP server spec for a given language ID.
///
/// language_id: &str — Language identifier (e.g. "rust", "python", "typescript").
pub fn spec_for_language(language_id: &str) -> Option<&'static LspServerSpec> {
    let lower = language_id.to_lowercase();
    SPECS.iter().find(|s| {
        s.language_id == lower
            || s.name.to_lowercase() == lower
            || match lower.as_str() {
                "javascript" | "js" | "jsx" => s.language_id == "typescript",
                "ts" | "tsx" => s.language_id == "typescript",
                "cpp" | "c++" | "cxx" => s.language_id == "c",
                "cs" | "c#" | "dotnet" => s.language_id == "csharp",
                "py" => s.language_id == "python",
                "rs" => s.language_id == "rust",
                "rb" => s.language_id == "ruby",
                "kt" | "kts" => s.language_id == "kotlin",
                "ex" | "exs" => s.language_id == "elixir",
                "hs" => s.language_id == "haskell",
                "sc" | "sbt" => s.language_id == "scala",
                "sh" | "shell" | "zsh" => s.language_id == "bash",
                "yml" => s.language_id == "yaml",
                "jsonc" => s.language_id == "json",
                "htm" => s.language_id == "html",
                "scss" | "less" | "sass" => s.language_id == "css",
                "tf" | "hcl" => s.language_id == "terraform",
                "docker" => s.language_id == "dockerfile",
                "md" => s.language_id == "markdown",
                "ml" | "mli" => s.language_id == "ocaml",
                "erl" => s.language_id == "erlang",
                "cljs" | "cljc" => s.language_id == "clojure",
                "pl" | "pm" => s.language_id == "perl",
                "sv" | "systemverilog" => s.language_id == "verilog",
                _ => false,
            }
    })
}


/// Detect the language ID from a file extension.
///
/// file_path: &str — Path to the source file.
pub fn detect_language_from_path(file_path: &str) -> Option<&'static str> {
    let path = Path::new(file_path);

    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        let name_lower = name.to_lowercase();
        if name_lower == "dockerfile" || name_lower.starts_with("dockerfile.") {
            return Some("dockerfile");
        }
        if name_lower == "cmakelists.txt" {
            return Some("cmake");
        }
    }

    let ext = path.extension()?.to_str()?;
    let dot_ext = format!(".{ext}");

    SPECS
        .iter()
        .find(|s| s.extensions.iter().any(|e| e.eq_ignore_ascii_case(&dot_ext)))
        .map(|s| s.language_id)
}


/// Check if a binary is available on PATH.
///
/// binary: &str — Binary name to search for.
pub fn resolve_binary(binary: &str) -> Option<String> {
    if binary == "rust-analyzer" {
        if let Ok(output) = Command::new("rustup").args(["which", binary]).output() {
            if output.status.success() {
                let resolved = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !resolved.is_empty() {
                    return Some(resolved);
                }
            }
        }
    }

    let locator = if cfg!(windows) { "where" } else { "which" };
    let output = Command::new(locator).arg(binary).output().ok()?;
    if !output.status.success() {
        return None;
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.to_string())
}

pub fn find_binary(binary: &str) -> bool {
    resolve_binary(binary).is_some()
}


/// Find the project root by walking up from a path looking for root markers.
///
/// start: &Path — Starting directory.
/// spec: &LspServerSpec — Server spec with root markers to search for.
pub fn find_project_root(start: &Path, spec: &LspServerSpec) -> Option<String> {
    let mut dir = if start.is_file() {
        start.parent()?
    } else {
        start
    };

    loop {
        for marker in spec.root_markers {
            if dir.join(marker).exists() {
                return Some(dir.to_string_lossy().to_string());
            }
        }
        dir = dir.parent()?;
    }
}
