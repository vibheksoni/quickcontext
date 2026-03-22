from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os
import shutil
import subprocess


PlatformName = Literal["windows", "linux"]

_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    "__pycache__",
    ".venv",
    "venv",
    ".quickcontext",
    "_ignore",
    "target",
}


@dataclass(frozen=True, slots=True)
class LspInstallStep:
    manager: str
    command: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class LspServerSetup:
    name: str
    language_id: str
    binary: str
    installed: bool
    detection_reasons: list[str]
    auto_install_supported: bool
    install_steps: list[LspInstallStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LspSetupPlan:
    target_path: str
    platform: PlatformName
    servers: list[LspServerSetup]


@dataclass(frozen=True, slots=True)
class LspInstallResult:
    server_name: str
    command: str
    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class LspServerCheck:
    name: str
    language_id: str
    binary: str
    status: str
    installed: bool
    auto_install_supported: bool
    detection_reasons: list[str]
    install_steps: list[LspInstallStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    probe_command: str | None = None
    probe_message: str | None = None


@dataclass(frozen=True, slots=True)
class LspCheckPlan:
    target_path: str
    platform: PlatformName
    servers: list[LspServerCheck]


@dataclass(frozen=True, slots=True)
class _ServerSpec:
    name: str
    language_id: str
    binary: str
    extensions: tuple[str, ...] = ()
    filenames: tuple[str, ...] = ()
    root_markers: tuple[str, ...] = ()
    windows_steps: tuple[LspInstallStep, ...] = ()
    linux_steps: tuple[LspInstallStep, ...] = ()
    notes: tuple[str, ...] = ()

    def install_steps_for(self, platform: PlatformName) -> list[LspInstallStep]:
        if platform == "windows":
            return list(self.windows_steps or self.linux_steps)
        return list(self.linux_steps or self.windows_steps)


def _npm_step(packages: str, note: str | None = None) -> LspInstallStep:
    return LspInstallStep(manager="npm", command=f"npm install -g {packages}", note=note)


def _cargo_step(command: str, note: str | None = None) -> LspInstallStep:
    return LspInstallStep(manager="cargo", command=command, note=note)


def _go_step(module: str, note: str | None = None) -> LspInstallStep:
    return LspInstallStep(manager="go", command=f"go install {module}", note=note)


def _pip_step(package: str, note: str | None = None) -> LspInstallStep:
    return LspInstallStep(manager="pip", command=f"python -m pip install {package}", note=note)


def _gem_step(package: str, note: str | None = None) -> LspInstallStep:
    return LspInstallStep(manager="gem", command=f"gem install {package}", note=note)


def _r_step(expression: str, note: str | None = None) -> LspInstallStep:
    return LspInstallStep(manager="R", command=f'R --slave -e "{expression}"', note=note)


def _winget_step(package_id: str, note: str | None = None) -> LspInstallStep:
    return LspInstallStep(manager="winget", command=f"winget install --exact --id {package_id}", note=note)


_SPECS: tuple[_ServerSpec, ...] = (
    _ServerSpec(
        name="rust-analyzer",
        language_id="rust",
        binary="rust-analyzer",
        extensions=(".rs",),
        root_markers=("Cargo.toml",),
        windows_steps=(
            LspInstallStep(manager="rustup", command="rustup component add rust-analyzer rust-src"),
        ),
        linux_steps=(
            LspInstallStep(manager="rustup", command="rustup component add rust-analyzer rust-src"),
        ),
    ),
    _ServerSpec(
        name="pyright",
        language_id="python",
        binary="pyright-langserver",
        extensions=(".py", ".pyi"),
        root_markers=("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"),
        windows_steps=(_npm_step("pyright"),),
        linux_steps=(_npm_step("pyright"),),
        notes=("Installs the pyright package, which provides the pyright-langserver binary.",),
    ),
    _ServerSpec(
        name="typescript-language-server",
        language_id="typescript",
        binary="typescript-language-server",
        extensions=(".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"),
        root_markers=("tsconfig.json", "jsconfig.json", "package.json"),
        windows_steps=(_npm_step("typescript-language-server typescript"),),
        linux_steps=(_npm_step("typescript-language-server typescript"),),
    ),
    _ServerSpec(
        name="gopls",
        language_id="go",
        binary="gopls",
        extensions=(".go",),
        root_markers=("go.mod",),
        windows_steps=(_go_step("golang.org/x/tools/gopls@latest"),),
        linux_steps=(_go_step("golang.org/x/tools/gopls@latest"),),
    ),
    _ServerSpec(
        name="clangd",
        language_id="c",
        binary="clangd",
        extensions=(".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"),
        root_markers=("compile_commands.json", "CMakeLists.txt", ".clangd"),
        windows_steps=(_winget_step("LLVM.LLVM", note="clangd ships with LLVM on Windows."),),
        linux_steps=(LspInstallStep(manager="system", command="Install clangd with your distro package manager."),),
    ),
    _ServerSpec(
        name="jdtls",
        language_id="java",
        binary="jdtls",
        extensions=(".java",),
        root_markers=("pom.xml", "build.gradle", "build.gradle.kts"),
        notes=("Manual install recommended. Eclipse JDT Language Server needs a local JDK and project-specific launcher setup.",),
    ),
    _ServerSpec(
        name="omnisharp",
        language_id="csharp",
        binary="OmniSharp",
        extensions=(".cs",),
        root_markers=(".csproj", ".sln"),
        notes=("Manual install recommended. OmniSharp packaging varies by platform and editor workflow.",),
    ),
    _ServerSpec(
        name="lua-language-server",
        language_id="lua",
        binary="lua-language-server",
        extensions=(".lua",),
        root_markers=(".luarc.json", ".luarc.jsonc"),
        windows_steps=(_winget_step("LuaLS.luals"),),
        linux_steps=(LspInstallStep(manager="manual", command="Install lua-language-server from LuaLS releases or your package manager."),),
    ),
    _ServerSpec(
        name="zls",
        language_id="zig",
        binary="zls",
        extensions=(".zig",),
        root_markers=("build.zig",),
        notes=("Manual install recommended. Install zls from the Zig package ecosystem or official releases.",),
    ),
    _ServerSpec(
        name="ruby-lsp",
        language_id="ruby",
        binary="ruby-lsp",
        extensions=(".rb", ".rake", ".gemspec"),
        root_markers=("Gemfile", ".ruby-version", "Rakefile"),
        windows_steps=(_gem_step("ruby-lsp"),),
        linux_steps=(_gem_step("ruby-lsp"),),
    ),
    _ServerSpec(
        name="phpactor",
        language_id="php",
        binary="phpactor",
        extensions=(".php",),
        root_markers=("composer.json", ".phpactor.json", ".phpactor.yml"),
        notes=("Manual install recommended. Install phpactor with Composer or the official PHAR release.",),
    ),
    _ServerSpec(
        name="kotlin-language-server",
        language_id="kotlin",
        binary="kotlin-language-server",
        extensions=(".kt", ".kts"),
        root_markers=("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "pom.xml"),
        notes=("Manual install recommended. Kotlin language server packaging is platform-specific.",),
    ),
    _ServerSpec(
        name="sourcekit-lsp",
        language_id="swift",
        binary="sourcekit-lsp",
        extensions=(".swift",),
        root_markers=("Package.swift", ".xcodeproj", ".xcworkspace"),
        notes=("Manual install recommended. sourcekit-lsp typically ships with the Swift toolchain.",),
    ),
    _ServerSpec(
        name="elixir-ls",
        language_id="elixir",
        binary="elixir-ls",
        extensions=(".ex", ".exs"),
        root_markers=("mix.exs",),
        notes=("Manual install recommended. Install ElixirLS from its official release package.",),
    ),
    _ServerSpec(
        name="haskell-language-server",
        language_id="haskell",
        binary="haskell-language-server-wrapper",
        extensions=(".hs", ".lhs"),
        root_markers=("hie.yaml", "cabal.project", "stack.yaml"),
        notes=("Manual install recommended. ghcup is the usual installer for Haskell Language Server.",),
    ),
    _ServerSpec(
        name="metals",
        language_id="scala",
        binary="metals",
        extensions=(".scala", ".sc", ".sbt"),
        root_markers=("build.sbt", "build.sc", ".scala-build"),
        windows_steps=(LspInstallStep(manager="coursier", command="cs install metals"),),
        linux_steps=(LspInstallStep(manager="coursier", command="cs install metals"),),
    ),
    _ServerSpec(
        name="dart-language-server",
        language_id="dart",
        binary="dart",
        extensions=(".dart",),
        root_markers=("pubspec.yaml",),
        notes=("The Dart language server ships with the Dart SDK. Install Dart or Flutter and ensure `dart` is on PATH.",),
    ),
    _ServerSpec(
        name="r-languageserver",
        language_id="r",
        binary="R",
        extensions=(".r", ".R", ".rmd", ".Rmd"),
        root_markers=("DESCRIPTION", ".Rproj"),
        windows_steps=(_r_step("install.packages('languageserver', repos='https://cloud.r-project.org')"),),
        linux_steps=(_r_step("install.packages('languageserver', repos='https://cloud.r-project.org')"),),
    ),
    _ServerSpec(
        name="bash-language-server",
        language_id="bash",
        binary="bash-language-server",
        extensions=(".sh", ".bash", ".zsh"),
        filenames=(".bashrc", ".bash_profile"),
        root_markers=(".bashrc", ".bash_profile"),
        windows_steps=(_npm_step("bash-language-server"),),
        linux_steps=(_npm_step("bash-language-server"),),
    ),
    _ServerSpec(
        name="yaml-language-server",
        language_id="yaml",
        binary="yaml-language-server",
        extensions=(".yaml", ".yml"),
        root_markers=(".yamllint", ".yamllint.yaml", ".yamllint.yml"),
        windows_steps=(_npm_step("yaml-language-server"),),
        linux_steps=(_npm_step("yaml-language-server"),),
    ),
    _ServerSpec(
        name="vscode-json-languageserver",
        language_id="json",
        binary="vscode-json-language-server",
        extensions=(".json", ".jsonc"),
        windows_steps=(_npm_step("vscode-langservers-extracted"),),
        linux_steps=(_npm_step("vscode-langservers-extracted"),),
    ),
    _ServerSpec(
        name="vscode-html-languageserver",
        language_id="html",
        binary="vscode-html-language-server",
        extensions=(".html", ".htm"),
        filenames=("index.html",),
        windows_steps=(_npm_step("vscode-langservers-extracted"),),
        linux_steps=(_npm_step("vscode-langservers-extracted"),),
    ),
    _ServerSpec(
        name="vscode-css-languageserver",
        language_id="css",
        binary="vscode-css-language-server",
        extensions=(".css", ".scss", ".less"),
        windows_steps=(_npm_step("vscode-langservers-extracted"),),
        linux_steps=(_npm_step("vscode-langservers-extracted"),),
    ),
    _ServerSpec(
        name="terraform-ls",
        language_id="terraform",
        binary="terraform-ls",
        extensions=(".tf", ".tfvars"),
        root_markers=(".terraform", "main.tf", "terraform.tfstate"),
        notes=("Manual install recommended. Install terraform-ls from the official HashiCorp releases or package manager.",),
    ),
    _ServerSpec(
        name="dockerfile-language-server",
        language_id="dockerfile",
        binary="docker-langserver",
        filenames=("Dockerfile",),
        extensions=(".dockerfile",),
        root_markers=("Dockerfile", "docker-compose.yml", "docker-compose.yaml"),
        windows_steps=(_npm_step("dockerfile-language-server-nodejs"),),
        linux_steps=(_npm_step("dockerfile-language-server-nodejs"),),
    ),
    _ServerSpec(
        name="taplo",
        language_id="toml",
        binary="taplo",
        extensions=(".toml",),
        root_markers=("Cargo.toml", "pyproject.toml"),
        windows_steps=(_cargo_step("cargo install taplo-cli --locked --features lsp"),),
        linux_steps=(_cargo_step("cargo install taplo-cli --locked --features lsp"),),
    ),
    _ServerSpec(
        name="marksman",
        language_id="markdown",
        binary="marksman",
        extensions=(".md", ".markdown"),
        root_markers=(".marksman.toml",),
        notes=("Manual install recommended. Install marksman from its release binaries or package manager.",),
    ),
    _ServerSpec(
        name="svelte-language-server",
        language_id="svelte",
        binary="svelteserver",
        extensions=(".svelte",),
        root_markers=("svelte.config.js", "svelte.config.ts"),
        windows_steps=(_npm_step("svelte-language-server"),),
        linux_steps=(_npm_step("svelte-language-server"),),
    ),
    _ServerSpec(
        name="vue-language-server",
        language_id="vue",
        binary="vue-language-server",
        extensions=(".vue",),
        root_markers=("vue.config.js", "nuxt.config.js", "nuxt.config.ts"),
        windows_steps=(_npm_step("@vue/language-server"),),
        linux_steps=(_npm_step("@vue/language-server"),),
    ),
    _ServerSpec(
        name="ocamllsp",
        language_id="ocaml",
        binary="ocamllsp",
        extensions=(".ml", ".mli"),
        root_markers=("dune-project", ".merlin"),
        windows_steps=(LspInstallStep(manager="opam", command="opam install ocaml-lsp-server"),),
        linux_steps=(LspInstallStep(manager="opam", command="opam install ocaml-lsp-server"),),
    ),
    _ServerSpec(
        name="erlang-ls",
        language_id="erlang",
        binary="erlang_ls",
        extensions=(".erl", ".hrl"),
        root_markers=("rebar.config", "erlang.mk"),
        notes=("Manual install recommended. Install erlang_ls from its release artifacts.",),
    ),
    _ServerSpec(
        name="clojure-lsp",
        language_id="clojure",
        binary="clojure-lsp",
        extensions=(".clj", ".cljs", ".cljc", ".edn"),
        root_markers=("deps.edn", "project.clj", "shadow-cljs.edn"),
        notes=("Manual install recommended. Install clojure-lsp from the official installer or package manager.",),
    ),
    _ServerSpec(
        name="solargraph",
        language_id="ruby_solargraph",
        binary="solargraph",
        root_markers=(".solargraph.yml",),
        windows_steps=(_gem_step("solargraph"),),
        linux_steps=(_gem_step("solargraph"),),
    ),
    _ServerSpec(
        name="cmake-language-server",
        language_id="cmake",
        binary="cmake-language-server",
        extensions=(".cmake",),
        filenames=("CMakeLists.txt",),
        root_markers=("CMakeLists.txt",),
        windows_steps=(_pip_step("cmake-language-server"),),
        linux_steps=(_pip_step("cmake-language-server"),),
    ),
    _ServerSpec(
        name="perl-navigator",
        language_id="perl",
        binary="perlnavigator",
        extensions=(".pl", ".pm", ".t"),
        root_markers=("Makefile.PL", "Build.PL", "cpanfile"),
        notes=("Manual install recommended. perlnavigator packaging varies by ecosystem.",),
    ),
    _ServerSpec(
        name="nim-langserver",
        language_id="nim",
        binary="nimlangserver",
        extensions=(".nim", ".nims", ".nimble"),
        root_markers=(".nimble",),
        notes=("Manual install recommended. Install nimlangserver from the Nim package ecosystem.",),
    ),
    _ServerSpec(
        name="vhdl-ls",
        language_id="vhdl",
        binary="vhdl_ls",
        extensions=(".vhd", ".vhdl"),
        root_markers=("vhdl_ls.toml",),
        notes=("Manual install recommended. Install vhdl_ls from its package manager or release artifacts.",),
    ),
    _ServerSpec(
        name="verible",
        language_id="verilog",
        binary="verible-verilog-ls",
        extensions=(".v", ".sv", ".svh"),
        root_markers=(".verible",),
        notes=("Manual install recommended. Install verible-verilog-ls from Verible releases.",),
    ),
)

_PROBE_COMMANDS: dict[str, tuple[str, ...]] = {
    "rust-analyzer": ("rust-analyzer", "--version"),
    "typescript-language-server": ("typescript-language-server", "--version"),
    "gopls": ("gopls", "version"),
    "clangd": ("clangd", "--version"),
    "lua-language-server": ("lua-language-server", "--version"),
    "zls": ("zls", "--version"),
    "ruby-lsp": ("ruby-lsp", "--version"),
    "phpactor": ("phpactor", "--version"),
    "kotlin-language-server": ("kotlin-language-server", "--version"),
    "sourcekit-lsp": ("sourcekit-lsp", "--version"),
    "haskell-language-server-wrapper": ("haskell-language-server-wrapper", "--version"),
    "metals": ("metals", "--version"),
    "dart": ("dart", "--version"),
    "R": ("R", "--version"),
    "bash-language-server": ("bash-language-server", "--version"),
    "yaml-language-server": ("yaml-language-server", "--version"),
    "vscode-json-language-server": ("vscode-json-language-server", "--version"),
    "vscode-html-language-server": ("vscode-html-language-server", "--version"),
    "vscode-css-language-server": ("vscode-css-language-server", "--version"),
    "terraform-ls": ("terraform-ls", "--version"),
    "taplo": ("taplo", "--version"),
    "marksman": ("marksman", "--version"),
    "vue-language-server": ("vue-language-server", "--version"),
    "ocamllsp": ("ocamllsp", "--version"),
    "erlang_ls": ("erlang_ls", "--version"),
    "clojure-lsp": ("clojure-lsp", "--version"),
    "solargraph": ("solargraph", "--version"),
    "cmake-language-server": ("cmake-language-server", "--version"),
    "perlnavigator": ("perlnavigator", "--version"),
    "nimlangserver": ("nimlangserver", "--version"),
    "vhdl_ls": ("vhdl_ls", "--version"),
    "verible-verilog-ls": ("verible-verilog-ls", "--version"),
}


def current_platform() -> PlatformName:
    return "windows" if os.name == "nt" else "linux"


def _iter_project_files(root: Path):
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _IGNORED_DIRS and not name.startswith(".")]
        current = Path(current_root)
        for filename in filenames:
            yield current / filename


def _detect_servers(path: str | Path) -> tuple[Path, list[LspServerSetup]]:
    target = Path(path).expanduser().resolve()
    root = target.parent if target.is_file() else target
    detected: dict[str, dict[str, set[str]]] = {}

    root_names = {child.name for child in root.iterdir()} if root.exists() and root.is_dir() else set()

    for spec in _SPECS:
        reasons = detected.setdefault(spec.name, {"markers": set(), "files": set()})
        for marker in spec.root_markers:
            if marker in root_names:
                reasons["markers"].add(marker)

    for file_path in _iter_project_files(root):
        suffix = file_path.suffix.lower()
        filename = file_path.name
        relative = str(file_path.relative_to(root)).replace("\\", "/")
        for spec in _SPECS:
            if suffix and suffix in spec.extensions:
                detected[spec.name]["files"].add(relative)
            elif filename in spec.filenames:
                detected[spec.name]["files"].add(relative)

    servers: list[LspServerSetup] = []
    platform = current_platform()
    for spec in _SPECS:
        reasons = detected.get(spec.name)
        if reasons is None:
            continue
        marker_reasons = sorted(reasons["markers"])
        file_reasons = sorted(reasons["files"])
        if not marker_reasons and not file_reasons:
            continue

        install_steps = spec.install_steps_for(platform)
        detection_reasons = [f"root marker: {item}" for item in marker_reasons]
        detection_reasons.extend(f"file: {item}" for item in file_reasons[:8])
        if len(file_reasons) > 8:
            detection_reasons.append(f"and {len(file_reasons) - 8} more matching files")

        servers.append(
            LspServerSetup(
                name=spec.name,
                language_id=spec.language_id,
                binary=spec.binary,
                installed=shutil.which(spec.binary) is not None,
                detection_reasons=detection_reasons,
                auto_install_supported=bool(install_steps),
                install_steps=install_steps,
                notes=list(spec.notes),
            )
        )

    servers.sort(key=lambda item: (item.installed, item.language_id, item.name))
    return root, servers


def build_lsp_setup_plan(path: str | Path) -> LspSetupPlan:
    root, servers = _detect_servers(path)
    return LspSetupPlan(
        target_path=str(root),
        platform=current_platform(),
        servers=servers,
    )


def _run_probe(plan: LspSetupPlan, command: tuple[str, ...]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=plan.target_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return False, "Binary not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "Probe timed out"
    except Exception as exc:
        return False, str(exc)

    output = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode == 0:
        return True, output.splitlines()[0] if output else "Probe succeeded"
    return False, output.splitlines()[0] if output else f"Probe failed with exit code {completed.returncode}"


def build_lsp_check_plan(path: str | Path) -> LspCheckPlan:
    plan = build_lsp_setup_plan(path)
    checks: list[LspServerCheck] = []

    for server in plan.servers:
        probe = _PROBE_COMMANDS.get(server.binary) or _PROBE_COMMANDS.get(server.name)
        if not server.installed:
            checks.append(
                LspServerCheck(
                    name=server.name,
                    language_id=server.language_id,
                    binary=server.binary,
                    status="missing",
                    installed=False,
                    auto_install_supported=server.auto_install_supported,
                    detection_reasons=list(server.detection_reasons),
                    install_steps=list(server.install_steps),
                    notes=list(server.notes),
                    probe_command=" ".join(probe) if probe else None,
                    probe_message="Binary is not on PATH",
                )
            )
            continue

        if probe is None:
            checks.append(
                LspServerCheck(
                    name=server.name,
                    language_id=server.language_id,
                    binary=server.binary,
                    status="installed",
                    installed=True,
                    auto_install_supported=server.auto_install_supported,
                    detection_reasons=list(server.detection_reasons),
                    install_steps=list(server.install_steps),
                    notes=list(server.notes),
                    probe_message="Binary is present, but no safe probe command is configured.",
                )
            )
            continue

        success, message = _run_probe(plan, probe)
        checks.append(
            LspServerCheck(
                name=server.name,
                language_id=server.language_id,
                binary=server.binary,
                status="ready" if success else "error",
                installed=True,
                auto_install_supported=server.auto_install_supported,
                detection_reasons=list(server.detection_reasons),
                install_steps=list(server.install_steps),
                notes=list(server.notes),
                probe_command=" ".join(probe),
                probe_message=message,
            )
        )

    return LspCheckPlan(
        target_path=plan.target_path,
        platform=plan.platform,
        servers=checks,
    )


def install_lsp_servers(
    plan: LspSetupPlan,
    server_names: list[str] | None = None,
) -> list[LspInstallResult]:
    selected = {name.lower() for name in server_names} if server_names else None
    seen_commands: set[str] = set()
    results: list[LspInstallResult] = []

    for server in plan.servers:
        if selected and server.name.lower() not in selected and server.language_id.lower() not in selected:
            continue
        if server.installed:
            continue
        if not server.auto_install_supported:
            results.append(
                LspInstallResult(
                    server_name=server.name,
                    command="",
                    success=False,
                    message="Automatic install is not supported for this server",
                )
            )
            continue

        for step in server.install_steps:
            key = f"{step.manager}|{step.command}"
            if key in seen_commands:
                continue
            seen_commands.add(key)
            if shutil.which(step.manager) is None:
                results.append(
                    LspInstallResult(
                        server_name=server.name,
                        command=step.command,
                        success=False,
                        message=f"Required tool '{step.manager}' is not on PATH",
                    )
                )
                continue

            if plan.platform == "windows":
                completed = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", step.command],
                    check=False,
                )
            else:
                completed = subprocess.run(step.command, shell=True, check=False)
            results.append(
                LspInstallResult(
                    server_name=server.name,
                    command=step.command,
                    success=completed.returncode == 0,
                    message="Installed" if completed.returncode == 0 else f"Install command failed with exit code {completed.returncode}",
                )
            )

    return results
