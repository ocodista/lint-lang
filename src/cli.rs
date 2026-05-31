use std::{ffi::OsString, path::PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::prompt::GrammarLocale;

#[derive(Debug)]
pub enum DomainCommand {
    Lint(Cli),
    Models(ModelsCli),
    Doctor,
}

#[derive(Debug, Parser)]
#[command(
    name = "lint-lang",
    version,
    about = "Lint grammar locally for free with Rust and native llama.cpp.",
    after_help = "Commands:\n  lint               Lint grammar explicitly; default when no command is provided\n  models list        List configured, local, Ollama, and downloadable models\n  models download    Download the default or custom GGUF model\n  models select      Select and save a model with the TUI\n  doctor             Check config, models, clipboard, and native llama.cpp support"
)]
pub struct Cli {
    #[command(flatten)]
    pub lint: LintArgs,
}

#[derive(Debug, Args)]
pub struct LintArgs {
    /// Text to grammar-fix. If omitted, stdin is used when piped.
    #[arg(value_name = "TEXT", trailing_var_arg = true)]
    pub text: Vec<String>,

    /// Override with an Ollama model name for this run.
    #[arg(short, long, conflicts_with = "model_path")]
    pub model: Option<String>,

    /// Override with a local model path (.gguf uses native llama.cpp unless --llama-cli is set).
    #[arg(long, value_name = "PATH", conflicts_with = "model")]
    pub model_path: Option<PathBuf>,

    /// Backend for --model-path. Auto infers from extension.
    #[arg(long, value_enum, requires = "model_path")]
    pub backend: Option<LocalBackend>,

    /// Path to llama.cpp's llama-cli binary for .gguf models.
    #[arg(long, value_name = "PATH")]
    pub llama_cli: Option<PathBuf>,

    /// Directory to scan for .llamafile and .gguf models. Can be repeated.
    #[arg(long, value_name = "DIR")]
    pub model_dir: Vec<PathBuf>,

    /// Download the default Qwen3 8B GGUF model, save it, and use it.
    #[arg(long)]
    pub download_model: bool,

    /// Download a custom model URL instead of the default Qwen3 8B GGUF.
    #[arg(long, value_name = "URL")]
    pub download_url: Option<String>,

    /// Download output file or directory. Defaults to lint-lang's app model directory.
    #[arg(long, value_name = "PATH")]
    pub download_output: Option<PathBuf>,

    /// Re-download even if the target model file already exists.
    #[arg(long)]
    pub force_download: bool,

    /// Print the selected local model path and exit when no text is provided.
    #[arg(long)]
    pub print_model_path: bool,

    /// Ollama HTTP endpoint, only used for Ollama fallback models.
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Prompt locale for grammar correction.
    #[arg(long, value_enum)]
    pub locale: Option<GrammarLocale>,

    /// Shortcut for --locale pt-br.
    #[arg(long = "pt-br", alias = "ptbr", conflicts_with = "locale")]
    pub pt_br: bool,

    /// Open the TUI model selector and save the selection.
    #[arg(long)]
    pub configure: bool,

    /// Print the config file path and exit, unless text/configuration was also requested.
    #[arg(long)]
    pub config_path: bool,

    /// Do not write the fixed string to the system clipboard.
    #[arg(long)]
    pub no_clipboard: bool,
}

#[derive(Debug, Parser)]
#[command(name = "lint-lang models", about = "Manage local grammar models")]
pub struct ModelsCli {
    /// Ollama HTTP endpoint, only used for Ollama fallback models.
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Path to llama.cpp's llama-cli binary for .gguf models.
    #[arg(long, value_name = "PATH")]
    pub llama_cli: Option<PathBuf>,

    /// Directory to scan for .llamafile and .gguf models. Can be repeated.
    #[arg(long, value_name = "DIR")]
    pub model_dir: Vec<PathBuf>,

    /// Locale to save with selected or downloaded models.
    #[arg(long, value_enum)]
    pub locale: Option<GrammarLocale>,

    /// Shortcut for --locale pt-br.
    #[arg(long = "pt-br", alias = "ptbr", conflicts_with = "locale")]
    pub pt_br: bool,

    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// List configured, local, Ollama, and downloadable models.
    List(ModelsListArgs),

    /// Download the default or custom GGUF model and save it as selected.
    Download(ModelsDownloadArgs),

    /// Select and save a model with the TUI.
    Select(ModelsSelectArgs),
}

#[derive(Debug, Args)]
pub struct ModelsListArgs {
    /// Print a plain text list instead of opening the TUI list.
    #[arg(long)]
    pub plain: bool,
}

#[derive(Debug, Args)]
pub struct ModelsDownloadArgs {
    /// Download a custom model URL instead of the default Qwen3 8B GGUF.
    #[arg(long, value_name = "URL")]
    pub url: Option<String>,

    /// Download output file or directory. Defaults to lint-lang's app model directory.
    #[arg(long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Re-download even if the target model file already exists.
    #[arg(long)]
    pub force: bool,

    /// Print the selected local model path after download.
    #[arg(long)]
    pub print_path: bool,
}

#[derive(Debug, Args)]
pub struct ModelsSelectArgs {
    /// Select this local model path directly instead of opening the TUI.
    #[arg(long, value_name = "PATH")]
    pub model_path: Option<PathBuf>,

    /// Backend for --model-path. Auto infers from extension.
    #[arg(long, value_enum, requires = "model_path")]
    pub backend: Option<LocalBackend>,
}

#[derive(Debug, Parser)]
#[command(name = "lint-lang doctor", about = "Check lint-lang setup")]
struct DoctorCli {}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LocalBackend {
    Auto,
    Llamafile,
    LlamaCpp,
}

impl LocalBackend {
    pub fn config_key(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::Llamafile => Some("llamafile"),
            Self::LlamaCpp => Some("llama-cpp"),
        }
    }
}

pub fn parse_domain_command() -> DomainCommand {
    parse_domain_command_from(std::env::args_os().collect()).unwrap_or_else(|error| error.exit())
}

fn parse_domain_command_from(mut args: Vec<OsString>) -> Result<DomainCommand, clap::Error> {
    if args.is_empty() {
        args.push(OsString::from("lint-lang"));
    }

    match args.get(1).and_then(|arg| arg.to_str()) {
        Some("doctor") => parse_doctor_command(args),
        Some("lint") => parse_lint_command(args),
        Some("models") => parse_models_command(args),
        _ => Cli::try_parse_from(args).map(DomainCommand::Lint),
    }
}

fn parse_doctor_command(mut args: Vec<OsString>) -> Result<DomainCommand, clap::Error> {
    args.remove(1);
    args[0] = OsString::from("lint-lang doctor");
    DoctorCli::try_parse_from(args).map(|_| DomainCommand::Doctor)
}

fn parse_lint_command(mut args: Vec<OsString>) -> Result<DomainCommand, clap::Error> {
    args.remove(1);
    args[0] = OsString::from("lint-lang lint");
    Cli::try_parse_from(args).map(DomainCommand::Lint)
}

fn parse_models_command(mut args: Vec<OsString>) -> Result<DomainCommand, clap::Error> {
    args.remove(1);
    args[0] = OsString::from("lint-lang models");
    ModelsCli::try_parse_from(args).map(DomainCommand::Models)
}

#[cfg(test)]
mod tests {
    use super::{DomainCommand, ModelsCommand, parse_domain_command_from};

    #[test]
    fn parses_default_lint_command() {
        let command = parse_domain_command_from(args(["lint-lang", "hello"])).unwrap();

        let DomainCommand::Lint(cli) = command else {
            panic!("expected lint command");
        };
        assert_eq!(cli.lint.text, ["hello"]);
    }

    #[test]
    fn parses_explicit_lint_command() {
        let command = parse_domain_command_from(args(["lint-lang", "lint", "hello"])).unwrap();

        let DomainCommand::Lint(cli) = command else {
            panic!("expected lint command");
        };
        assert_eq!(cli.lint.text, ["hello"]);
    }

    #[test]
    fn parses_models_list_command() {
        let command =
            parse_domain_command_from(args(["lint-lang", "models", "list", "--plain"])).unwrap();

        let DomainCommand::Models(cli) = command else {
            panic!("expected models command");
        };
        assert!(matches!(cli.command, ModelsCommand::List(list) if list.plain));
    }

    #[test]
    fn parses_doctor_command() {
        let command = parse_domain_command_from(args(["lint-lang", "doctor"])).unwrap();

        assert!(matches!(command, DomainCommand::Doctor));
    }

    fn args<const N: usize>(values: [&str; N]) -> Vec<std::ffi::OsString> {
        values.into_iter().map(std::ffi::OsString::from).collect()
    }
}
