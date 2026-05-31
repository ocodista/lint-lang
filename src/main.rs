mod bundled;
mod cli;
mod config;
mod discovery;
mod download;
mod inference;
mod ollama;
mod prompt;
mod tui;

use std::{
    io::{self, IsTerminal, Read},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use arboard::Clipboard;
use clap::Parser;
use config::{
    AppConfig, ConfiguredModel, DEFAULT_ENDPOINT, ModelBackend, config_path,
    configured_model_from_path, expand_home,
};
use prompt::GrammarLocale;

use crate::cli::{Cli, LocalBackend};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.config_path {
        println!("{}", config_path()?.display());
        if !cli.configure && cli.text.is_empty() {
            return Ok(());
        }
    }

    let mut config = AppConfig::load()?;
    let endpoint = cli
        .endpoint
        .clone()
        .or_else(|| config.endpoint.clone())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned());
    let llama_cli = cli.llama_cli.clone().or_else(|| config.llama_cli.clone());
    let model_dirs = model_dirs(&cli, &config);
    let locale = selected_locale(&cli, &config);
    let bundled_model = bundled::configured_model(llama_cli.clone())?;
    let downloaded_model = if cli.download_model || cli.download_url.is_some() {
        let downloaded_model = download_selected_model(&cli, llama_cli.clone()).await?;
        save_config(
            &mut config,
            downloaded_model.clone(),
            &endpoint,
            llama_cli.clone(),
            &model_dirs,
            locale,
        )?;

        if cli.print_model_path
            && let Some(path) = local_model_path(&downloaded_model)
        {
            println!("{}", path.display());
        }

        if cli.text.is_empty() && !cli.configure && io::stdin().is_terminal() {
            return Ok(());
        }

        Some(downloaded_model)
    } else {
        None
    };

    let cli_model = cli_model_override(&cli, &endpoint, llama_cli.clone())?;
    let mut config_model = preferred_config_model(&config, bundled_model.clone(), &endpoint).await;

    if cli.configure {
        let selected_model = match cli_model.clone() {
            Some(model) => model,
            None => {
                configure_model(
                    &endpoint,
                    config_model.as_ref(),
                    &model_dirs,
                    llama_cli.clone(),
                    bundled_model.clone(),
                )
                .await?
            }
        };

        save_config(
            &mut config,
            selected_model.clone(),
            &endpoint,
            llama_cli.clone(),
            &model_dirs,
            locale,
        )?;

        if cli.text.is_empty() && io::stdin().is_terminal() {
            return Ok(());
        }
    }

    if cli_model.is_none()
        && downloaded_model.is_none()
        && config_model.is_none()
        && tui::terminal_available()
    {
        let selected_model = configure_model(
            &endpoint,
            None,
            &model_dirs,
            llama_cli.clone(),
            bundled_model.clone(),
        )
        .await?;
        save_config(
            &mut config,
            selected_model.clone(),
            &endpoint,
            llama_cli.clone(),
            &model_dirs,
            locale,
        )?;
        config_model = Some(selected_model);

        if cli.text.is_empty() && io::stdin().is_terminal() {
            return Ok(());
        }
    }

    if cli.print_model_path && cli.text.is_empty() && io::stdin().is_terminal() {
        if let Some(model) = cli_model
            .clone()
            .or(downloaded_model.clone())
            .or(config_model.clone())
            .or(bundled_model.clone())
            && let Some(path) = local_model_path(&model)
        {
            println!("{}", path.display());
            return Ok(());
        }

        let downloaded_model = download_selected_model(&cli, llama_cli.clone()).await?;
        save_config(
            &mut config,
            downloaded_model.clone(),
            &endpoint,
            llama_cli.clone(),
            &model_dirs,
            locale,
        )?;
        if let Some(path) = local_model_path(&downloaded_model) {
            println!("{}", path.display());
        }
        return Ok(());
    }

    let input = read_input(&cli.text)?;
    if input.trim().is_empty() {
        bail!("provide a non-empty string to fix");
    }

    let selected_model = match cli_model.or(downloaded_model).or(config_model) {
        Some(model) => model,
        None => {
            let selected_model = configure_model(
                &endpoint,
                None,
                &model_dirs,
                llama_cli.clone(),
                bundled_model,
            )
            .await?;
            save_config(
                &mut config,
                selected_model.clone(),
                &endpoint,
                llama_cli.clone(),
                &model_dirs,
                locale,
            )?;
            selected_model
        }
    };
    let selected_model = apply_runtime_config(selected_model, &endpoint, llama_cli);

    let model_label = selected_model.name.clone();
    let task_model = selected_model.clone();
    let task_input = input.clone();
    let fix_task =
        tokio::spawn(async move { inference::fix_grammar(&task_model, &task_input, locale).await });

    let fixed = tui::wait_with_loading(fix_task, &model_label, locale).await?;

    if !cli.no_clipboard
        && let Err(error) = copy_to_clipboard(&fixed)
    {
        eprintln!("warning: could not copy to clipboard: {error:#}");
    }

    println!("{fixed}");
    Ok(())
}

async fn configure_model(
    endpoint: &str,
    current_model: Option<&ConfiguredModel>,
    model_dirs: &[PathBuf],
    llama_cli: Option<PathBuf>,
    bundled_model: Option<ConfiguredModel>,
) -> Result<ConfiguredModel> {
    let mut candidates = Vec::new();
    if let Some(model) = bundled_model {
        candidates.push(model);
    }
    candidates.extend(
        discovery::discover_local_models(model_dirs, llama_cli.clone())
            .into_iter()
            .filter(|model| !is_deprecated_default_model(model)),
    );

    match ollama::list_models(endpoint).await {
        Ok(models) => candidates.extend(models.into_iter().map(|model| ConfiguredModel {
            name: model.clone(),
            backend: ModelBackend::Ollama {
                model,
                endpoint: Some(endpoint.to_owned()),
            },
        })),
        Err(error) => {
            eprintln!("warning: could not list Ollama fallback models from {endpoint}: {error:#}")
        }
    }

    match tui::select_setup_action(&candidates, current_model, true)? {
        tui::ModelSelection::ConfiguredModel(model) => Ok(model),
        tui::ModelSelection::DownloadDefault => {
            download::download_model(download::DownloadOptions {
                url: download::DEFAULT_MODEL_URL.to_owned(),
                output: None,
                force: false,
                llama_cli,
            })
            .await
        }
    }
}

async fn download_selected_model(cli: &Cli, llama_cli: Option<PathBuf>) -> Result<ConfiguredModel> {
    download::download_model(download::DownloadOptions {
        url: cli
            .download_url
            .clone()
            .unwrap_or_else(|| download::DEFAULT_MODEL_URL.to_owned()),
        output: cli.download_output.clone(),
        force: cli.force_download,
        llama_cli,
    })
    .await
}

fn save_config(
    config: &mut AppConfig,
    selected_model: ConfiguredModel,
    endpoint: &str,
    llama_cli: Option<PathBuf>,
    model_dirs: &[PathBuf],
    locale: GrammarLocale,
) -> Result<()> {
    config.selected_model = Some(selected_model.clone());
    config.model = None;
    config.endpoint = Some(endpoint.to_owned());
    config.llama_cli = llama_cli;
    config.model_dirs = model_dirs.to_vec();
    config.locale = Some(locale);

    let path = config.save()?;
    eprintln!(
        "Saved model `{}` and locale `{}` to {}",
        selected_model.label(),
        locale.label(),
        path.display()
    );
    Ok(())
}

fn cli_model_override(
    cli: &Cli,
    endpoint: &str,
    llama_cli: Option<PathBuf>,
) -> Result<Option<ConfiguredModel>> {
    if let Some(model_path) = &cli.model_path {
        let explicit_backend = cli.backend.and_then(LocalBackend::config_key);
        return configured_model_from_path(model_path.clone(), explicit_backend, llama_cli)
            .map(Some);
    }

    if let Some(model) = &cli.model {
        return Ok(Some(ConfiguredModel {
            name: model.clone(),
            backend: ModelBackend::Ollama {
                model: model.clone(),
                endpoint: Some(endpoint.to_owned()),
            },
        }));
    }

    Ok(None)
}

async fn preferred_config_model(
    config: &AppConfig,
    bundled_model: Option<ConfiguredModel>,
    endpoint: &str,
) -> Option<ConfiguredModel> {
    match config.configured_model() {
        Some(model) if is_deprecated_default_model(&model) => {
            eprintln!(
                "warning: configured model `{}` is the old weak default; opening setup for a better llama.cpp model",
                model.label()
            );
            bundled_model
        }
        Some(model) if local_model_exists(&model) => Some(model),
        Some(model) if is_local_model(&model) => {
            eprintln!(
                "warning: configured local model is missing; ignoring `{}`",
                model.label()
            );
            bundled_model
        }
        Some(model) => {
            if let ModelBackend::Ollama {
                model: model_name, ..
            } = &model.backend
            {
                let model_name = model_name.clone();
                if ollama_model_is_installed(endpoint, &model_name).await {
                    return Some(model);
                }

                eprintln!(
                    "warning: configured Ollama model `{model_name}` is unavailable; using a local/downloaded model instead"
                );
            }

            bundled_model
        }
        None => bundled_model,
    }
}

async fn ollama_model_is_installed(endpoint: &str, model_name: &str) -> bool {
    ollama::list_models(endpoint)
        .await
        .map(|models| models.iter().any(|model| model == model_name))
        .unwrap_or(false)
}

fn is_deprecated_default_model(model: &ConfiguredModel) -> bool {
    ["Qwen2.5-0.5B-Instruct", "qwen2.5-1.5b-instruct"]
        .iter()
        .any(|deprecated_name| {
            model.name.contains(deprecated_name)
                || local_model_path(model)
                    .map(|path| path.to_string_lossy().contains(deprecated_name))
                    .unwrap_or(false)
        })
}

fn is_local_model(model: &ConfiguredModel) -> bool {
    matches!(
        &model.backend,
        ModelBackend::Llamafile { .. } | ModelBackend::LlamaCpp { .. }
    )
}

fn local_model_exists(model: &ConfiguredModel) -> bool {
    local_model_path(model)
        .map(|path| path.exists())
        .unwrap_or(false)
}

fn local_model_path(model: &ConfiguredModel) -> Option<&std::path::Path> {
    match &model.backend {
        ModelBackend::Llamafile { path } => Some(path.as_path()),
        ModelBackend::LlamaCpp { model_path, .. } => Some(model_path.as_path()),
        ModelBackend::Ollama { .. } => None,
    }
}

fn apply_runtime_config(
    mut model: ConfiguredModel,
    endpoint: &str,
    llama_cli: Option<PathBuf>,
) -> ConfiguredModel {
    match &mut model.backend {
        ModelBackend::Ollama {
            endpoint: model_endpoint,
            ..
        } => {
            if model_endpoint.is_none() {
                *model_endpoint = Some(endpoint.to_owned());
            }
        }
        ModelBackend::LlamaCpp {
            llama_cli: model_llama_cli,
            ..
        } => {
            if model_llama_cli.is_none() {
                *model_llama_cli = llama_cli;
            }
        }
        ModelBackend::Llamafile { .. } => {}
    }

    model
}

fn selected_locale(cli: &Cli, config: &AppConfig) -> GrammarLocale {
    if cli.pt_br {
        return GrammarLocale::PtBr;
    }

    cli.locale.or(config.locale).unwrap_or(GrammarLocale::En)
}

fn model_dirs(cli: &Cli, config: &AppConfig) -> Vec<PathBuf> {
    if cli.model_dir.is_empty() {
        return config.model_dirs_or_defaults();
    }

    cli.model_dir.iter().cloned().map(expand_home).collect()
}

fn read_input(args: &[String]) -> Result<String> {
    if !args.is_empty() {
        return Ok(args.join(" "));
    }

    let mut stdin = io::stdin();
    if stdin.is_terminal() {
        bail!("provide text as an argument, pipe text on stdin, or run `lint-lang --configure`");
    }

    let mut buffer = String::new();
    stdin
        .read_to_string(&mut buffer)
        .context("failed to read text from stdin")?;

    while buffer.ends_with('\n') || buffer.ends_with('\r') {
        buffer.pop();
    }

    Ok(buffer)
}

fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut clipboard = Clipboard::new().context("failed to open the system clipboard")?;
    clipboard
        .set_text(text.to_owned())
        .context("failed to write to the system clipboard")
}
