use std::path::PathBuf;

use anyhow::Result;

use crate::{
    bundled,
    cli::{ModelsCli, ModelsCommand},
    config::{
        AppConfig, ConfiguredModel, DEFAULT_ENDPOINT, config_path, configured_model_from_path,
        expand_home,
    },
    download,
    lint::{collect_model_candidates, configure_model, local_model_path, save_config},
    prompt::GrammarLocale,
    tui,
};

pub async fn run(args: ModelsCli) -> Result<()> {
    let mut config = AppConfig::load()?;
    let endpoint = endpoint_from_models_args(&args, &config);
    let llama_cli = args.llama_cli.clone().or_else(|| config.llama_cli.clone());
    let model_dirs = models_model_dirs(&args, &config);
    let locale = models_locale(&args, &config);

    match &args.command {
        ModelsCommand::List(list_args) => {
            list_models(
                &endpoint,
                &model_dirs,
                llama_cli,
                config.configured_model().as_ref(),
                list_args.plain,
            )
            .await
        }
        ModelsCommand::Download(download_args) => {
            let selected_model = download::download_model(download::DownloadOptions {
                url: download_args
                    .url
                    .clone()
                    .unwrap_or_else(|| download::DEFAULT_MODEL_URL.to_owned()),
                output: download_args.output.clone(),
                force: download_args.force,
                llama_cli: llama_cli.clone(),
            })
            .await?;

            save_config(
                &mut config,
                selected_model.clone(),
                &endpoint,
                llama_cli,
                &model_dirs,
                locale,
            )?;

            if download_args.print_path
                && let Some(path) = local_model_path(&selected_model)
            {
                println!("{}", path.display());
            }

            Ok(())
        }
        ModelsCommand::Select(select_args) => {
            let selected_model = match &select_args.model_path {
                Some(path) => configured_model_from_path(
                    path.clone(),
                    select_args.backend.and_then(|backend| backend.config_key()),
                    llama_cli.clone(),
                )?,
                None => {
                    let bundled_model = bundled::configured_model(llama_cli.clone())?;
                    let current_model = config.configured_model();
                    configure_model(
                        &endpoint,
                        current_model.as_ref(),
                        &model_dirs,
                        llama_cli.clone(),
                        bundled_model,
                    )
                    .await?
                }
            };

            save_config(
                &mut config,
                selected_model,
                &endpoint,
                llama_cli,
                &model_dirs,
                locale,
            )
        }
    }
}

async fn list_models(
    endpoint: &str,
    model_dirs: &[PathBuf],
    llama_cli: Option<PathBuf>,
    current_model: Option<&ConfiguredModel>,
    plain: bool,
) -> Result<()> {
    let bundled_model = bundled::configured_model(llama_cli.clone())?;
    let candidates = collect_model_candidates(endpoint, model_dirs, llama_cli, bundled_model).await;

    if plain || !tui::terminal_available() {
        print_model_list(current_model, &candidates)?;
        return Ok(());
    }

    tui::browse_model_list(&candidates, current_model, true)
}

fn print_model_list(
    current_model: Option<&ConfiguredModel>,
    candidates: &[ConfiguredModel],
) -> Result<()> {
    println!("lint-lang models");
    println!();

    if let Some(current_model) = current_model {
        println!("current: {}", current_model.label());
        println!();
    }

    for candidate in candidates {
        let marker = if Some(candidate) == current_model {
            "*"
        } else {
            "-"
        };
        println!(
            "{marker} {} · {}",
            candidate.name,
            candidate.backend.kind_label()
        );
    }

    println!("- Download Qwen3 8B GGUF · default downloadable model");
    println!();
    println!("config: {}", config_path()?.display());
    Ok(())
}

fn endpoint_from_models_args(args: &ModelsCli, config: &AppConfig) -> String {
    args.endpoint
        .clone()
        .or_else(|| config.endpoint.clone())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned())
}

fn models_model_dirs(args: &ModelsCli, config: &AppConfig) -> Vec<PathBuf> {
    if args.model_dir.is_empty() {
        return config.model_dirs_or_defaults();
    }

    args.model_dir.iter().cloned().map(expand_home).collect()
}

fn models_locale(args: &ModelsCli, config: &AppConfig) -> GrammarLocale {
    if args.pt_br {
        return GrammarLocale::PtBr;
    }

    args.locale.or(config.locale).unwrap_or(GrammarLocale::En)
}

#[cfg(test)]
mod tests {
    use super::print_model_list;
    use crate::config::{ConfiguredModel, ModelBackend};

    #[test]
    fn prints_model_list_without_errors() {
        let model = ConfiguredModel {
            name: "qwen.gguf".to_owned(),
            backend: ModelBackend::LlamaCpp {
                model_path: "qwen.gguf".into(),
                llama_cli: None,
            },
        };

        print_model_list(Some(&model), std::slice::from_ref(&model)).expect("print model list");
    }
}
