use std::collections::HashSet;

use crate::{config::ConfiguredModel, tui::ModelSelection};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelChoice {
    pub(crate) selection: ModelSelection,
    pub(crate) name: String,
    pub(crate) detail: String,
}

pub(crate) fn model_choices(
    candidates: &[ConfiguredModel],
    current_model: Option<&ConfiguredModel>,
    allow_download: bool,
) -> Vec<ModelChoice> {
    let mut seen = HashSet::new();
    let mut choices = Vec::new();

    if let Some(current) = current_model {
        add_choice(
            &mut choices,
            &mut seen,
            ModelSelection::ConfiguredModel(current.clone()),
            current.name.clone(),
            format!("current config · {}", current.backend.kind_label()),
        );
    }

    for candidate in candidates {
        add_choice(
            &mut choices,
            &mut seen,
            ModelSelection::ConfiguredModel(candidate.clone()),
            candidate.name.clone(),
            candidate_detail(candidate),
        );
    }

    if allow_download {
        add_choice(
            &mut choices,
            &mut seen,
            ModelSelection::DownloadDefault,
            "Download Qwen3 8B GGUF".to_owned(),
            "default setup · native llama.cpp + Metal on Mac · no Ollama needed · ~5.0 GB"
                .to_owned(),
        );
    }

    choices
}

fn candidate_detail(candidate: &ConfiguredModel) -> String {
    match &candidate.backend {
        crate::config::ModelBackend::Llamafile { path } => {
            format!("{} · {}", candidate.backend.kind_label(), path.display())
        }
        crate::config::ModelBackend::LlamaCpp { model_path, .. } => {
            format!(
                "{} · {}",
                candidate.backend.kind_label(),
                model_path.display()
            )
        }
        crate::config::ModelBackend::Ollama { .. } => candidate.backend.kind_label().to_owned(),
    }
}

fn add_choice(
    choices: &mut Vec<ModelChoice>,
    seen: &mut HashSet<ModelSelection>,
    selection: ModelSelection,
    name: String,
    detail: String,
) {
    if seen.insert(selection.clone()) {
        choices.push(ModelChoice {
            selection,
            name,
            detail,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::model_choices;
    use crate::{
        config::{ConfiguredModel, ModelBackend},
        tui::ModelSelection,
    };
    use std::path::PathBuf;

    #[test]
    fn keeps_current_model_first() {
        let current = ConfiguredModel {
            name: "current.llamafile".to_owned(),
            backend: ModelBackend::Llamafile {
                path: PathBuf::from("current.llamafile"),
            },
        };
        let candidate = ConfiguredModel {
            name: "qwen.gguf".to_owned(),
            backend: ModelBackend::LlamaCpp {
                model_path: PathBuf::from("qwen.gguf"),
                llama_cli: None,
            },
        };

        let choices = model_choices(&[candidate], Some(&current), false);

        assert_eq!(
            choices[0].selection,
            ModelSelection::ConfiguredModel(current)
        );
    }

    #[test]
    fn removes_duplicate_candidates() {
        let candidate = ConfiguredModel {
            name: "qwen.gguf".to_owned(),
            backend: ModelBackend::LlamaCpp {
                model_path: PathBuf::from("qwen.gguf"),
                llama_cli: None,
            },
        };

        let choices = model_choices(&[candidate.clone(), candidate], None, false);

        assert_eq!(choices.len(), 1);
    }

    #[test]
    fn adds_download_choice_for_setup() {
        let choices = model_choices(&[], None, true);

        assert_eq!(choices[0].selection, ModelSelection::DownloadDefault);
    }
}
