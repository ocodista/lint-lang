use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;

use crate::config::{ConfiguredModel, ModelBackend, extension_is};

const MAX_SCAN_DEPTH: u8 = 3;
const MAX_MODELS: usize = 200;

pub fn discover_local_models(
    model_dirs: &[PathBuf],
    llama_cli: Option<PathBuf>,
) -> Vec<ConfiguredModel> {
    let mut models = Vec::new();
    let mut seen_paths = HashSet::new();

    for dir in model_dirs {
        if models.len() >= MAX_MODELS {
            break;
        }

        scan_dir(dir, 0, &mut seen_paths, &mut models, llama_cli.clone());
    }

    sort_models(&mut models);
    models
}

fn scan_dir(
    dir: &Path,
    depth: u8,
    seen_paths: &mut HashSet<PathBuf>,
    models: &mut Vec<ConfiguredModel>,
    llama_cli: Option<PathBuf>,
) {
    if depth > MAX_SCAN_DEPTH || models.len() >= MAX_MODELS {
        return;
    }

    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        if models.len() >= MAX_MODELS {
            return;
        }

        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if file_type.is_dir() {
            scan_dir(&path, depth + 1, seen_paths, models, llama_cli.clone());
            continue;
        }

        let Some(model) = model_from_file(&path, llama_cli.clone()) else {
            continue;
        };

        let canonical_path = model_path(&model)
            .and_then(canonicalize_if_possible)
            .unwrap_or_else(|| path.clone());
        if seen_paths.insert(canonical_path) {
            models.push(model);
        }
    }
}

fn model_from_file(path: &Path, llama_cli: Option<PathBuf>) -> Option<ConfiguredModel> {
    let file_name = path.file_name()?.to_str()?.to_owned();

    if extension_is(path, "llamafile") {
        return Some(ConfiguredModel {
            name: file_name,
            backend: ModelBackend::Llamafile {
                path: path.to_path_buf(),
            },
        });
    }

    if extension_is(path, "gguf") {
        return Some(ConfiguredModel {
            name: file_name,
            backend: ModelBackend::LlamaCpp {
                model_path: path.to_path_buf(),
                llama_cli,
            },
        });
    }

    None
}

fn model_path(model: &ConfiguredModel) -> Option<&Path> {
    match &model.backend {
        ModelBackend::Llamafile { path } => Some(path),
        ModelBackend::LlamaCpp { model_path, .. } => Some(model_path),
        ModelBackend::Ollama { .. } => None,
    }
}

fn canonicalize_if_possible(path: &Path) -> Option<PathBuf> {
    path.canonicalize().ok()
}

fn sort_models(models: &mut [ConfiguredModel]) {
    models.sort_by(|left, right| {
        let left_qwen = left.name.to_lowercase().contains("qwen");
        let right_qwen = right.name.to_lowercase().contains("qwen");

        right_qwen
            .cmp(&left_qwen)
            .then_with(|| backend_rank(left).cmp(&backend_rank(right)))
            .then_with(|| left.name.cmp(&right.name))
    });
}

fn backend_rank(model: &ConfiguredModel) -> u8 {
    match &model.backend {
        ModelBackend::Llamafile { .. } => 0,
        ModelBackend::LlamaCpp { .. } => 1,
        ModelBackend::Ollama { .. } => 2,
    }
}

#[allow(dead_code)]
pub fn validate_local_model(model: &ConfiguredModel) -> Result<()> {
    match &model.backend {
        ModelBackend::Llamafile { path } => {
            if !path.exists() {
                anyhow::bail!("llamafile not found: {}", path.display());
            }
        }
        ModelBackend::LlamaCpp { model_path, .. } => {
            if !model_path.exists() {
                anyhow::bail!("GGUF model not found: {}", model_path.display());
            }
        }
        ModelBackend::Ollama { .. } => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::discover_local_models;
    use crate::config::ModelBackend;
    use std::{fs, path::PathBuf};

    #[test]
    fn discovers_gguf_and_llamafile_models() {
        let root = std::env::temp_dir().join(format!("lint-lang-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("qwen.gguf"), "").unwrap();
        fs::write(root.join("qwen.llamafile"), "").unwrap();
        fs::write(root.join("notes.txt"), "").unwrap();

        let models = discover_local_models(&[root.clone()], Some(PathBuf::from("llama-cli")));

        assert_eq!(models.len(), 2);
        assert!(matches!(models[0].backend, ModelBackend::Llamafile { .. }));
        assert!(
            models
                .iter()
                .any(|model| matches!(model.backend, ModelBackend::Llamafile { .. }))
        );
        assert!(
            models
                .iter()
                .any(|model| matches!(model.backend, ModelBackend::LlamaCpp { .. }))
        );

        let _ = fs::remove_dir_all(&root);
    }
}
