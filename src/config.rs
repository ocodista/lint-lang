use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use directories::{BaseDirs, ProjectDirs};
use serde::{Deserialize, Serialize};

pub const DEFAULT_ENDPOINT: &str = "http://localhost:11434";
pub const DEFAULT_LLAMA_CLI: &str = "llama-cli";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppConfig {
    pub selected_model: Option<ConfiguredModel>,
    pub endpoint: Option<String>,
    pub llama_cli: Option<PathBuf>,
    pub locale: Option<crate::prompt::GrammarLocale>,
    #[serde(default)]
    pub model_dirs: Vec<PathBuf>,

    // Legacy config from the first Ollama-only version. Kept so existing configs migrate.
    pub model: Option<String>,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let path = config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed to read config at {}", path.display()))?;
        toml::from_str(&contents)
            .with_context(|| format!("failed to parse config at {}", path.display()))
    }

    pub fn save(&self) -> Result<PathBuf> {
        let path = config_path()?;
        let parent = path
            .parent()
            .ok_or_else(|| anyhow!("config path has no parent: {}", path.display()))?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;

        let contents = toml::to_string_pretty(self).context("failed to serialize config")?;
        fs::write(&path, contents)
            .with_context(|| format!("failed to write config at {}", path.display()))?;
        Ok(path)
    }

    pub fn configured_model(&self) -> Option<ConfiguredModel> {
        self.selected_model.clone().or_else(|| {
            self.model.as_ref().map(|model| ConfiguredModel {
                name: model.clone(),
                backend: ModelBackend::Ollama {
                    model: model.clone(),
                    endpoint: self.endpoint.clone(),
                },
            })
        })
    }

    pub fn model_dirs_or_defaults(&self) -> Vec<PathBuf> {
        if self.model_dirs.is_empty() {
            return default_model_dirs();
        }

        dedup_paths(self.model_dirs.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ConfiguredModel {
    pub name: String,
    pub backend: ModelBackend,
}

impl ConfiguredModel {
    pub fn label(&self) -> String {
        match &self.backend {
            ModelBackend::Llamafile { path } => format!("{} ({})", self.name, path.display()),
            ModelBackend::LlamaCpp { model_path, .. } => {
                format!("{} ({})", self.name, model_path.display())
            }
            ModelBackend::Ollama { model, .. } => format!("{model} (Ollama)"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ModelBackend {
    /// A self-contained llamafile executable. This is the closest option to a model binary.
    Llamafile { path: PathBuf },

    /// A GGUF weights file executed through a local llama.cpp binary.
    LlamaCpp {
        model_path: PathBuf,
        #[serde(default)]
        llama_cli: Option<PathBuf>,
    },

    /// Legacy daemon mode through Ollama's local HTTP API.
    Ollama {
        model: String,
        #[serde(default)]
        endpoint: Option<String>,
    },
}

impl ModelBackend {
    pub fn kind_label(&self) -> &'static str {
        match self {
            Self::Llamafile { .. } => "llamafile binary",
            Self::LlamaCpp {
                llama_cli: Some(_), ..
            } => "GGUF + llama-cli",
            Self::LlamaCpp { .. } => "GGUF + native llama.cpp",
            Self::Ollama { .. } => "Ollama daemon",
        }
    }
}

pub fn config_path() -> Result<PathBuf> {
    let project_dirs = ProjectDirs::from("com", "caio", "lint-lang")
        .ok_or_else(|| anyhow!("could not find a config directory for this platform"))?;
    Ok(project_dirs.config_dir().join("config.toml"))
}

pub fn default_model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(current_dir) = std::env::current_dir() {
        dirs.push(current_dir.join("models"));
    }

    if let Some(base_dirs) = BaseDirs::new() {
        dirs.push(base_dirs.home_dir().join("models"));
    }

    if let Ok(app_models) = app_model_dir() {
        dirs.push(app_models);
    }

    dedup_paths(dirs)
}

pub fn app_model_dir() -> Result<PathBuf> {
    let project_dirs = ProjectDirs::from("com", "caio", "lint-lang")
        .ok_or_else(|| anyhow!("could not find an app data directory for this platform"))?;
    Ok(project_dirs.data_dir().join("models"))
}

pub fn configured_model_from_path(
    path: PathBuf,
    explicit_backend: Option<&str>,
    llama_cli: Option<PathBuf>,
) -> Result<ConfiguredModel> {
    let expanded_path = expand_home(path);
    let file_name = expanded_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("local-model")
        .to_owned();

    match explicit_backend {
        Some("llamafile") => Ok(ConfiguredModel {
            name: file_name,
            backend: ModelBackend::Llamafile {
                path: expanded_path,
            },
        }),
        Some("llama-cpp") => Ok(ConfiguredModel {
            name: file_name,
            backend: ModelBackend::LlamaCpp {
                model_path: expanded_path,
                llama_cli,
            },
        }),
        Some(other) => bail!("unsupported local backend `{other}`"),
        None => infer_model_from_path(expanded_path, llama_cli),
    }
}

pub fn expand_home(path: PathBuf) -> PathBuf {
    let Some(path_as_str) = path.to_str() else {
        return path;
    };

    if path_as_str == "~"
        && let Some(base_dirs) = BaseDirs::new()
    {
        return base_dirs.home_dir().to_path_buf();
    }

    if let Some(rest) = path_as_str.strip_prefix("~/")
        && let Some(base_dirs) = BaseDirs::new()
    {
        return base_dirs.home_dir().join(rest);
    }

    path
}

fn infer_model_from_path(path: PathBuf, llama_cli: Option<PathBuf>) -> Result<ConfiguredModel> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("local-model")
        .to_owned();

    if extension_is(&path, "gguf") {
        return Ok(ConfiguredModel {
            name: file_name,
            backend: ModelBackend::LlamaCpp {
                model_path: path,
                llama_cli,
            },
        });
    }

    if extension_is(&path, "llamafile") || is_probably_executable(&path) {
        return Ok(ConfiguredModel {
            name: file_name,
            backend: ModelBackend::Llamafile { path },
        });
    }

    bail!(
        "could not infer local model backend for {}; use --backend llamafile or --backend llama-cpp",
        path.display()
    )
}

pub fn extension_is(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();

    for path in paths {
        if seen.insert(path.clone()) {
            deduped.push(path);
        }
    }

    deduped
}

#[cfg(unix)]
fn is_probably_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    path.metadata()
        .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_probably_executable(path: &Path) -> bool {
    extension_is(path, "exe")
}

#[cfg(test)]
mod tests {
    use super::{ModelBackend, configured_model_from_path, expand_home};
    use std::path::PathBuf;

    #[test]
    fn infers_gguf_as_llama_cpp() {
        let model = configured_model_from_path(PathBuf::from("qwen.gguf"), None, None).unwrap();

        assert!(matches!(model.backend, ModelBackend::LlamaCpp { .. }));
    }

    #[test]
    fn infers_llamafile_as_binary() {
        let model =
            configured_model_from_path(PathBuf::from("qwen.llamafile"), None, None).unwrap();

        assert!(matches!(model.backend, ModelBackend::Llamafile { .. }));
    }

    #[test]
    fn expands_home_prefix() {
        let expanded = expand_home(PathBuf::from("~/models/qwen.gguf"));

        assert!(!expanded.to_string_lossy().starts_with('~'));
    }
}
