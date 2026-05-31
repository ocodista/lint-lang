mod bundled;
mod cli;
mod config;
mod discovery;
mod doctor;
mod download;
mod inference;
mod lint;
mod models;
mod ollama;
mod prompt;
mod tui;
mod tui_model_list;

use anyhow::Result;

use crate::cli::DomainCommand;

#[tokio::main]
async fn main() -> Result<()> {
    match cli::parse_domain_command() {
        DomainCommand::Lint(cli) => lint::run(cli.lint).await,
        DomainCommand::Models(cli) => models::run(cli).await,
        DomainCommand::Doctor => doctor::run().await,
    }
}
