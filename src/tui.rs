use std::{
    io::{self, IsTerminal, Stdout, Write},
    time::Duration,
};

use anyhow::{Result, anyhow, bail};
use crossterm::{
    cursor::{Hide, MoveToColumn, Show},
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{
        Clear as TerminalClear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
        disable_raw_mode, enable_raw_mode, size as terminal_size,
    },
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph},
};
use tokio::task::JoinHandle;

use crate::{
    config::ConfiguredModel,
    prompt::GrammarLocale,
    tui_model_list::{ModelChoice, model_choices},
};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelSelection {
    ConfiguredModel(ConfiguredModel),
    DownloadDefault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelListMode {
    Select,
    Browse,
}

pub fn terminal_available() -> bool {
    io::stdout().is_terminal() && io::stdin().is_terminal()
}

pub fn select_setup_action(
    candidates: &[ConfiguredModel],
    current_model: Option<&ConfiguredModel>,
    allow_download: bool,
) -> Result<ModelSelection> {
    if !terminal_available() {
        bail!(
            "setup requires an interactive terminal; run `lint-lang --configure` in a terminal, or pass --download-model/--model-path"
        );
    }

    let choices = model_choices(candidates, current_model, allow_download);
    if choices.is_empty() {
        bail!("no model choices available");
    }

    let mut selected = initial_model_choice(&choices, current_model);
    let mut session = TuiSession::new()?;

    loop {
        session.terminal.draw(|frame| {
            draw_model_selector(frame, &choices, selected, ModelListMode::Select);
        })?;

        if !event::poll(Duration::from_millis(200))? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };

        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => bail!("setup cancelled"),
            KeyCode::Enter => return Ok(choices[selected].selection.clone()),
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(choices.len() - 1);
            }
            KeyCode::Home => selected = 0,
            KeyCode::End => selected = choices.len() - 1,
            _ => {}
        }
    }
}

pub fn browse_model_list(
    candidates: &[ConfiguredModel],
    current_model: Option<&ConfiguredModel>,
    allow_download: bool,
) -> Result<()> {
    if !terminal_available() {
        bail!("model list requires an interactive terminal; pass --plain to print a text list");
    }

    let choices = model_choices(candidates, current_model, allow_download);
    if choices.is_empty() {
        bail!("no model choices available");
    }

    let mut selected = initial_model_choice(&choices, current_model);
    let mut session = TuiSession::new()?;

    loop {
        session.terminal.draw(|frame| {
            draw_model_selector(frame, &choices, selected, ModelListMode::Browse);
        })?;

        if !event::poll(Duration::from_millis(200))? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };

        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc | KeyCode::Enter => return Ok(()),
            KeyCode::Up | KeyCode::Char('k') => selected = selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => selected = (selected + 1).min(choices.len() - 1),
            KeyCode::Home => selected = 0,
            KeyCode::End => selected = choices.len() - 1,
            _ => {}
        }
    }
}

pub async fn wait_with_loading<T>(
    handle: JoinHandle<Result<T>>,
    model_label: &str,
    locale: GrammarLocale,
) -> Result<T>
where
    T: Send + 'static,
{
    if !io::stderr().is_terminal() {
        return await_task(handle).await;
    }

    let mut frame_index = 0usize;
    loop {
        if handle.is_finished() {
            break;
        }

        draw_inline_loading(model_label, locale, frame_index)?;
        frame_index = frame_index.wrapping_add(1);
        tokio::time::sleep(Duration::from_millis(90)).await;
    }

    clear_inline_loading()?;
    await_task(handle).await
}

async fn await_task<T>(handle: JoinHandle<Result<T>>) -> Result<T> {
    match handle.await {
        Ok(result) => result,
        Err(error) if error.is_cancelled() => bail!("grammar fix cancelled"),
        Err(error) => Err(anyhow!("grammar fix task failed: {error}")),
    }
}

fn draw_model_selector(
    frame: &mut Frame<'_>,
    choices: &[ModelChoice],
    selected: usize,
    mode: ModelListMode,
) {
    let area = centered_rect(88, choices.len() as u16 + 8, frame.area());
    frame.render_widget(Clear, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(2),
        ])
        .split(area);

    let help = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                "lint-lang",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(model_list_title(mode)),
        ]),
        Line::from(
            "Choose a local GGUF/.llamafile, an installed Ollama model, or download the default Qwen GGUF.",
        ),
    ])
    .block(Block::default().borders(Borders::ALL).title("Settings"));
    frame.render_widget(help, chunks[0]);

    let items: Vec<ListItem> = choices
        .iter()
        .map(|choice| {
            ListItem::new(Line::from(vec![
                Span::styled(choice.name.clone(), Style::default().fg(Color::White)),
                Span::raw("  "),
                Span::styled(choice.detail.clone(), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(selected));
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Models"))
        .highlight_symbol("› ")
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_stateful_widget(list, chunks[1], &mut state);

    let footer = Paragraph::new(model_list_footer(mode))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(footer, chunks[2]);
}

fn draw_inline_loading(model_label: &str, locale: GrammarLocale, frame_index: usize) -> Result<()> {
    let terminal_width = terminal_size()
        .map(|(width, _)| usize::from(width))
        .unwrap_or(80)
        .saturating_sub(1)
        .max(20);
    let line = truncate_line(
        &loading_line(model_label, locale, frame_index),
        terminal_width,
    );

    let mut stderr = io::stderr();
    execute!(
        stderr,
        MoveToColumn(0),
        TerminalClear(ClearType::CurrentLine)
    )?;
    write!(stderr, "{line}")?;
    stderr.flush()?;
    Ok(())
}

fn clear_inline_loading() -> Result<()> {
    let mut stderr = io::stderr();
    execute!(
        stderr,
        MoveToColumn(0),
        TerminalClear(ClearType::CurrentLine)
    )?;
    stderr.flush()?;
    Ok(())
}

fn loading_line(model_label: &str, locale: GrammarLocale, frame_index: usize) -> String {
    let spinner = SPINNER[frame_index % SPINNER.len()];
    format!(
        "{spinner} Fixing {} grammar with {}",
        locale.label(),
        compact_model_label(model_label)
    )
}

fn compact_model_label(model_label: &str) -> &str {
    model_label
        .split_once(" (")
        .map_or(model_label, |(name, _)| name)
}

fn truncate_line(line: &str, max_chars: usize) -> String {
    if line.chars().count() <= max_chars {
        return line.to_owned();
    }

    let keep = max_chars.saturating_sub(1);
    let mut truncated = line.chars().take(keep).collect::<String>();
    truncated.push('…');
    truncated
}

fn model_list_title(mode: ModelListMode) -> &'static str {
    match mode {
        ModelListMode::Select => " setup needs a grammar model.",
        ModelListMode::Browse => " models found for grammar linting.",
    }
}

fn model_list_footer(mode: ModelListMode) -> &'static str {
    match mode {
        ModelListMode::Select => "↑/↓ or j/k move · Enter saves · Esc/q cancels",
        ModelListMode::Browse => "↑/↓ or j/k move · Enter/Esc/q closes",
    }
}

fn initial_model_choice(choices: &[ModelChoice], current_model: Option<&ConfiguredModel>) -> usize {
    current_model
        .and_then(|current| {
            choices.iter().position(|choice| {
                choice.selection == ModelSelection::ConfiguredModel(current.clone())
            })
        })
        .or_else(|| {
            choices.iter().position(|choice| {
                matches!(
                    choice.selection,
                    ModelSelection::ConfiguredModel(ConfiguredModel {
                        backend: crate::config::ModelBackend::Llamafile { .. },
                        ..
                    }) | ModelSelection::ConfiguredModel(ConfiguredModel {
                        backend: crate::config::ModelBackend::LlamaCpp { .. },
                        ..
                    })
                )
            })
        })
        .or_else(|| {
            choices
                .iter()
                .position(|choice| matches!(choice.selection, ModelSelection::DownloadDefault))
        })
        .unwrap_or(0)
}

fn centered_rect(width_percent: u16, height: u16, area: Rect) -> Rect {
    let height = height.min(area.height.saturating_sub(2)).max(3);
    let vertical_margin = area.height.saturating_sub(height) / 2;

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(vertical_margin),
            Constraint::Length(height),
            Constraint::Min(0),
        ])
        .split(area);

    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_percent) / 2),
            Constraint::Percentage(width_percent),
            Constraint::Percentage((100 - width_percent) / 2),
        ])
        .split(vertical[1]);

    horizontal[1]
}

pub(crate) struct TuiSession {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TuiSession {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }
}

impl Drop for TuiSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), Show, LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[cfg(test)]
mod tests {
    use super::{loading_line, truncate_line};
    use crate::prompt::GrammarLocale;

    #[test]
    fn loading_status_is_single_line() {
        let line = loading_line(
            "Qwen3-8B-Q4_K_M.gguf (/Users/caioborghi/Library/Application Support/com.caio.lint-lang/models/Qwen3-8B-Q4_K_M.gguf)",
            GrammarLocale::PtBr,
            0,
        );

        assert!(line.contains("Fixing pt-BR grammar"));
        assert!(line.contains("Qwen3-8B-Q4_K_M.gguf"));
        assert!(!line.contains("/Users/"));
        assert!(!line.contains('\n'));
    }

    #[test]
    fn truncates_loading_status_to_fit_terminal_width() {
        let line = truncate_line("abcdef", 4);

        assert_eq!(line, "abc…");
        assert_eq!(line.chars().count(), 4);
    }
}
