use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum GrammarLocale {
    #[default]
    En,
    PtBr,
}

impl GrammarLocale {
    pub fn label(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::PtBr => "pt-BR",
        }
    }
}

pub static EN_SYSTEM_PROMPT: &str = "You are a conservative grammar-only correction engine. Fix only grammar, spelling, accents, punctuation, and capitalization. Preserve the original meaning, person, number, verb tense, point of view, tone, vocabulary, formatting, and language. Do not rewrite. Do not improve style. Do not add information. Do not answer questions. Return only the corrected text. Do not add explanations, markdown, labels, or wrapping quotes unless they were present in the original text.";

pub static PT_BR_SYSTEM_PROMPT: &str = "Você é um corretor gramatical conservador para textos em português brasileiro. Corrija somente gramática, ortografia, acentos, pontuação e capitalização. Preserve obrigatoriamente o significado original, pessoa, número, tempo verbal, ponto de vista, tom, vocabulário, formatação e idioma. Não reescreva. Não melhore estilo. Não acrescente informação. Não responda perguntas. Retorne apenas o texto corrigido, sem explicações, markdown, rótulos ou aspas extras.";

static EN_USER_INSTRUCTION: &str = "Correct only grammar, spelling, punctuation, and capitalization in the text below. Preserve every word choice and verb form unless it is grammatically incorrect. Return only the corrected text.";
static PT_BR_USER_INSTRUCTION: &str = "Corrija somente gramática, ortografia, acentos, pontuação e capitalização no texto abaixo. Preserve cada escolha de palavra, pessoa e forma verbal, a menos que esteja gramaticalmente incorreta. Retorne apenas o texto corrigido.";

pub fn system_prompt(locale: GrammarLocale) -> &'static str {
    match locale {
        GrammarLocale::En => EN_SYSTEM_PROMPT,
        GrammarLocale::PtBr => PT_BR_SYSTEM_PROMPT,
    }
}

pub fn user_instruction(locale: GrammarLocale) -> &'static str {
    match locale {
        GrammarLocale::En => EN_USER_INSTRUCTION,
        GrammarLocale::PtBr => PT_BR_USER_INSTRUCTION,
    }
}

pub fn ollama_user_prompt(input: &str, locale: GrammarLocale) -> String {
    format!(
        "{instruction}\n\n{text}",
        instruction = user_instruction(locale),
        text = input.trim()
    )
}

pub fn qwen_instruct_prompt(input: &str, locale: GrammarLocale) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{examples}{instruction}\n/no_think\n\n<text>\n{text}\n</text><|im_end|>\n<|im_start|>assistant\n",
        system = system_prompt(locale),
        examples = examples(locale),
        instruction = user_instruction(locale),
        text = input.trim()
    )
}

pub fn postprocess_correction(output: &str, locale: GrammarLocale) -> String {
    match locale {
        GrammarLocale::En => output.trim().to_owned(),
        GrammarLocale::PtBr => postprocess_pt_br(output),
    }
}

fn postprocess_pt_br(output: &str) -> String {
    output
        .trim()
        .replace("À propósito", "A propósito")
        .replace("à propósito", "a propósito")
}

fn examples(locale: GrammarLocale) -> &'static str {
    match locale {
        GrammarLocale::En => "Example:\nInput: i has a aplle\nOutput: I have an apple.\n\n",
        GrammarLocale::PtBr => {
            "Exemplo:\nEntrada: As verspera da prova, resolvi ir ao cinema. À propósito: penso em chegar tarde\nSaída: Às vésperas da prova, resolvi ir ao cinema. A propósito: penso em chegar tarde.\n\n"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GrammarLocale, PT_BR_SYSTEM_PROMPT, postprocess_correction, qwen_instruct_prompt,
        system_prompt,
    };

    #[test]
    fn exposes_static_pt_br_prompt() {
        assert_eq!(system_prompt(GrammarLocale::PtBr), PT_BR_SYSTEM_PROMPT);
        assert!(PT_BR_SYSTEM_PROMPT.contains("português brasileiro"));
    }

    #[test]
    fn qwen_prompt_uses_selected_locale() {
        let prompt = qwen_instruct_prompt("eu vai", GrammarLocale::PtBr);

        assert!(prompt.contains("português brasileiro"));
        assert!(prompt.contains("eu vai"));
    }

    #[test]
    fn pt_br_example_preserves_person_and_number() {
        let prompt = qwen_instruct_prompt(
            "As verspera da prova, resolvi ir ao cinema. À propósito: penso em chegar tarde",
            GrammarLocale::PtBr,
        );

        assert!(prompt.contains("resolvi ir ao cinema"));
        assert!(!prompt.contains("resolvemos ir ao cinema"));
    }

    #[test]
    fn pt_br_postprocesses_common_crase_error() {
        let output =
            "Às vésperas da prova, resolvi ir ao cinema. À propósito: penso em chegar tarde.";

        assert_eq!(
            postprocess_correction(output, GrammarLocale::PtBr),
            "Às vésperas da prova, resolvi ir ao cinema. A propósito: penso em chegar tarde."
        );
    }
}
