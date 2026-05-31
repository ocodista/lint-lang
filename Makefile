CARGO ?= cargo
BIN := lint-lang
TEXT ?= i has a apple
FIRST_GOAL := $(firstword $(MAKECMDGOALS))
EXTRA_TEXT := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
RUN_GOALS := run run-ptbr run-release run-release-ptbr
ifneq ($(filter $(RUN_GOALS),$(FIRST_GOAL)),)
ifneq ($(strip $(EXTRA_TEXT)),)
TEXT := $(EXTRA_TEXT)
endif
endif

LOCALE ?= pt-br
MODEL_PATH ?=
LLAMA_CLI ?=
DEFAULT_MODEL_URL ?= https://huggingface.co/ggml-org/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
DEFAULT_MODEL_FILENAME ?= Qwen3-8B-Q4_K_M.gguf
DEFAULT_MODEL_PATH ?= $(HOME)/models/$(DEFAULT_MODEL_FILENAME)
MODEL_URL ?= $(DEFAULT_MODEL_URL)
DOWNLOAD_MODEL_PATH = $(if $(strip $(MODEL_PATH)),$(MODEL_PATH),$(DEFAULT_MODEL_PATH))
LOCALE_ARGS := $(if $(strip $(LOCALE)),--locale "$(LOCALE)",)
LLAMA_CLI_ARGS := $(if $(strip $(LLAMA_CLI)),--llama-cli "$(LLAMA_CLI)",)
MODEL_ARGS := $(if $(strip $(MODEL_PATH)),--model-path "$(MODEL_PATH)" $(LLAMA_CLI_ARGS),)

.PHONY: help build release download-model bundle run run-ptbr run-release run-release-ptbr configure configure-ptbr test fmt fmt-check clippy check install config-path reset-config clean

help:
	@printf "\n$(BIN) targets:\n"
	@printf "  make build                         Build debug binary\n"
	@printf "  make release                       Build release binary\n"
	@printf "  make download-model                Download default Qwen GGUF\n"
	@printf "  make bundle                        Download default model and embed it\n"
	@printf "  make bundle MODEL_PATH='...'       Download if missing, then embed it\n"
	@printf "  make run TEXT='...'                Run; opens setup TUI first if needed\n"
	@printf "  make run 'text here'               Same as above, quick shell style\n"
	@printf "  make run-ptbr TEXT='...'           Run with pt-BR prompt\n"
	@printf "  make run-release TEXT='...'        Run target/release/lint-lang\n"
	@printf "  make run-release-ptbr TEXT='...'   Run target/release/lint-lang with pt-BR\n"
	@printf "  make configure                     Open model config TUI\n"
	@printf "  make configure-ptbr                Open model config TUI and save pt-BR locale\n"
	@printf "  make check                         fmt-check + tests + clippy\n"
	@printf "  make install                       cargo install --path .\n"
	@printf "  make config-path                   Print saved config path\n"
	@printf "  make reset-config                  Delete saved config\n"
	@printf "  make clean                         Remove build artifacts\n\n"
	@printf "Optional vars: TEXT, LOCALE (default pt-br), MODEL_PATH, MODEL_URL, LLAMA_CLI\n"
	@printf "Example: make run-ptbr TEXT='eu vai no mercado'\n"
	@printf "Example: make run 'As vespera da prova, resolvi ir ao cinema'\n"
	@printf "Example: make bundle MODEL_PATH='~/models/qwen.llamafile'\n\n"

build:
	$(CARGO) build

release:
	$(CARGO) build --release

download-model:
	@set -eu; \
		model_path="$$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' '$(DOWNLOAD_MODEL_PATH)')"; \
		if [ -f "$$model_path" ]; then \
			case "$$model_path" in *.gguf) ;; *) chmod +x "$$model_path" ;; esac; \
			echo "model already exists: $$model_path"; \
			exit 0; \
		fi; \
		mkdir -p "$$(dirname "$$model_path")"; \
		echo "Downloading $(MODEL_URL)"; \
		echo "Saving to $$model_path"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L --fail --continue-at - -o "$$model_path" "$(MODEL_URL)"; \
		else \
			python3 -c 'import sys,urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$(MODEL_URL)" "$$model_path"; \
		fi; \
		case "$$model_path" in *.gguf) ;; *) chmod +x "$$model_path" ;; esac; \
		echo "downloaded $$model_path"

bundle: download-model
	@set -eu; \
		model_path="$$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' '$(DOWNLOAD_MODEL_PATH)')"; \
		echo "Embedding $$model_path into target/release/$(BIN)"; \
		LINT_LANG_BUNDLED_MODEL="$$model_path" $(CARGO) build --release --features bundled-model

run:
	$(CARGO) run -- $(MODEL_ARGS) $(LOCALE_ARGS) "$(TEXT)"

run-ptbr:
	$(CARGO) run -- $(MODEL_ARGS) --pt-br "$(TEXT)"

run-release:
	./target/release/$(BIN) $(MODEL_ARGS) $(LOCALE_ARGS) "$(TEXT)"

run-release-ptbr:
	./target/release/$(BIN) $(MODEL_ARGS) --pt-br "$(TEXT)"

configure:
	$(CARGO) run -- --configure $(MODEL_ARGS) $(LOCALE_ARGS)

configure-ptbr:
	$(CARGO) run -- --configure $(MODEL_ARGS) --pt-br

test:
	$(CARGO) test

fmt:
	$(CARGO) fmt

fmt-check:
	$(CARGO) fmt --check

clippy:
	$(CARGO) clippy -- -D warnings

check: fmt-check test clippy

install:
	$(CARGO) install --path .

config-path:
	@$(CARGO) run --quiet -- --config-path

reset-config:
	@path="$$($(CARGO) run --quiet -- --config-path)"; \
		rm -f "$$path"; \
		echo "removed $$path"

clean:
	$(CARGO) clean

ifneq ($(filter $(RUN_GOALS),$(FIRST_GOAL)),)
%:
	@:
endif
