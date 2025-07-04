.DEFAULT: help

help:
	@echo "venv"
	@echo "        Create virtual environment"
	@echo "install"
	@echo "        Install laplax and dependencies"
	@echo "install-dev"
	@echo "        Install laplax and development tools"
	@echo "install-notebooks"
	@echo "        Install laplax and tools for the tutorial notebooks"
	@echo "install-docs"
	@echo "        Install laplax and documentation tools"
	@echo "lint"
	@echo "        Run all linting actions"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "ruff-format"
	@echo "        Run ruff format on the project"
	@echo "ruff-format-check"
	@echo "        Check if ruff format would change files"
	@echo "ruff"
	@echo "        Run ruff on the project and fix errors"
	@echo "ruff-check"
	@echo "        Run ruff check on the project without fixing errors"

.PHONY: venv

venv:
	@uv venv --python=3.13

.PHONY: install

install:
	@uv sync

.PHONY: install-dev

install-dev:
	@uv sync --inexact --extra dev

.PHONY: install-notebooks

install-notebooks:
	@uv sync --inexact --extra notebooks

.PHONY: install-docs

install-docs:
	@uv sync --inexact --extra docs

.PHONY: test

test:
	@uv run pytest -vx --cov=laplax

.PHONY: ruff-format ruff-format-check

ruff-format:
	@uv run ruff format .

ruff-format-check:
	@uv run ruff format --check .

.PHONY: ruff-check

ruff:
	@uv run ruff check . --fix

ruff-check:
	@uv run ruff check .

.PHONY: lint

lint:
	make ruff-format-check
	make ruff-check
