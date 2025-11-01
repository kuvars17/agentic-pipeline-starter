# Agentic Pipeline Starter - Makefile
# Common development commands for the agentic AI pipeline

.PHONY: help install install-dev test test-cov lint format type-check clean run dev docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Agentic Pipeline Starter - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=agentic_pipeline_starter --cov-report=html --cov-report=term-missing

test-fast: ## Run tests excluding slow tests
	pytest tests/ -v -m "not slow"

lint: ## Run linting with ruff
	ruff check agentic_pipeline_starter/ tests/

lint-fix: ## Run linting with auto-fix
	ruff check --fix agentic_pipeline_starter/ tests/

format: ## Format code with black and ruff
	black agentic_pipeline_starter/ tests/
	ruff format agentic_pipeline_starter/ tests/

type-check: ## Run type checking with mypy
	mypy agentic_pipeline_starter/

check-all: lint type-check test ## Run all quality checks

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

run: ## Run the API server in production mode
	uvicorn agentic_pipeline_starter.api.server:app --host 0.0.0.0 --port 8000

dev: ## Run the API server in development mode
	uvicorn agentic_pipeline_starter.api.server:app --host 127.0.0.1 --port 8000 --reload

dev-mock: ## Run in development mode with mock LLM
	LLM_MODE=mock uvicorn agentic_pipeline_starter.api.server:app --host 127.0.0.1 --port 8000 --reload

dev-ollama: ## Run in development mode with Ollama
	LLM_MODE=ollama uvicorn agentic_pipeline_starter.api.server:app --host 127.0.0.1 --port 8000 --reload

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

build: ## Build the package
	python -m build

release-test: ## Upload to test PyPI
	python -m twine upload --repository testpypi dist/*

release: ## Upload to PyPI
	python -m twine upload dist/*

docker-build: ## Build Docker image
	docker build -t agentic-pipeline-starter:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 agentic-pipeline-starter:latest

# Development workflow shortcuts
dev-setup: install-dev ## Set up development environment
	@echo "Development environment ready!"
	@echo "Run 'make dev' to start the development server"

ci-checks: lint type-check test ## Run all CI checks locally

# Git workflow helpers
git-hooks: ## Install git hooks
	pre-commit install
	@echo "Git hooks installed!"

update-deps: ## Update all dependencies
	pip install --upgrade pip
	pip install --upgrade -r requirements-dev.txt