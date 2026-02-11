.PHONY: test lint validate smoke ci

test:  ## Run unit tests (no GPU, no hardware needed)
	python -m pytest tests/ -v

lint:  ## Run ruff linter
	ruff check jetson_assistant/ tests/

validate:  ## Validate docker-compose.yml (needs docker installed)
	docker compose config -q

smoke:  ## Run smoke test (needs pip install done, no GPU)
	bash scripts/smoke-test.sh

ci: lint test validate  ## Full pre-push check (lint + test + validate)
