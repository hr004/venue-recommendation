SHELL = /usr/bin/env bash

install-python:
	uv python install 3.12.6
	uv venv
	source .venv/bin/activate
	uv sync

activate-venv:
	source .venv/bin/activate

install-dependencies:
	uv sync

run-server:
	PYTHONPATH=. uv run python src/main.py

index-documents:
	curl -X POST http://localhost:8000/api/v1/index \
		-H "Content-Type: application/json" \
		-d '{"event_history_path": "data/venue/event_history.json"}'

start-opensearch:
	docker compose -f local/opensearch/docker-compose.yml up -d
	./local/opensearch/create_index.sh
	echo "OpenSearch started and index created"

run: install-dependencies start-opensearch run-server
	@echo "Server started. Run 'make index-documents' separately to index data."

.PHONY: install-python activate-venv install-dependencies run-server index-documents start-opensearch run