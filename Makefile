.PHONY: help install install-dev test lint format check run-dashboard run-api clean

help:
	@echo "Available commands:"
	@echo "  make install        Install runtime dependencies"
	@echo "  make install-dev    Install runtime and development dependencies"
	@echo "  make test           Run the test suite"
	@echo "  make lint           Check code with Ruff"
	@echo "  make format         Format code with Ruff"
	@echo "  make check          Run linting and tests"
	@echo "  make run-dashboard  Start the Streamlit dashboard"
	@echo "  make run-api        Start the FastAPI server"
	@echo "  make clean          Remove Python cache files"

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest

lint:
	ruff check src tests app

format:
	ruff format src tests app

check: lint test

run-dashboard:
	streamlit run app/Home.py

run-api:
	uvicorn elliptic_fraud.serving.main:app --host 127.0.0.1 --port 8000

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete