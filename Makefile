.PHONY: help install setup docker-up docker-down batch-run streaming-run ml-train test clean

help:
	@echo "Urban Energy Trust Lakehouse - Available Commands"
	@echo "=================================================="
	@echo "  make install      - Install Python dependencies"
	@echo "  make setup        - Setup virtual environment and install deps"
	@echo "  make docker-up    - Start Docker Compose services"
	@echo "  make docker-down  - Stop Docker Compose services"
	@echo "  make batch-run    - Run end-to-end batch pipeline"
	@echo "  make streaming-run - Run streaming pipeline"
	@echo "  make ml-train     - Train quality risk ML model"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean temporary files and data"
	@echo ""

install:
	pip install -r requirements.txt

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"
	pip install -r requirements.txt

docker-up:
	docker-compose up -d
	@echo "Services started. Spark UI: http://localhost:8080"
	@echo "Jupyter: http://localhost:8888"

docker-down:
	docker-compose down

batch-run:
	python -m src.pipelines.batch_pipeline

streaming-run:
	python -m src.pipelines.streaming_pipeline

ml-train:
	python -m src.ml.train_quality_model

test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/
	@echo "Cleaned Python cache files"

clean-data:
	rm -rf data/lakehouse/*/.*
	rm -rf data/lakehouse/*/*
	rm -rf data/streaming_checkpoint
	@echo "Cleaned lakehouse data (kept raw data)"

clean-all: clean clean-data
	@echo "Cleaned all temporary files and data"
