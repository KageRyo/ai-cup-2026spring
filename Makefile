PYTHON ?= python
IMAGE ?= ghcr.io/kageryo/ai-cup-2026spring:local
GPU_IMAGE ?= ghcr.io/kageryo/ai-cup-2026spring:gpu-local

.PHONY: install install-dev lint pycompile check tabular-help lstm-help audit-help docker-build docker-build-gpu docker-run-help docker-run-gpu-help

install:
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e .

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check src

pycompile:
	$(PYTHON) -m py_compile src/*.py

tabular-help:
	aicup-tabular-baseline --help

lstm-help:
	aicup-lstm-baseline --help

audit-help:
	aicup-audit-validation --help

check: lint pycompile tabular-help lstm-help audit-help

docker-build:
	docker build -t $(IMAGE) .

docker-build-gpu:
	docker build -f Dockerfile.gpu -t $(GPU_IMAGE) .

docker-run-help:
	docker run --rm $(IMAGE)

docker-run-gpu-help:
	docker run --rm --gpus all $(GPU_IMAGE)
