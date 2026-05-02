.PHONY: install install-dev format comment-check hygiene-check quality-gate download-model preflight smoke build verify release-check clean

TOOLS = PYTHONPATH=src python -m RaCoT.tools

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install twine

format:
	python -m black .

comment-check:
	$(TOOLS) comment

hygiene-check:
	$(TOOLS) hygiene

quality-gate:
	$(TOOLS) quality

download-model:
	$(TOOLS) download-model

preflight:
	python -m compileall -q .
	python examples/minimal_racot_demo.py
	$(TOOLS) quality
	if [ -d tests ] && find tests -name 'test_*.py' -print -quit | grep -q .; then \
		python -m pytest -q; \
	else \
		echo "No tests found; skipping pytest."; \
	fi

smoke: preflight

build:
	python -m build

verify: preflight
	python -m black --check .
	python -m build

release-check: verify
	python -m twine check dist/*

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name "build" -prune -exec rm -rf {} +
	find . -type d -name "dist" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
