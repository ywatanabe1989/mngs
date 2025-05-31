# Makefile for MNGS development

.PHONY: help install test lint docs clean build release

help:
	@echo "Available commands:"
	@echo "  make install    Install package in development mode"
	@echo "  make test       Run all tests"
	@echo "  make test-fast  Run tests in parallel"
	@echo "  make lint       Run code quality checks"
	@echo "  make format     Format code with black and isort"
	@echo "  make docs       Build documentation"
	@echo "  make clean      Clean build artifacts"
	@echo "  make build      Build distribution packages"
	@echo "  make release    Create a new release"

install:
	pip install -e .
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:
	python -m pytest tests/ -v --cov=src/mngs --cov-report=html --cov-report=term

test-fast:
	python -m pytest tests/ -v -n auto --cov=src/mngs

test-module:
	@echo "Usage: make test-module MODULE=gen"
	python -m pytest tests/mngs/$(MODULE) -v

lint:
	flake8 src/mngs --max-line-length=88 --extend-ignore=E203,W503
	black --check src/mngs tests
	isort --check-only src/mngs tests
	mypy src/mngs --ignore-missing-imports

format:
	black src/mngs tests
	isort src/mngs tests

docs:
	cd docs && make clean && make html
	@echo "Documentation built in docs/_build/html/"

clean:
	rm -rf build dist *.egg-info
	rm -rf .coverage htmlcov .pytest_cache
	rm -rf docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

release: test lint
	@echo "Creating release..."
	@echo "Current version: $$(python -c 'import mngs; print(mngs.__version__)')"
	@echo "Enter new version: "; read VERSION; \
	echo "__version__ = '$$VERSION'" > src/mngs/__version__.py; \
	git add src/mngs/__version__.py; \
	git commit -m "Bump version to $$VERSION"; \
	git tag -a "v$$VERSION" -m "Release version $$VERSION"; \
	echo "Version bumped to $$VERSION"
	@echo "Run 'git push && git push --tags' to publish"

# Development shortcuts
dev-install:
	pip install -e ".[dev]"

dev-test:
	python -m pytest tests/ -v -x --tb=short

dev-coverage:
	python -m pytest tests/ --cov=src/mngs --cov-report=html
	open htmlcov/index.html

# CI/CD helpers
ci-test:
	python -m pytest tests/ --cov=src/mngs --cov-report=xml

ci-lint:
	flake8 src/mngs --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check src/mngs
	isort --check-only src/mngs

# Documentation helpers
docs-serve:
	cd docs/_build/html && python -m http.server

docs-watch:
	cd docs && sphinx-autobuild . _build/html