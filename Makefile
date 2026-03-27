.PHONY: install install-dev test lint clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,train,curve]"

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=rl_drone --cov-report=term-missing

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
