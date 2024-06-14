lint:
	@echo "Running linter..."
	ruff check .

format:
	@echo "Running formatter..."
	ruff check --fix .
	ruff format .

install:
	@echo "Installing virtual environment and dependencies..."
	pyenv local 3.11.9
	poetry install
	poetry shell
	poetry env info