lint:
	@echo "Running linter..."
	ruff check .

format:
	@echo "Running formatter..."
	ruff check --fix .
	ruff format .