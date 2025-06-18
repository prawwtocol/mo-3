.PHONY: install run lint format clean all

# ====================================================================================
#  HELP
# ====================================================================================
help:
	@echo "Commands:"
	@echo "  install    : Install all dependencies using Poetry."
	@echo "  run        : Run the main experiment script."
	@echo "  lint       : Lint the code using ruff."
	@echo "  format     : Format the code using ruff."
	@echo "  clean      : Clean up temporary Python files."
	@echo "  all        : Run install, format, lint, and run."

# ====================================================================================
#  MAIN COMMANDS
# ====================================================================================

all: install format lint run

install:
	@echo "--> Installing dependencies with Poetry..."
	@poetry install

run:
	@echo "--> Running the main script..."
	@poetry run python src/main.py

lint:
	@echo "--> Running linter (ruff)..."
	@poetry run ruff check .

format:
	@echo "--> Running formatter (ruff)..."
	@poetry run ruff format .

clean:
	@echo "--> Cleaning up..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name ".DS_Store" -delete 