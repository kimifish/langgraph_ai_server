# AI Server - Agent Guidelines

## Build/Lint/Test Commands

- **Install dependencies**: `uv sync` or `pip install -r requirements.txt`
- **Run all tests**: `pytest`
- **Run single test**: `pytest tests/unit/test_llms.py::TestLLMs::test_get_llm -v`
- **Run application**: `python -m ai_server.main` or `ai_server` (after installation)
- **Lint**: `ruff check .` and `ruff format .`
- **Format**: `black .`
- **Type check**: `mypy .` or pyright (configured with `# pyright: basic`)
- **Security check**: `bandit -r src/`

## Code Style Guidelines

### Imports
- Standard library first, third-party second, local imports last
- Use absolute imports for local modules

### Types
- Use type hints for parameters and return values
- Use `|` union syntax (Python 3.10+)
- Explicit `None` in type hints when applicable

### Naming
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

### Error Handling
- Use specific exception types in except blocks
- Log errors with appropriate levels
- Return error dicts with "error" and "answer" keys

### Code Structure
- Use dataclasses for config (via kimiconfig)
- Follow async/await patterns for concurrency
- Keep functions focused on single responsibilities
- Use Rich logging with themes and context
