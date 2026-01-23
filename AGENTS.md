# AGENTS.md - Agentic Coding Guidelines

This document provides guidelines for agentic coding assistants working in this repository. It contains build/lint/test commands and code style guidelines to ensure consistency across the codebase.

## Repository Overview

This is a Python course project repository for "AI for Business" from Columbia University. The repository contains multiple assignments, each with their own Python environment and dependencies.

## Project Structure

- **Assignment-1/**: Linear regression on real estate prices
- **Assignment-2/**: Handwritten digit classification with logistic regression
- Each assignment has its own virtual environment managed by `uv`
- Python 3.12+ required

## Build/Lint/Test Commands

### Environment Setup

Each assignment directory contains its own virtual environment:

```bash
# For Assignment-1
cd Assignment-1
uv sync  # Install dependencies

# For Assignment-2
cd Assignment-2
uv sync  # Install dependencies
```

### Running Code

```bash
# Run main script for Assignment-1
cd Assignment-1
uv run python main.py

# Run assignment script for Assignment-1
cd Assignment-1
uv run python Assignment_1.py

# Run main script for Assignment-2
cd Assignment-2
uv run python main.py

# Run assignment script for Assignment-2
cd Assignment-2
uv run python Assignment_2.py
```

### Testing Commands

Since this is a course project, there are currently no formal test suites. For future development, implement tests using pytest:

```bash
# Install testing dependencies (add to pyproject.toml)
uv add --dev pytest pytest-cov

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=.

# Run a specific test file
uv run pytest tests/test_assignment_1.py

# Run a single test function
uv run pytest tests/test_assignment_1.py::test_linear_regression -v

# Run tests in verbose mode
uv run pytest -v

# Run tests with detailed output
uv run pytest -s
```

### Linting and Code Quality

For code quality assurance, use these tools:

```bash
# Install linting tools
uv add --dev flake8 black isort mypy

# Code formatting with Black
uv run black .

# Import sorting with isort
uv run isort .

# Linting with flake8
uv run flake8 .

# Type checking with mypy
uv run mypy .

# Run all quality checks together
uv run black . && uv run isort . && uv run flake8 . && uv run mypy .
```

### Jupyter Notebook Execution

```bash
# Run Jupyter notebooks (if needed)
uv run jupyter nbconvert --to notebook --execute Assignment_1.ipynb
uv run jupyter nbconvert --to notebook --execute Assignment_2.ipynb
```

## Code Style Guidelines

### Python Version and Environment

- **Python Version**: 3.12+ (as specified in pyproject.toml)
- **Virtual Environment**: Use `uv` for dependency management
- **Project Structure**: Each assignment in its own directory with isolated environment

### File Structure and Naming

- **Python Files**: Use `snake_case` for filenames (e.g., `assignment_1.py`, `main.py`)
- **Jupyter Notebooks**: Use `PascalCase` with underscores (e.g., `Assignment_1.ipynb`)
- **Directories**: Use `kebab-case` for assignment directories (e.g., `Assignment-1`, `Assignment-2`)
- **Generated Files**: Save plots and outputs in the same directory as the script

### Imports

```python
# Standard library imports first
import os
import sys
from pathlib import Path

# Third-party imports (alphabetically sorted)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Local imports (if any)
# from .local_module import something
```

- Group imports by type (standard library, third-party, local)
- Sort imports alphabetically within each group
- Use absolute imports over relative imports when possible
- Avoid wildcard imports (`from module import *`)

### Code Formatting

- **Line Length**: Maximum 88 characters (Black's default)
- **Indentation**: 4 spaces (standard Python)
- **Quotes**: Use double quotes for strings, single quotes for character constants
- **Trailing Commas**: Include in multi-line structures for cleaner diffs

```python
# Good
features = [
    "House age",
    "Distance to MRT",
    "Number of stores",
]

# Avoid
features = ["House age", "Distance to MRT", "Number of stores"]
```

### Naming Conventions

- **Variables**: `snake_case` (e.g., `house_age`, `predicted_values`, `mse_score`)
- **Functions**: `snake_case` (e.g., `generate_model()`, `plot_results()`)
- **Classes**: `PascalCase` (if any classes are added)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `RANDOM_STATE = 42`)
- **Descriptive Names**: Use meaningful names that explain purpose
  - ✅ `training_features`, `test_predictions`
  - ❌ `x`, `y`, `data2`

### Documentation and Comments

- **Docstrings**: Use triple-quoted docstrings for all functions

```python
def generate_model(X_train, y_train):
    """
    Trains a LinearRegression model and returns the fitted model.

    Parameters:
        X_train: Training features
        y_train: Training target values

    Returns:
        fitted sklearn LinearRegression model
    """
```

- **Comments**: Use inline comments for complex logic
- **Section Headers**: Use comment blocks to separate major sections

```python
# ===========================
# 1. IMPORT LIBRARIES
# ===========================

# ===========================
# 2. LOAD THE DATASET
# ===========================
```

### Data Science Specific Guidelines

- **Random State**: Always use `random_state=42` for reproducibility
- **Train/Test Split**: Standard 80/20 split unless specified otherwise
- **Plot Configuration**: Use consistent matplotlib settings

```python
# Configure plot appearance
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Save plots with consistent parameters
plt.savefig('filename.png', dpi=300, bbox_inches='tight')
```

- **Data Handling**: Use pandas for data manipulation, numpy for numerical operations
- **Model Evaluation**: Report relevant metrics (MSE, accuracy, etc.) with appropriate precision

### Error Handling

- **Try/Except Blocks**: Use for operations that might fail (file I/O, network requests)

```python
try:
    data = pd.read_csv('RealEstate.csv')
except FileNotFoundError:
    print("Error: RealEstate.csv not found")
    sys.exit(1)
```

- **Validation**: Check assumptions about data shapes and types
- **Informative Messages**: Provide clear error messages for debugging

### Performance and Best Practices

- **Vectorized Operations**: Prefer pandas/numpy vectorized operations over loops
- **Memory Efficiency**: Be mindful of large datasets, use appropriate data types
- **Reproducibility**: Set random seeds, save model parameters
- **Modularity**: Break complex operations into well-defined functions

### File Organization

- **Data Files**: Store in assignment directories alongside code
- **Output Files**: Save generated plots and results in the same directory
- **Virtual Environment**: Use `.venv/` (already in .gitignore)
- **Cache Files**: `__pycache__/` and `*.pyc` files are ignored

### Git and Version Control

- **Commit Messages**: Use descriptive messages explaining what changed and why
- **Branching**: Create feature branches for new work
- **Secrets**: Never commit API keys, credentials, or sensitive data

### Testing Guidelines (Future Implementation)

When adding tests:

- **Test Structure**: `tests/` directory with `test_*.py` files
- **Test Naming**: `test_function_name` or `TestClassName`
- **Coverage**: Aim for high coverage of critical functions
- **Mocking**: Use for external dependencies (APIs, file I/O)

```python
def test_linear_regression_fit():
    # Test that model fits without errors
    pass

def test_prediction_accuracy():
    # Test prediction quality metrics
    pass
```

### IDE and Development Tools

- **Recommended Extensions**: Python, Jupyter, Black Formatter, Pylance
- **Code Analysis**: Enable pylint/flake8 integration
- **Jupyter**: Use `%matplotlib inline` for notebook plotting

This document should be updated as the project evolves and new tools or conventions are adopted.</content>
<parameter name="filePath">/home/y/MY_PROJECTS/AI-for-Business-Columbia+University/AGENTS.md