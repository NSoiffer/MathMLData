# Nemeth Braille to MathML Translation Evaluation

This script evaluates the translation ability of OpenAI's o3 reasoning model from Nemeth Braille to MathML using few-shot prompting with explicit encoding rules.

## Overview

The evaluation script:
- Loads paired Nemeth Braille and MathML expressions from the highschool dataset
- Filters pairs of low(ish) MathML length (20-200 characters)
- Randomly samples test pairs for evaluation
- Uses 40 few-shot examples with diverse pattern coverage
- Provides explicit Nemeth Braille encoding rules in the prompt
- Calls OpenAI's `o3` model with high reasoning effort
- Exports timestamped results to CSV files with comprehensive metadata

## Performance

**Current Results (o3 model with 40 few-shot examples):**
- **Semantic Accuracy: about 55%** - Mathematically correct translations

## Requirements

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key with access to `o3` model

## Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

Run the script with uv:
```bash
uv run python nemeth_to_mathml_evaluation.py
```

Or activate the virtual environment and run directly:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python nemeth_to_mathml_evaluation.py
```

The script will:
1. Load and filter highschool Nemeth Braille/MathML pairs
2. Sample 20 test pairs (configurable)
3. Process each pair with 40 few-shot examples
4. Save results to `results/nemeth_mathml_evaluation_results_YYYYMMDD_HHMMSS.csv`
s to line N in `.mmls`).