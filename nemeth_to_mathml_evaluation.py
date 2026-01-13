#!/usr/bin/env python3
"""
Nemeth Braille to MathML Translation Evaluation Script

Requirements:
    - Python 3.8+
    - openai library
    - Environment variable OPENAI_API_KEY must be set

Installation:
    uv sync

Usage:
    export OPENAI_API_KEY="your-api-key"
    uv run python nemeth_to_mathml_evaluation.py
"""

import asyncio
import csv
import glob
import json
import os
import random
import re
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Dict, Any

from openai import AsyncOpenAI

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent
BRAILLE_DIR = BASE_DIR / "BrailleData" / "Braille" / "Nemeth" / "highschool"
MATHML_DIR = BASE_DIR / "SimpleSpeakData" / "highschool"
RUST_TEST_DATA_DIR = BASE_DIR / "RustTestData"
RUST_NEMETH_BRLS = RUST_TEST_DATA_DIR / "Nemeth.brls"
RUST_NEMETH_MMLS = RUST_TEST_DATA_DIR / "Nemeth.mmls"
RESULTS_DIR = BASE_DIR / "results"

# Filter parameters
MIN_MATHML_LENGTH = 20
MAX_MATHML_LENGTH = 300

# Sampling parameters
NUM_FEW_SHOT_EXAMPLES = 200
NUM_TEST_PAIRS = 200

# Model configuration
MODEL_NAME = "gpt-5.2-2025-12-11"
REASONING_EFFORT = "high"  # Only used for o-series models
MAX_COMPLETION_TOKENS = 10_000

# Output
OUTPUT_CSV_TEMPLATE = "nemeth_mathml_evaluation_results_{timestamp}.csv"
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N pairs

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================================
# Data Loading Module
# ============================================================================

def scan_highschool_files() -> List[Path]:
    """Scan for all -no-dups.brls files in highschool directory."""
    pattern = str(BRAILLE_DIR / "*-no-dups.brls")
    brls_files = glob.glob(pattern)

    print(f"Found {len(brls_files)} Braille files in highschool directory")
    return [Path(f) for f in brls_files]


def load_paired_expressions(brls_path: Path) -> List[Dict[str, Any]]:
    """
    Load paired Nemeth Braille and MathML expressions from matching files.

    Args:
        brls_path: Path to .brls file

    Returns:
        List of dicts with keys: nemeth, mathml, source_file, line_number
    """
    # Construct corresponding MathML path
    mmls_path = Path(str(brls_path).replace(
        str(BRAILLE_DIR),
        str(MATHML_DIR)
    ).replace(".brls", ".mmls"))

    pairs = []

    with open(brls_path, 'r', encoding='utf-8') as bf, \
         open(mmls_path, 'r', encoding='utf-8') as mf:

        braille_lines = bf.readlines()
        mathml_lines = mf.readlines()

        # Ensure matching line counts
        if len(braille_lines) != len(mathml_lines):
            print(f"Warning: Line count mismatch in {brls_path.name}")
            min_lines = min(len(braille_lines), len(mathml_lines))
            braille_lines = braille_lines[:min_lines]
            mathml_lines = mathml_lines[:min_lines]

        for line_num, (braille, mathml) in enumerate(zip(braille_lines, mathml_lines), 1):
            nemeth = braille.strip()
            mathml_content = mathml.strip()

            # Skip empty lines
            if not nemeth or not mathml_content:
                continue

            pairs.append({
                'nemeth': nemeth,
                'mathml': mathml_content,
                'source_file': brls_path.name,
                'line_number': line_num
            })

    return pairs


def filter_by_size(pairs: List[Dict[str, Any]],
                   min_len: int = MIN_MATHML_LENGTH,
                   max_len: int = MAX_MATHML_LENGTH) -> List[Dict[str, Any]]:
    """Filter pairs by MathML character length."""
    filtered = [p for p in pairs if min_len <= len(p['mathml']) <= max_len]
    return filtered


def load_all_highschool_pairs() -> List[Dict[str, Any]]:
    """Load all filtered pairs from highschool directory."""
    all_pairs = []

    brls_files = scan_highschool_files()

    for brls_path in brls_files:
        pairs = load_paired_expressions(brls_path)
        filtered_pairs = filter_by_size(pairs)
        all_pairs.extend(filtered_pairs)
        print(f"  {brls_path.name}: {len(filtered_pairs)} pairs (after filtering)")

    print(f"\nTotal filtered pairs: {len(all_pairs)}")
    return all_pairs


def load_rust_test_data_pairs(max_pairs: int = 200) -> List[Dict[str, Any]]:
    """
    Load paired Nemeth Braille and MathML expressions from RustTestData directory.

    Args:
        max_pairs: Maximum number of pairs to load (default: 200)

    Returns:
        List of dicts with keys: nemeth, mathml, source_file, line_number
    """
    pairs = []

    print(f"Loading RustTestData pairs from {RUST_NEMETH_BRLS.name}...")

    with open(RUST_NEMETH_BRLS, 'r', encoding='utf-8') as bf, \
         open(RUST_NEMETH_MMLS, 'r', encoding='utf-8') as mf:

        braille_lines = bf.readlines()
        mathml_lines = mf.readlines()

        # Ensure matching line counts
        if len(braille_lines) != len(mathml_lines):
            print(f"Warning: Line count mismatch in RustTestData")
            min_lines = min(len(braille_lines), len(mathml_lines))
            braille_lines = braille_lines[:min_lines]
            mathml_lines = mathml_lines[:min_lines]

        for line_num, (braille, mathml) in enumerate(zip(braille_lines, mathml_lines), 1):
            nemeth = braille.strip()
            mathml_content = mathml.strip()

            # Skip empty lines
            if not nemeth or not mathml_content:
                continue

            pairs.append({
                'nemeth': nemeth,
                'mathml': mathml_content,
                'source_file': RUST_NEMETH_BRLS.name,
                'line_number': line_num
            })

            # Stop if we've reached max_pairs
            if len(pairs) >= max_pairs:
                break

    print(f"Loaded {len(pairs)} pairs from RustTestData")
    return pairs


# ============================================================================
# Few-Shot Sampling Module
# ============================================================================

def sample_few_shot_examples(example_pool: List[Dict[str, Any]],
                             n: int,
                             exclude_pair: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Randomly sample n few-shot examples from the pool.

    Args:
        example_pool: Pool of all available examples
        n: Number of examples to sample
        exclude_pair: Current test pair to exclude from sampling

    Returns:
        List of n sampled examples
    """
    # Create pool excluding current test pair
    if exclude_pair:
        pool = [p for p in example_pool
                if not (p['source_file'] == exclude_pair['source_file'] and
                        p['line_number'] == exclude_pair['line_number'])]
    else:
        pool = example_pool.copy()

    # Sample
    if len(pool) < n:
        print(f"Warning: Pool has only {len(pool)} examples, requested {n}")
        return random.sample(pool, len(pool))

    return random.sample(pool, n)


# ============================================================================
# Prompt Engineering Module
# ============================================================================

def build_system_prompt() -> str:
    """Build the system instruction for the model."""
    return """You are an expert in Nemeth Braille mathematics notation. Your task is to translate Nemeth Braille expressions into valid MathML (Mathematical Markup Language) XML format.

NEMETH BRAILLE ENCODING RULES:

Numbers and Digits:
- ⠼ = Number indicator (starts a numeric sequence)
- After ⠼, the following symbols represent digits:
  ⠁=1, ⠆=2, ⠒=3, ⠲=4, ⠢=5, ⠖=6, ⠶=7, ⠦=8, ⠔=9, ⠴=0
- Number sequences terminate at: space, letter (variable), or operation symbol
- Example: ⠼⠁⠆ = "12" (number 12)
- Example: ⠼⠆⠁ = "2a" (2 times variable a, NOT 21)

Decimal Points:
- ⠨ before/within digits = decimal point
- Example: ⠼⠁⠨⠆ = "1.2"
- Example: ⠼⠴⠨⠴⠦ = "0.08"
- IMPORTANT: ⠨ in number context is ALWAYS a decimal point, never a comma

Operators and Relations:
- ⠨⠅ = equals sign (=)
- ⠐⠅ = less than (<)
- ⠐⠅⠱ = less than or equal to (≤)
- ⠨⠂ = greater than or equal to (≥)
- ⠬ = plus (+)
- ⠤ = minus (−)
- ⠈⠡ = times (×)

Superscripts and Subscripts:
- ⠘ = superscript start
- ⠸ = superscript end (when multi-character or before continuing)
- ⠰ = subscript start
- Example: ⠭⠘⠆ = x² (x squared)
- Example: ⠭⠘⠒⠸⠬⠁ = x³+a (superscript ends before the +)
- Example: ⠭⠰⠂ = x₁ (x subscript 1)

Fractions:
- ⠹ = fraction start
- ⠌ = fraction bar (separates numerator from denominator)
- ⠼ = fraction end
- Example: ⠹⠁⠌⠆⠼ = 1/2 (one half)
- Example: ⠹⠒⠌⠲⠼ = 3/4 (three quarters)

Grouping Symbols:
- ⠷ = open parenthesis (
- ⠾ = close parenthesis )
- ⠨⠷ = open bracket [
- ⠨⠾ = close bracket ]

Variables and Letters:
- Letters use standard braille codes
- When NOT preceded by ⠼, braille letters represent variables
- Example: ⠭ = x, ⠁ = a (as variable), ⠃ = b, etc.

IMPORTANT INSTRUCTIONS:
- Provide ONLY the MathML output
- Do NOT include explanations, comments, or any text besides the MathML
- Ensure the output is valid, well-formed XML
- Include the proper xmlns namespace in the <math> tag
- Preserve mathematical meaning and structure accurately
- Pay special attention to number indicators (⠼) and decimal points (⠨)
- When you see ⠨ within a number context, it is ALWAYS a decimal point"""


def format_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    """Format few-shot examples as demonstration text."""
    formatted = "EXAMPLES:\n\n"

    for i, ex in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"Input (Nemeth Braille): {ex['nemeth']}\n"
        formatted += f"Output (MathML): {ex['mathml']}\n\n"

    return formatted


def build_full_prompt(nemeth_input: str, few_shot_examples: List[Dict[str, Any]]) -> str:
    """Build the complete prompt for translation."""
    system = build_system_prompt()
    examples = format_few_shot_examples(few_shot_examples)

    query = f"""NOW TRANSLATE THE FOLLOWING:

Input (Nemeth Braille): {nemeth_input}
Output (MathML):"""

    full_prompt = f"{system}\n\n{examples}{query}"
    return full_prompt


# ============================================================================
# API Integration Module
# ============================================================================


async def call_openai_model(client: AsyncOpenAI, prompt: str) -> Dict[str, Any]:
    """
    Call OpenAI model with the given prompt.

    Returns dict with:
        - predicted_mathml: The model's output
        - prompt_tokens, completion_tokens, total_tokens
        - reasoning_tokens (if available)
        - response_time_seconds
        - model_name
        - reasoning_effort: The reasoning effort used (if applicable)
        - error: None if successful, error message otherwise
    """
    start_time = time.time()

    # Build API call parameters
    api_params = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": MAX_COMPLETION_TOKENS
    }

    # Add reasoning_effort for o-series models
    if MODEL_NAME.startswith("o"):
        api_params["reasoning_effort"] = REASONING_EFFORT

    response = await client.chat.completions.create(**api_params)

    response_time = time.time() - start_time

    # Extract predicted MathML
    predicted = response.choices[0].message.content.strip()

    # Extract token usage
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Try to get reasoning tokens (may not be available for all models)
    reasoning_tokens = getattr(usage, 'reasoning_tokens', None)

    return {
        'predicted_mathml': predicted,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'reasoning_tokens': reasoning_tokens,
        'response_time_seconds': round(response_time, 2),
        'model_name': response.model,
        'reasoning_effort': REASONING_EFFORT if MODEL_NAME.startswith("o") else None,
        'error': None
    }




def extract_mathml_from_response(response_text: str) -> str:
    """
    Extract MathML from model response, handling cases where
    the model might include extra text.
    """
    # Try to find <math>...</math> tags
    match = re.search(r'<math[^>]*>.*?</math>', response_text, re.DOTALL)

    if match:
        return match.group(0)

    # If no match, return the whole response (it might be valid MathML)
    return response_text.strip()


# ============================================================================
# Evaluation Loop
# ============================================================================

async def process_single_pair(client: AsyncOpenAI,
                               test_pair: Dict[str, Any],
                               example_pool: List[Dict[str, Any]],
                               pair_id: int) -> Dict[str, Any]:
    """
    Process a single test pair.

    Args:
        client: AsyncOpenAI client
        test_pair: The pair to evaluate
        example_pool: Pool for sampling few-shot examples
        pair_id: ID for this pair in results

    Returns:
        Result dictionary
    """
    # Sample few-shot examples (excluding current test pair)
    few_shot_examples = sample_few_shot_examples(
        example_pool,
        NUM_FEW_SHOT_EXAMPLES,
        exclude_pair=test_pair
    )

    # Build prompt
    prompt = build_full_prompt(test_pair['nemeth'], few_shot_examples)

    # Call API
    api_result = await call_openai_model(client, prompt)

    # Extract clean MathML
    predicted_mathml = extract_mathml_from_response(api_result['predicted_mathml'])

    # Compile result
    result = {
        'id': pair_id,
        'source_file': test_pair['source_file'],
        'line_number': test_pair['line_number'],
        'nemeth_braille': test_pair['nemeth'],
        'nemeth_length': len(test_pair['nemeth']),
        'ground_truth_mathml': test_pair['mathml'],
        'ground_truth_length': len(test_pair['mathml']),
        'predicted_mathml': predicted_mathml,
        'prompt': prompt,
        'model_name': api_result['model_name'],
        'timestamp': datetime.now(UTC).isoformat(),
        'prompt_tokens': api_result['prompt_tokens'],
        'completion_tokens': api_result['completion_tokens'],
        'total_tokens': api_result['total_tokens'],
        'reasoning_tokens': api_result['reasoning_tokens'],
        'reasoning_effort': api_result['reasoning_effort'],
        'response_time_seconds': api_result['response_time_seconds'],
        'few_shot_example_indices': json.dumps([
            f"{ex['source_file']}:{ex['line_number']}"
            for ex in few_shot_examples
        ]),
        'error': api_result['error']
    }

    return result


async def run_evaluation(client: AsyncOpenAI,
                         test_pairs: List[Dict[str, Any]],
                         example_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run evaluation on test pairs with few-shot prompting as a single batch.

    Args:
        client: AsyncOpenAI client
        test_pairs: Pairs to evaluate
        example_pool: Pool for sampling few-shot examples

    Returns:
        List of result dictionaries
    """
    print(f"\nStarting evaluation of {len(test_pairs)} pairs...")
    print(f"Using {NUM_FEW_SHOT_EXAMPLES} few-shot examples per prompt")

    tasks = [
        process_single_pair(client, test_pair, example_pool, i)
        for i, test_pair in enumerate(test_pairs, 1)
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    print(f"\nEvaluation complete! Processed {len(results)} pairs.")
    return list(results)


# ============================================================================
# CSV Export Module
# ============================================================================

def write_results_csv(results: List[Dict[str, Any]], output_path: Path):
    """Write results to CSV file."""
    if not results:
        print("Warning: No results to write")
        return

    fieldnames = [
        'id', 'source_file', 'line_number',
        'nemeth_braille', 'nemeth_length',
        'ground_truth_mathml', 'ground_truth_length',
        'predicted_mathml',
        'prompt',
        'model_name', 'timestamp',
        'prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens', 'reasoning_effort',
        'response_time_seconds',
        'few_shot_example_indices',
        'error'
    ]


    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")
    print(f"Total rows: {len(results)}")



# ============================================================================
# Main
# ============================================================================

async def main():
    """Main execution function."""
    print("=" * 80)
    print("Nemeth Braille to MathML Translation Evaluation")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Filter: MathML length {MIN_MATHML_LENGTH}-{MAX_MATHML_LENGTH} characters")
    print(f"Few-shot examples: {NUM_FEW_SHOT_EXAMPLES} (from RustTestData)")
    print(f"Test pairs: {NUM_TEST_PAIRS} (from highschool directory)")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)
    print()

    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(exist_ok=True)

    # Generate output CSV path in results folder
    output_csv = RESULTS_DIR / OUTPUT_CSV_TEMPLATE.format(timestamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    client = AsyncOpenAI(api_key=(os.getenv("OPENAI_API_KEY")))

    # Load few-shot examples from RustTestData
    few_shot_pool = load_rust_test_data_pairs(NUM_FEW_SHOT_EXAMPLES)

    # Load test pairs from highschool directory
    all_highschool_pairs = load_all_highschool_pairs()
    test_pairs = random.sample(all_highschool_pairs, NUM_TEST_PAIRS)

    print("Running evaluation...")
    results = await run_evaluation(client, test_pairs, few_shot_pool)

    write_results_csv(results, output_csv)

    # 6. Summary statistics
    print("SUMMARY")

    successful = [r for r in results if r['error'] is None]
    failed = [r for r in results if r['error'] is not None]

    print(f"Total pairs processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_prompt_tokens = sum(r['prompt_tokens'] for r in successful) / len(successful)
        avg_completion_tokens = sum(r['completion_tokens'] for r in successful) / len(successful)
        total_tokens = sum(r['total_tokens'] for r in successful)
        avg_response_time = sum(r['response_time_seconds'] for r in successful) / len(successful)

        print(f"\nToken Usage:")
        print(f"  Average prompt tokens: {avg_prompt_tokens:.0f}")
        print(f"  Average completion tokens: {avg_completion_tokens:.0f}")
        print(f"  Total tokens used: {total_tokens:,}")
        print(f"\nPerformance:")
        print(f"  Average response time: {avg_response_time:.2f}s")

    print(f"\nOutput file: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
