"""
Utility functions for analyzing braille/MathML translation accuracy.

This module provides functions for:
- Parsing result files from Gemini translation runs
- Computing length and complexity metrics
- Generating stratified accuracy statistics
- Extracting features from MathML
"""

import re
from typing import NamedTuple
import pandas as pd
from xml.etree import ElementTree as ET


class TranslationResult(NamedTuple):
    """Single translation result from a test file."""
    is_correct: bool
    input_text: str
    expected: str
    computed: str


def parse_result_file(filepath: str, use_normalized: bool = False) -> list[TranslationResult]:
    """
    Parse a result file and extract translation results.

    Args:
        filepath: Path to the result .txt file
        use_normalized: If True, read normalized MathML section (second section).
                       If False, read non-normalized section (first section).

    Returns:
        List of TranslationResult objects
    """
    results = []
    in_target_section = False
    sections_seen = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Detect section headers
            if line.startswith('# Match | Test Input') or line.startswith('Match | Test Input'):
                sections_seen += 1
                # For normalized, we want section 2; for non-normalized, section 1
                target_section = 2 if use_normalized else 1
                in_target_section = (sections_seen == target_section)
                continue

            # Stop if we've passed our target section
            if sections_seen > (2 if use_normalized else 1):
                break

            # Skip comments and empty lines
            if not in_target_section or line.startswith('#') or not line:
                continue

            # Parse result line: ✓/✗ | input | expected | computed
            if line.startswith('✓') or line.startswith('✗'):
                parts = line.split(' | ')
                if len(parts) == 4:
                    is_correct = parts[0] == '✓'
                    input_text = parts[1]
                    expected = parts[2]
                    computed = parts[3]

                    results.append(TranslationResult(
                        is_correct=is_correct,
                        input_text=input_text,
                        expected=expected,
                        computed=computed
                    ))

    return results


def count_braille_chars(text: str) -> int:
    """
    Count Unicode braille characters in text.

    Braille characters are in the range U+2800 to U+28FF.
    """
    return sum(1 for char in text if '\u2800' <= char <= '\u28ff')


def count_mathml_elements(mathml: str) -> int:
    """
    Count the number of MathML elements in an expression.

    Args:
        mathml: MathML string

    Returns:
        Number of XML elements, or 0 if parsing fails
    """
    try:
        root = ET.fromstring(mathml)
        return len(list(root.iter()))
    except:
        # If parsing fails, count tags manually
        return len(re.findall(r'<[^/][^>]*>', mathml))


def get_mathml_nesting_depth(mathml: str) -> int:
    """
    Calculate the maximum nesting depth of MathML elements.

    Args:
        mathml: MathML string

    Returns:
        Maximum nesting depth, or 0 if parsing fails
    """
    try:
        root = ET.fromstring(mathml)

        def depth(elem):
            if len(elem) == 0:
                return 1
            return 1 + max(depth(child) for child in elem)

        return depth(root)
    except:
        return 0


def extract_mathml_features(mathml: str) -> dict:
    """
    Extract features from a MathML expression.

    Args:
        mathml: MathML string

    Returns:
        Dictionary with features including:
        - has_frac: contains fraction
        - has_sup: contains superscript
        - has_sub: contains subscript
        - has_sqrt: contains square root
        - has_matrix: contains matrix/table
        - num_elements: number of MathML elements
        - nesting_depth: maximum nesting depth
    """
    features = {
        'has_frac': '<mfrac' in mathml,
        'has_sup': '<msup' in mathml,
        'has_sub': '<msub' in mathml,
        'has_sqrt': '<msqrt' in mathml or '<mroot' in mathml,
        'has_matrix': '<mtable' in mathml or '<mtr' in mathml,
        'num_elements': count_mathml_elements(mathml),
        'nesting_depth': get_mathml_nesting_depth(mathml),
    }

    return features


def create_dataframe_from_results(results: list[TranslationResult],
                                   input_is_braille: bool) -> pd.DataFrame:
    """
    Convert list of TranslationResults to a pandas DataFrame with computed metrics.

    Args:
        results: List of TranslationResult objects
        input_is_braille: True if input is braille, False if input is MathML

    Returns:
        DataFrame with columns for all metrics
    """
    data = []

    for result in results:
        row = {
            'is_correct': result.is_correct,
            'input': result.input_text,
            'expected': result.expected,
            'computed': result.computed,
        }

        # Compute lengths
        row['input_length'] = len(result.input_text)
        row['expected_length'] = len(result.expected)
        row['computed_length'] = len(result.computed)

        if input_is_braille:
            # Input is braille, output is MathML
            row['braille_length'] = count_braille_chars(result.input_text)
            row['expected_mathml_elements'] = count_mathml_elements(result.expected)
            row['computed_mathml_elements'] = count_mathml_elements(result.computed)

            # Extract MathML features from expected output
            expected_features = extract_mathml_features(result.expected)
            for key, val in expected_features.items():
                row[f'expected_{key}'] = val

        else:
            # Input is MathML, output is braille
            row['mathml_elements'] = count_mathml_elements(result.input_text)
            row['expected_braille_length'] = count_braille_chars(result.expected)
            row['computed_braille_length'] = count_braille_chars(result.computed)

            # Extract MathML features from input
            input_features = extract_mathml_features(result.input_text)
            for key, val in input_features.items():
                row[f'input_{key}'] = val

        data.append(row)

    return pd.DataFrame(data)


def create_stratified_bins(df: pd.DataFrame, column: str,
                           num_bins: int = 4,
                           strategy: str = 'quantile') -> pd.Series:
    """
    Create stratified bins for a numeric column.

    Args:
        df: DataFrame
        column: Column name to bin
        num_bins: Number of bins to create
        strategy: 'quantile' for equal-sized bins, 'fixed' for equal-width bins

    Returns:
        Series with bin labels
    """
    if strategy == 'quantile':
        return pd.qcut(df[column], q=num_bins, duplicates='drop',
                      labels=[f'Q{i+1}' for i in range(num_bins)])
    else:  # fixed width
        return pd.cut(df[column], bins=num_bins, duplicates='drop',
                     include_lowest=True)


def compute_accuracy_by_bins(df: pd.DataFrame, column: str,
                             bins: list = None, bin_column_name: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute accuracy statistics grouped by bins.

    Args:
        df: DataFrame with 'is_correct' column
        column: Column to group by
        bins: Optional list of bin edges. If None, uses quantiles.
        bin_column_name: Optional name for the bin column. If None, uses '{column}_bin'

    Returns:
        Tuple of (accuracy_stats_df, bin_series)
        - accuracy_stats_df: DataFrame with bin ranges, counts, and accuracy percentages
        - bin_series: Series with bin labels for each row (can be added to df)
    """
    # Create bins without modifying original df
    if bins is None:
        # Create quartile bins
        bin_series = pd.qcut(df[column], q=4, duplicates='drop')
    else:
        bin_series = pd.cut(df[column], bins=bins, include_lowest=True)

    # Compute accuracy by bin
    grouped = df.groupby(bin_series, observed=True).agg({
        'is_correct': ['sum', 'count', 'mean']
    }).round(4)

    grouped.columns = ['correct', 'total', 'accuracy']
    grouped['accuracy_pct'] = (grouped['accuracy'] * 100).round(2)

    result = grouped.reset_index()
    result.columns = [bin_column_name or f'{column}_bin', 'correct', 'total', 'accuracy', 'accuracy_pct']

    return result, bin_series


def compute_accuracy_by_feature(df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
    """
    Compute accuracy grouped by a boolean feature.

    Args:
        df: DataFrame with 'is_correct' column
        feature_column: Boolean feature column name

    Returns:
        DataFrame with feature values and accuracy stats
    """
    grouped = df.groupby(feature_column).agg({
        'is_correct': ['sum', 'count', 'mean']
    }).round(4)

    grouped.columns = ['correct_count', 'total_count', 'accuracy']
    grouped['accuracy_pct'] = (grouped['accuracy'] * 100).round(2)

    return grouped.reset_index()


def print_summary_statistics(df: pd.DataFrame, title: str = "Summary Statistics"):
    """
    Print summary statistics for the dataset.

    Args:
        df: DataFrame with analysis results
        title: Title for the summary
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")

    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"Total translations: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Overall accuracy: {accuracy:.2f}%")

    print(f"\n{'Length Statistics':^60}")
    print("-" * 60)
    print(df[['input_length', 'expected_length', 'computed_length']].describe())
