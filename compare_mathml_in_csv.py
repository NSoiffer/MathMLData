"""
Usage: python script_name.py <input_file.csv>

Description:
    This script processes a CSV file containing MathML data. It compares
    'ground_truth_mathml' and 'predicted_mathml' columns using a canonical
    equality function and inserts the boolean result as a new column
    'is_equal' immediately before the ground truth column.

Output:
    Saves a new file named '<original_name>_processed.csv'.
"""

import pandas as pd
import sys
import os
import libmathcat_py as libmathcat
from bs4 import BeautifulSoup   # pip install BeautifulSoup4

# Attributes to ignore during comparison (might need a few more)
IGNORE_ATTRS = ['id', 'displaystyle', 'scriptlevel', 'xmlns',
                'data-id-added', 'data-added', 'data-changed',
                'data-previous-space-width', 'data-following-space-width',
                'data-empty-in-2d', 'data-width', 'data-function-guess',
                'stretchy',
                'columnspacing', 'rowspacing', 'rowlines', 'columnlines','minlabelspacing',
                'columnalign', 'rowalign', 'equalcolumns', 'equalrows',
                'mathcolor', 'mathbackground', 'mathsize', 'mathvariant']


def setMathCATPreferences(prefs: dict[str, str]):
    """
    Initial MathCAT by setting the rules directory and any preferences.
    Do this once outside of a loop
    """
    try:
        libmathcat.SetRulesDir("MathCATRules")
    except Exception as e:
        sys.exit(f"problem with finding the MathCAT rules: {e}")

    try:
        for key, value in prefs.items():
            libmathcat.SetPreference(key, value)
    except Exception as e:
        sys.exit(f"problem with setting a preference: {e}")


def setMathMLForMathCAT(mathml: str):
    """
    Sets the MathML and returns a canonical form.
    """
    try:
        canonicalMathML = libmathcat.SetMathML(mathml)
        return strip_mathml_attributes(canonicalMathML, IGNORE_ATTRS)
    except Exception as e:
        raise e


def strip_mathml_attributes(mathml_string: str, attributes_to_remove: list[str]) -> str:
    """
    Removes a list of attributes and their values from a MathML string.
    Also removes all MathJax attrs (those starting with 'data-mjx-').

    Args:
        mathml_string: The raw MathML string.
        attributes_to_remove: A list of attribute names (e.g., ['display', 'xmlns']).

    Returns:
        The cleaned MathML string as a single line.
    """
    # Use 'xml' if you have lxml installed for better MathML precision,
    # otherwise 'html.parser' works well for general stripping.
    soup = BeautifulSoup(mathml_string, 'html.parser')
    attributes_to_remove += [attr for attr in soup.find_all(True)[0].attrs.keys() if attr.startswith('data-mjx-')]

    for tag in soup.find_all(True):
        for attr in [a for a in tag.attrs if a.startswith("data-mjx") or a in attributes_to_remove]:
            del tag[attr]

    # Clean up whitespace and return as a single line
    return " ".join(str(soup).split())


failure_count: int = 0  # Global counter for match failures


def areCanonicallyEqual(original: str, computed: str) -> bool:
    """
    Placeholder for your canonical comparison logic.
    Returns True if the MathML strings are equivalent, False otherwise.
    """
    global failure_count
    if type(computed) is not str:
        failure_count += 1
        return False

    # Remove these after fixing the input data
    cannonicalOriginal: str = libmathcat.SetMathML(original)
    cannonicalOriginal = strip_mathml_attributes(cannonicalOriginal, IGNORE_ATTRS)
    canonicalComputed: str = libmathcat.SetMathML(computed)
    canonicalComputed = strip_mathml_attributes(canonicalComputed, IGNORE_ATTRS)
    result = cannonicalOriginal.strip() == canonicalComputed.strip()
    if not result:
        failure_count += 1
        print(f"\nNot the same:\nOriginal: {cannonicalOriginal}")
        print(f"Predicted: {canonicalComputed}")
    return result


def process_mathml_csv(input_file: str) -> None:
    # Verify file existence and extension
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    if not input_file.lower().endswith('.csv'):
        print("Error: Please provide a valid CSV file.")
        return

    # initial MathCAT
    setMathCATPreferences({})

    # Load the CSV
    try:
        # Explicitly typing the DataFrame
        df: pd.DataFrame = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check for required columns
    required_cols: list[str] = ['ground_truth_mathml', 'predicted_mathml']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return

    # Apply the comparison function to each row
    # results is a pandas Series of booleans
    results: pd.Series = df.apply(
        lambda row: areCanonicallyEqual(row['ground_truth_mathml'], row['predicted_mathml']),
        axis=1
    )

    # Find the index of 'ground_truth_mathml' to insert before it
    gt_index: int = df.columns.get_loc('ground_truth_mathml')

    # Insert the result column (is_equal) at the specific index
    df.insert(loc=gt_index, column='is_equal', value=results)

    # Generate output filename
    base: str
    ext: str
    base, ext = os.path.splitext(input_file)
    output_file: str = f"{base}_processed{ext}"

    # Save the updated CSV
    df.to_csv(output_file, index=False)
    print(f"Processed {len(df)} rows; {failure_count} failures ({failure_count/len(df)*100:.2f}%)."
          f" Saved to: {output_file}")


if __name__ == "__main__":
    # Check if exactly one filename argument was provided
    if len(sys.argv) != 2:
        print("Usage: python compare_mathML_files.py <filename.csv>")
    else:
        file_path: str = sys.argv[1]
        process_mathml_csv(file_path)
