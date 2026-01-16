import sys
from typing import List
from compare_mathml_in_csv import setMathCATPreferences, areCanonicallyEqual

sys.stdout.reconfigure(encoding='utf-8')


def compare_files(file_paths: List[str]) -> None:
    """
    Compares two files and prints line-by-line results plus summary statistics.
    """
    if len(file_paths) < 2:
        print("Usage: python compare_mathML_files.py <file1> <file2>")
        return

    # initial MathCAT
    setMathCATPreferences({})

    path1, path2 = file_paths[0], file_paths[1]

    # Initialize statistics counters
    match_count: int = 0
    total_lines: int = 0

    try:
        with open(path1, 'r', encoding='utf-8') as f1, \
             open(path2, 'r', encoding='utf-8') as f2:

            for line1, line2 in zip(f1, f2):
                total_lines += 1
                try:
                    if areCanonicallyEqual(line1, line2):
                        match_count += 1
                    else:
                        print(f"Line {total_lines}: Different")
                except Exception as e:
                    print(f"Error comparing lines: {e}")
                    continue

        # Calculate and display statistics
        if total_lines > 0:
            match_percentage: float = (match_count / total_lines) * 100
            print("\n--- Statistics ---")
            print(f"Total Lines Compared: {total_lines}")
            print(f"Matching Lines:      {match_count}")
            print(f"Mismatching Lines:   {total_lines - match_count}")
            print(f"Match Percentage:    {match_percentage:.2f}%")
        else:
            print("\nNo lines were compared (one or both files are empty).")

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
    except UnicodeDecodeError:
        print("Error: Files must be encoded in UTF-8.")


if __name__ == "__main__":
    compare_files(sys.argv[1:])
