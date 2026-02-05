import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple
from compare_mathml_in_csv import setMathCATPreferences, setMathMLForMathCAT

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


class ErrorInfo(NamedTuple):
    """Information about a canonicalization error."""
    line_num: int
    mathml: str
    error_message: str


def canonicalize_mmls_file(input_file: Path, output_file: Path) -> tuple[int, list[ErrorInfo]]:
    """
    Canonicalize all MathML expressions in an .mmls file.

    Args:
        input_file: Path to the input .mmls file
        output_file: Path to the output .cnclz.mmls file

    Returns:
        Tuple of (processed_count, error_list) where error_list contains ErrorInfo objects
    """
    # Initialize MathCAT for this thread
    try:
        setMathCATPreferences({})
    except Exception as e:
        print(f"Warning: Can't set MathCAT preferences in thread for {input_file.name}: {e}")

    processed_count = 0
    error_list = []

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            mathml = line.rstrip('\n\r')

            # Skip empty lines (write blank line without whitespace)
            if not mathml.strip():
                outfile.write('\n')
                continue

            try:
                canonical = setMathMLForMathCAT(mathml)
                canonical_clean = " ".join(canonical.split()).strip()
                outfile.write(f"{canonical_clean}\n")
                processed_count += 1
            except Exception as e:
                error_message = str(e)
                error_list.append(ErrorInfo(line_num, mathml, error_message))
                # Write blank line without whitespace to keep line count aligned
                outfile.write('\n')

    return processed_count, error_list


def process_single_file(mmls_file: Path) -> tuple[str, int, list[ErrorInfo], int]:
    """
    Process a single .mmls file and return results.

    Args:
        mmls_file: Path to the .mmls file to process

    Returns:
        Tuple of (filename, processed_count, error_list, total_lines) where error_list contains ErrorInfo objects
    """
    # Create output filename: insert -cnclz before .mmls extension
    output_file = mmls_file.with_stem(mmls_file.stem + '-cnclz')

    # Count lines in input file
    with open(mmls_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    processed, error_list = canonicalize_mmls_file(mmls_file, output_file)
    return mmls_file.name, processed, error_list, total_lines


def canonicalize_directory(directory: str, num_processes: int = 8) -> None:
    """
    Canonicalize all .mmls files in a directory using multiple processes.

    Args:
        directory: Path to the directory containing .mmls files
        num_processes: Number of processes to use for parallel processing
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory")
        sys.exit(1)

    # Find all .mmls files, excluding those ending with cnclz.mmls
    mmls_files = [f for f in dir_path.glob("*.mmls") if not f.name.endswith("cnclz.mmls")]

    if not mmls_files:
        print(f"No .mmls files found in '{directory}' (excluding cnclz.mmls files)")
        return

    print(f"Found {len(mmls_files)} .mmls file(s) in '{directory}'")
    print(f"Using {num_processes} process(es) for processing")
    print()

    total_processed = 0
    all_errors = []  # List of tuples: (filename, ErrorInfo)

    # Process files in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, mmls_file): mmls_file
                          for mmls_file in sorted(mmls_files)}

        # Process completed tasks as they finish
        for future in as_completed(future_to_file):
            try:
                filename, processed, error_list, total_lines = future.result()
                print(f"Processing {filename}...")
                print(f"  -> {filename.replace('.mmls', '-cnclz.mmls')}: {processed}/{total_lines} lines processed")
                total_processed += processed
                # Collect errors with filename
                for error_info in error_list:
                    all_errors.append((filename, error_info))
            except Exception as e:
                mmls_file = future_to_file[future]
                print(f"Error processing {mmls_file.name}: {e}")
                all_errors.append((mmls_file.name, ErrorInfo(0, "", str(e))))

    # Write errors to errors.log
    errors_log_path = dir_path / "errors.log"
    if all_errors:
        with open(errors_log_path, 'w', encoding='utf-8') as error_file:
            for filename, error_info in all_errors:
                error_file.write(f"File: {filename}\n")
                error_file.write(f"Line: {error_info.line_num}\n")
                error_file.write(f"Input: {error_info.mathml}\n")
                error_file.write(f"Error: {error_info.error_message}\n")
                error_file.write("\n")
        print(f"Errors written to {errors_log_path}")

    print()
    print(
        f"Completed: {len(mmls_files)} file(s) processed, "
        f"{total_processed} MathML expressions canonicalized, "
        f"{len(all_errors)} errors"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python canonicalize_mmls_directory.py <directory> [num_processes]")
        print("  directory: Path to directory containing .mmls files")
        print("  num_processes: Number of processes to use (default: 8)")
        sys.exit(1)

    directory = sys.argv[1]
    num_processes = int(sys.argv[2]) if len(sys.argv) == 3 else 8

    if num_processes < 1:
        print("Error: num_processes must be at least 1")
        sys.exit(1)

    canonicalize_directory(directory, num_processes)
