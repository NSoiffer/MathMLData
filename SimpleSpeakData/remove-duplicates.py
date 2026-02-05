import pathlib
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import sys
from pathlib import Path

# Add parent directory to sys.path so libmathcat can be imported
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from compare_mathml_in_csv import setMathCATPreferences, setMathMLForMathCAT  # noqa: E402


def print_usage():
    """Displays instructions on how to run the script."""
    usage = """
Python MMLS Deduplicator
------------------------
Usage:
    python script_name.py [directory_path]

Arguments:
    directory_path    The path to the folder containing .mmls files.
    --help            Show this help message.

Description:
    Recursively finds all .mmls files, removes duplicate lines (ignoring
    trailing whitespace), and saves a new file with the '-no-dups' suffix.
    """
    print(usage)


def make_output_filename(path: Path, canonicalize: bool) -> Path:
    stem = path.stem  # "foo"
    suffix = path.suffix  # ".mmls"

    new_stem = f"{stem}-no-dups"
    if canonicalize:
        new_stem += "-cnclz"

    return path.with_name(new_stem + suffix)


def process_mmls_file(args: tuple[Path, bool, bool]) -> tuple[bool, int, int]:
    file_path, canonicalize, dry_run = args

    try:
        setMathCATPreferences({}, dir="..")
    except Exception as e:
        print(f"problem with finding the MathCAT rules: {e}")
        return False, 0, 0

    try:
        output_path = make_output_filename(file_path, canonicalize)

        original_count = 0
        written_count = 0
        mathml_encountered = set()
        unique_lines = []

        with open(file_path, "r", encoding="utf-8") as in_stream:
            for line in in_stream:
                line = line.rstrip('\n\r').strip()
                if not line:
                    continue
                original_count += 1

                try:
                    canonicalized = setMathMLForMathCAT(line)
                except Exception as e:
                    print(f"Error canonicalizing MathML in {file_path}: {e}")
                    break

                if canonicalized not in mathml_encountered:
                    mathml_encountered.add(canonicalized)
                    unique_lines.append(canonicalized if canonicalize else line)
                    written_count += 1

        if dry_run:
            print(f"\nDry run: {output_path} would have been created with {written_count} lines")
            return True, original_count, written_count
        else:
            with open(output_path, "w", encoding="utf-8") as out_stream:
                for line in unique_lines:
                    out_stream.write(line + "\n")
            return True, original_count, written_count
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False, 0, 0


def is_original_mmls(path: pathlib.Path) -> bool:
    """ use only the base .mmls files -- not the -cnclz or -no-dups files """
    stem = path.stem
    return (
        stem.endswith(".mmls") or True
    ) and "-cnclz" not in stem and "-no-dups" not in stem


def process_mmls_files(root_dir: str, num_procs: int, canonicalize: bool, dry_run: bool):
    path = pathlib.Path(root_dir)

    if not path.is_dir():
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    files = [f for f in path.rglob("*.mmls") if f.is_file() and is_original_mmls(f)]
    if not files:
        print("No .mmls files found in the specified directory.")
        return

    stats = {
        "files_created": 0,
        "errors": 0,
        "total_original": 0,
        "total_written": 0,
    }

    work_items = [(f, canonicalize, dry_run) for f in files]

    pool = Pool(processes=num_procs)
    try:
        for success, orig, written in tqdm(
            pool.imap_unordered(process_mmls_file, work_items),
            total=len(files),
            desc="Processing",
            unit="file"
        ):
            if success:
                stats["files_created"] += 1
                stats["total_original"] += orig
                stats["total_written"] += written
            else:
                stats["errors"] += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating workers...")

        # *** THIS IS THE IMPORTANT PART ***
        pool.terminate()   # kill workers immediately
        pool.join()        # wait for them to exit

        print("Exited cleanly.")
        return

    else:
        pool.close()
        pool.join()

    print_summary(stats)


def print_summary(stats):
    print("\n" + "="*35)
    print(f"{'PROCESS COMPLETE':^35}")
    print("="*35)
    print(f"New Files Created:  {stats['files_created']}")
    print(f"Total Original Lines: {stats['total_original']}")
    print(f"Total Written Lines:  {stats['total_written']}")
    print(f"Total Lines Removed:  {stats['total_original'] - stats['total_written']}")
    if stats["errors"] > 0:
        print(f"Errors encountered:  {stats['errors']}")
    print("="*35)


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in ("true", "yes", "y"):
        return True
    if value in ("false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(
        "Expected one of: true, false, yes, no, y, n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate .mmls files")
    parser.add_argument("directory", help="Directory containing .mmls files")
    parser.add_argument(
        "-r", "--run",
        type=int,
        default=8,
        help="Number of parallel processes to use (default: 8)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files but do not write output files"
    )
    parser.add_argument(
        "-c", "--canonicalize",
        type=parse_bool,
        help="Canonicalize MathML before processing (true/false/yes/no/y/n)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        process_mmls_files(args.directory, args.run, args.canonicalize, args.dry_run)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting cleanly.")
        sys.exit(1)
