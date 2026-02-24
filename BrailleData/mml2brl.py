"""
Batch process MathML files to generate braille output using MathCAT.
The "-no-dups.mmls" files from source directories are processed.
 python mml2brl.py \
     --code Nemeth UEB \
     --dir ../SimpleSpeakData/highschool ../SimpleSpeakData/college \
"""
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from typing import cast
import sys
import argparse

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
import libmathcat_py as libmathcat  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

logging.basicConfig(filename='mml2brl.log', level=logging.ERROR)

debug_logger = logging.getLogger("mml2brl")
debug_handler = logging.FileHandler("mml2brl.log", encoding="utf-8")
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.ERROR)


def setMathCATPreferences(braille_code, dir: str = "."):
    try:
        libmathcat.SetRulesDir(f"{dir}/MathCATRules")
    except Exception as e:
        sys.exit(f"problem with finding the MathCAT rules: {e}")

    try:
        libmathcat.SetPreference("BrailleNavHighlight", "Off")
        libmathcat.SetPreference("BrailleCode", braille_code)
    except Exception as e:
        sys.exit(f"problem with setting a preference: {e}")


def setMathMLForMathCAT(mathml: str):
    try:
        libmathcat.SetMathML(mathml)
    except Exception as e:
        raise e


def getSpeech():
    try:
        return libmathcat.GetSpokenText()
    except Exception as e:
        raise e


def getBraille():
    try:
        return libmathcat.GetBraille("")
    except Exception as e:
        raise e


def make_output_filename(path: Path, dest_dir: str, brailleCode: str) -> Path:
    # The parent directory name of the input file
    subdir, _ = path.parts[-2:]

    out_path = Path(dest_dir) / brailleCode / subdir / (path.stem + ".brls")

    # Ensure all directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    return out_path


def ProcessFile(file_name: str, dest_dir: str, config: dict[str, str | bool]):
    """
    Read all the MathML lines from file_path, convert to braille, and write the braille to dest_folder

    """
    file_path = Path(file_name)
    brailleCode = cast(str, config["BrailleCode"])
    out_path = make_output_filename(file_path, dest_dir, brailleCode)
    if config.get("dry_run", False):
        print(f"[DRY RUN] {file_path} -> {out_path}")
        return

    try:
        setMathCATPreferences(brailleCode, dir="..")
    except Exception as e:
        print(f"Can't set rules dir/preference: {e}")
    try:
        with open(file_path, 'r', encoding='utf8') as in_stream, \
             open(out_path, 'w', encoding='utf8') as out_stream:
            for i, line in enumerate(in_stream.readlines()):
                try:
                    setMathMLForMathCAT(line)
                    braille = getBraille()
                    # technically, the braille spaces are probably correct to be there, but this normalizes things
                    braille = braille.strip("⠀")
                    out_stream.write(braille)
                    out_stream.write("\n")
                except Exception as e:
                    print(f"Error in {file_path} -> {brailleCode} on line {i+1}: see mml2brl.log for details")
                    out_stream.write("⠀\n")   # write something to the output file to keep the line count aligned
                    debug_logger.error(
                        f"File: {file_path} -> {brailleCode} on line {i+1}\nMathML:\n{line}\nError: {e}\n{'-'*60}"
                    )
                    # Continue processing the rest of the file
                    continue
    except Exception as e:
        raise e


def ProcessFileList(file_paths: list[str], dest_dir: str,
                    config: dict[str, str | bool], max_workers: int):

    total_files = len(file_paths)
    successes = 0
    failures = 0
    written = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(ProcessFile, path, dest_dir, config): path
            for path in file_paths
        }

        with tqdm(total=total_files, desc="Batch Processing") as pbar:
            for future in as_completed(future_to_file):
                try:
                    output_path = future.result()
                    successes += 1
                    if output_path:
                        written += 1
                except Exception as exc:
                    failures += 1
                    logging.error(f"Error on {future_to_file[future]}: {exc}")
                pbar.update(1)

    return {
        "total_files": total_files,
        "successes": successes,
        "failures": failures,
        "written": written,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process MathML files into braille output"
    )

    # Mutually exclusive: either --files OR --dir
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--files",
        nargs="+",
        help="Explicit list of MathML files to process"
    )

    group.add_argument(
        "--dir",
        nargs="+",
        help="One or more source directories to process"
    )

    parser.add_argument(
        "--code",
        nargs="+",
        required=True,
        help="One or more braille codes (e.g., Nemeth UEB)"
    )

    parser.add_argument(
        "--use-dups",
        action="store_true",
        help="Enable use-dups mode (default: off)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files but do not write output files"
    )

    return parser.parse_args()


def print_summary(summary):
    print("\n==================== SUMMARY ====================")
    for entry in summary:
        s = entry["stats"]
        print(f"\nDirectory:     {entry['dir']}")
        print(f"Braille code:  {entry['code']}")
        print(f"Dry run:       {entry['dry_run']}")
        print(f"Files:         {s['total_files']}")
        print(f"Successes:     {s['successes']}")
        print(f"Failures:      {s['failures']}")
        print(f"Output files:  {s['written']}")
    print("\n=================================================\n")


def main():
    args = parse_args()
    dest_dir = "Braille"

    # Determine file list based on mode
    if args.files:
        # Explicit file mode
        file_list = []
        for f in args.files:
            if not os.path.isfile(f):
                sys.exit(f"Error: file does not exist: {f}")
            file_list.append(f)
        mode = "files"
    else:
        # Directory mode
        for d in args.dir:
            if not os.path.isdir(d):
                sys.exit(f"Error: directory does not exist: {d}")

        # Discover files from directories
        file_list = []
        for source_dir in args.dir:
            for root, dirs, files in os.walk(source_dir):
                for f in files:
                    if f.find("cnclz") >= 0:
                        continue   # skip canonicalized files
                    if args.use_dups and f.endswith("no-dups.mmls"):
                        file_list.append(f"{root}/{f}")
                    elif not args.use_dups and f.find("no-dups") == -1:   # avoid no-dups.mmls and no-dups-cnclz.mmls
                        file_list.append(f"{root}/{f}")
        mode = "dirs"

    # Summary accumulator
    summary = []

    # Unified processing loop
    for code in args.code:
        config = {
            "BrailleCode": code,
            "dry_run": args.dry_run,
        }

        print(f"\n=== Processing {mode}:")
        print(f"    Braille code: {code}")
        print(f"    Dry run:      {args.dry_run}")
        print(f"    Files:        {len(file_list)}")
        print("")

        stats = ProcessFileList(
            file_list,
            dest_dir,
            config,
            max_workers=24
        )

        summary.append({
            "dir": "(explicit files)" if args.files else args.dir,
            "code": code,
            "dry_run": args.dry_run,
            "stats": stats,
        })

    print_summary(summary)


if __name__ == "__main__":
    main()
