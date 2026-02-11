from __future__ import annotations

from pathlib import Path
from typing import List
import argparse


def merge_subdirs_to_single_file(root_dir: Path, suffix: str) -> None:
    """
    Given a directory containing 'highschool' and 'college' subdirectories,
    read all files with the given suffix (e.g. '.brls', '.mmls') from both
    subdirectories, excluding any whose names contain '-no-dups', and write
    them into a single file named '<root><suffix>' placed at the same level
    as the directory.

    Example:
        merge_subdirs_to_single_file(Path("Braille/Nemeth"), ".brls")
        → writes Braille/Nemeth.brls
    """
    if not root_dir.is_dir():
        raise ValueError(f"Not a directory: {root_dir}")

    high = root_dir / "highschool"
    col = root_dir / "college"

    if not high.is_dir() or not col.is_dir():
        raise ValueError(
            f"Directory {root_dir} must contain 'highschool' and 'college' subdirectories."
        )

    # Collect files with the given suffix, excluding "-no-dups"
    def filtered_files(directory: Path) -> List[Path]:
        return sorted(
            f for f in directory.glob(f"*{suffix}")
            if "-cnclz" not in f.name
        )

    files: List[Path] = filtered_files(high) + filtered_files(col)
    print("First 5 files being read:", "\n".join(str(f) for f in files[:5]))

    # Output file: <root><suffix> at the same level as root_dir
    out_file = root_dir.with_suffix(suffix)

    with open(out_file, "w", encoding="utf-8") as out:
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)

    print(f"Wrote merged file: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge highschool/ and college/ subdirectory files into a single output file."
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help="File suffix to merge (e.g. .brls, .mmls). Must include the leading dot.",
    )
    parser.add_argument(
        "-dirs",
        nargs="+",
        help="One or more directories containing highschool/ and college/ subdirectories.",
    )

    args = parser.parse_args()
    suffix: str = args.suffix

    if not suffix.startswith("."):
        raise ValueError("Suffix must begin with a dot, e.g. '.brls'")

    for d in args.dirs:
        merge_subdirs_to_single_file(Path(d), suffix)


if __name__ == "__main__":
    main()
