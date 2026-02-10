from pathlib import Path
from collections import defaultdict

# The 7 parallel directories you listed
ROOT_DIRS = [
    Path("../../SimpleSpeakData"),
    Path("ASCIIMath"),
    Path("ASCIIMath-6"),
    Path("LateX"),
    Path("LateX-6"),
    Path("Nemeth"),
    Path("UEB"),
]

SUBDIRS = ["highschool", "college"]


def count_lines(path: Path) -> int:
    """Return the number of lines in a file."""
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def base_name(path: Path) -> str:
    """
    Return the filename without its suffix.
    Example: foo.mmls → foo
             foo.brls → foo
    """
    return path.stem


def collect_files():
    """
    Walk all 7 dirs, collect files in highschool/college,
    ignoring any file containing '-no-dups'.
    Group by (subdir, base_filename).
    Returns:
        files[(subdir, base_filename)] = list of (root_dir, full_path)
    """
    files = defaultdict(list)

    for root in ROOT_DIRS:
        for sub in SUBDIRS:
            subpath = root / sub
            if not subpath.exists():
                print(f"WARNING: Missing subdir: {subpath}")
                continue

            for file in subpath.iterdir():
                if not file.is_file():
                    continue

                # Skip any file containing "-no-dups"
                if "-no-dups" in file.name:
                    continue

                key = (sub, base_name(file))
                files[key].append((root, file))

    return files


def verify_line_counts(files):
    """
    For each (subdir, base_filename), check that all 7 dirs
    have the same line count. Print mismatches.
    """
    all_good = True

    for (subdir, base), file_list in sorted(files.items()):
        counts = []
        for root, path in file_list:
            try:
                n = count_lines(path)
            except Exception as e:
                print(f"ERROR reading {path}: {e}")
                continue
            counts.append((root.name, path.name, n))

        # If not all 7 dirs have this file, warn
        if len(counts) != len(ROOT_DIRS):
            all_good = False
            print(f"⚠️  Missing versions of '{base}' in '{subdir}':")
            for root, fname, n in counts:
                print(f"    {root}: {fname} → {n} lines")
            print()
            continue

        # Check consistency
        line_counts = {n for (_, _, n) in counts}
        if len(line_counts) != 1:
            all_good = False
            print(f"❌ MISMATCH for {subdir}/{base} (suffixes ignored):")
            for root, fname, n in counts:
                print(f"    {root}: {fname} → {n} lines")
            print()

    if all_good:
        print("✅ All matching files (ignoring suffixes) have consistent line counts across all 7 directories.")


def main():
    files = collect_files()
    verify_line_counts(files)


if __name__ == "__main__":
    main()
