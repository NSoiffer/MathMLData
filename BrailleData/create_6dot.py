import argparse
import os

UEB_PUNCT = {
    ",": 1, ";": 1, ":": 1, ".": 1, "!": 1, "?": 1,
    "'": 1, "-": 1, "–": 1, "—": 1,
    "(": 1, ")": 1,
    "[": 1, "]": 1,
    "/": 1, "\\": 1,
    "\"": 1,
}


def count_ueb_cells(text: str) -> int:
    total = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Space
        if ch == " ":
            total += 1
            i += 1
            continue

        # Digits: number indicator + digits
        if ch.isdigit():
            total += 1  # number indicator ⠼
            while i < n and text[i].isdigit():
                total += 1
                i += 1

            # Grade‑1 indicator if next char is a–j
            if i < n and text[i].isalpha() and text[i].lower() in "abcdefghij":
                total += 1  # ⠰

            continue

        # Capital letters
        if ch.isalpha() and ch.isupper():
            start = i
            while i < n and text[i].isalpha() and text[i].isupper():
                i += 1
            length = i - start

            if length == 1:
                total += 2  # ⠠ + letter
            else:
                total += 2 + length  # ⠠⠠ + letters
            continue

        # Lowercase letters
        if ch.isalpha():
            total += 1
            i += 1
            continue

        # Punctuation
        if ch in UEB_PUNCT:
            total += UEB_PUNCT[ch]
            i += 1
            continue

        # Fallback
        total += 1
        i += 1

    return total


def process_file(infile: str, outfile: str):
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(infile, "r", encoding="utf-8") as fin, \
         open(outfile, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.rstrip("\n")
            count = count_ueb_cells(line)
            fout.write("*" * count + "\n")


def main():
    parser = argparse.ArgumentParser(description="UEB cell counter")
    parser.add_argument("--in", dest="indir", required=True,
                        help="Input directory to read files from")
    parser.add_argument("--out", dest="outdir", required=True,
                        help="Output directory to write processed files into")

    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir

    # Create output root if needed
    os.makedirs(outdir, exist_ok=True)

    # Walk input directory recursively
    for root, dirs, files in os.walk(indir):
        for fname in files:
            infile = os.path.join(root, fname)

            # Compute relative path
            rel = os.path.relpath(infile, indir)
            outfile = os.path.join(outdir, rel)

            process_file(infile, outfile)


if __name__ == "__main__":
    main()
