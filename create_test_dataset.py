#!/usr/bin/env python3
"""
Script to create test and example datasets by randomly sampling elements each
from high school sources, keeping alignment across Nemeth, UEB, and MathML versions.
The test and example datasets are guaranteed to be disjoint per source file.
"""

import argparse
import random
from pathlib import Path
from compare_mathml_in_csv import setMathCATPreferences, setMathMLForMathCAT


def get_source_files(base_path):
    """Get all deduplicated source files from the highschool directory."""
    sources = []
    for category in ['highschool']:  # could also add "college" if desired
        category_path = base_path / category
        if category_path.exists():
            for file in category_path.glob('*-no-dups.*'):
                sources.append((category, file.stem.replace('-no-dups', ''), file.suffix))
    return sources


def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def sample_aligned_data(nemeth_file, ueb_file, mathml_file, sample_size):
    """
    Sample aligned data from all three file types.
    Returns the sampled lines from each file, maintaining alignment.
    """
    # Read all files
    nemeth_lines = read_file_lines(nemeth_file)
    ueb_lines = read_file_lines(ueb_file)
    mathml_lines = read_file_lines(mathml_file)

    # Verify all files have the same number of lines
    min_lines = min(len(nemeth_lines), len(ueb_lines), len(mathml_lines))

    if min_lines == 0:
        return [], [], []

    # Adjust sample size if file is smaller
    actual_sample_size = min(sample_size, min_lines)

    # Randomly sample line indices
    sampled_indices = sorted(random.sample(range(min_lines), actual_sample_size))

    # Extract sampled lines
    nemeth_samples = [nemeth_lines[i] for i in sampled_indices]
    ueb_samples = [ueb_lines[i] for i in sampled_indices]
    mathml_samples = [mathml_lines[i] for i in sampled_indices]

    return nemeth_samples, ueb_samples, mathml_samples


def filter_out_mtable_elements(mathml_lines, nemeth_lines, ueb_lines) -> tuple[list[str], list[str], list[str]]:
    # Use zip to pair them, filter out 'mtable', then unpack them back into lists
    mathml, nemeth, ueb = zip(*[
        (m, n, u) for m, n, u in zip(mathml_lines, nemeth_lines, ueb_lines)
        if "mtable" not in m
    ])
    return list(mathml), list(nemeth), list(ueb)


def write_samples(output_dir, subfolder, source_name, extension, lines):
    """Write sampled lines to output file."""
    output_subdir = output_dir / subfolder
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_file = output_subdir / f"{source_name}{extension}"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def split_aligned_data(nemeth_file, ueb_file, mathml_file, sample_size):
    """
    Sample two disjoint aligned sets from all three file types.
    Returns:
        Dictionary with keys: test_nemeth, test_ueb, test_mathml,
        example_nemeth, example_ueb, example_mathml
    """
    nemeth_lines = read_file_lines(nemeth_file)
    ueb_lines = read_file_lines(ueb_file)
    mathml_lines = read_file_lines(mathml_file)

    if len(nemeth_lines) != len(ueb_lines) or len(ueb_lines) != len(mathml_lines):
        print(f"Error: {nemeth_file}[{len(nemeth_lines)}] or "
              f"{ueb_file}[{len(ueb_lines)}] or "
              f"{mathml_file}[{len(mathml_lines)}] has different lengths")
        return {
            'test_nemeth': [],
            'test_ueb': [],
            'test_mathml': [],
            'example_nemeth': [],
            'example_ueb': [],
            'example_mathml': [],
        }

    mathml_lines, nemeth_lines, ueb_lines = filter_out_mtable_elements(mathml_lines, nemeth_lines, ueb_lines)
    n_lines = len(nemeth_lines)
    test_size = min(2*sample_size, n_lines)

    all_indices = list(range(0, test_size))
    random_indices = random.sample(all_indices, test_size)
    unique_indices: list[int] = []
    unique_lines = set()
    for i in range(0, len(random_indices)):
        if all_indices[i] not in unique_lines:
            unique_lines.add(all_indices[i])
            unique_indices.append(random_indices[i])
    if len(unique_indices) < sample_size:
        return split_aligned_data(nemeth_file, ueb_file, mathml_file, sample_size)

    unique_indices = unique_indices[:2*sample_size]
    unique_indices = sorted(unique_indices)
    test_nemeth = []
    test_ueb = []
    test_mathml = []
    example_nemeth = []
    example_ueb = []
    example_mathml = []
    for i in range(0, len(unique_indices), 2):
        test_nemeth.append(nemeth_lines[unique_indices[i]])
        test_ueb.append(ueb_lines[unique_indices[i]])
        test_mathml.append(mathml_lines[unique_indices[i]])

        example_nemeth.append(nemeth_lines[unique_indices[i+1]])
        example_ueb.append(ueb_lines[unique_indices[i+1]])
        example_mathml.append(mathml_lines[unique_indices[i+1]])

    return {
        'test_nemeth': test_nemeth,
        'test_ueb': test_ueb,
        'test_mathml': test_mathml,
        'example_nemeth': example_nemeth,
        'example_ueb': example_ueb,
        'example_mathml': example_mathml,
    }


def collect_existing_data(
    test_data_dir: Path,
    example_data_dir: Path
) -> dict[str, list[str]]:
    """
    Collect existing data from test_data and example_data directories.

    Reads from subdirectories: Nemeth, UEB, MathML, and CanonicalMathML.
    Matches files by base name (without extension) to maintain alignment.

    Args:
        test_data_dir: Path to test_data directory
        example_data_dir: Path to example_data directory

    Returns:
        Dictionary with keys: test_nemeth, test_ueb, test_mathml,
        example_nemeth, example_ueb, example_mathml,
        canonical_test_mathml, canonical_example_mathml
        Each list contains all lines from all matching files.
    """
    def collect_from_directory(base_dir: Path) -> tuple[list[str], list[str], list[str], list[str]]:
        """Collect data from a single directory (test or example)."""
        nemeth_dir = base_dir / 'Nemeth'
        ueb_dir = base_dir / 'UEB'
        mathml_dir = base_dir / 'MathML'
        canonical_dir = base_dir / 'CanonicalMathML'

        # Find all files and group by base name (without extension)
        nemeth_files = {}
        ueb_files = {}
        mathml_files = {}
        canonical_files = {}

        if nemeth_dir.exists():
            for file in nemeth_dir.glob('*.brls'):
                base_name = file.stem
                nemeth_files[base_name] = file

        if ueb_dir.exists():
            for file in ueb_dir.glob('*.brls'):
                base_name = file.stem
                ueb_files[base_name] = file

        if mathml_dir.exists():
            for file in mathml_dir.glob('*.mmls'):
                base_name = file.stem
                mathml_files[base_name] = file

        if canonical_dir.exists():
            for file in canonical_dir.glob('*.mmls'):
                base_name = file.stem
                canonical_files[base_name] = file

        # Find common base names across all four types
        all_base_names = set(nemeth_files.keys()) & set(ueb_files.keys()) & \
                        set(mathml_files.keys()) & set(canonical_files.keys())
        all_base_names = sorted(all_base_names)  # Sort for consistent ordering

        # Collect lines from all matching files
        nemeth_lines = []
        ueb_lines = []
        mathml_lines = []
        canonical_lines = []

        for base_name in all_base_names:
            # Read all lines from each file
            nemeth_lines.extend(read_file_lines(nemeth_files[base_name]))
            ueb_lines.extend(read_file_lines(ueb_files[base_name]))
            mathml_lines.extend(read_file_lines(mathml_files[base_name]))
            canonical_lines.extend(read_file_lines(canonical_files[base_name]))

        return nemeth_lines, ueb_lines, mathml_lines, canonical_lines

    # Collect test and example data
    test_nemeth, test_ueb, test_mathml, test_canonical_mathml = collect_from_directory(test_data_dir)
    example_nemeth, example_ueb, example_mathml, example_canonical_mathml = collect_from_directory(example_data_dir)

    return {
        'test_nemeth': test_nemeth,
        'test_ueb': test_ueb,
        'test_mathml': test_mathml,
        'example_nemeth': example_nemeth,
        'example_ueb': example_ueb,
        'example_mathml': example_mathml,
        'canonical_test_mathml': test_canonical_mathml,
        'canonical_example_mathml': example_canonical_mathml,
    }


def process_source_files(
    sources: list[tuple[str, str, str]],
    excluded_sources: set[str],
    paths: dict[str, Path],
    sample_size: int,
) -> dict[str, list[str]]:
    """
    Process all source files, sampling data and writing output files.

    Args:
        sources: List of (category, source_name, suffix) tuples
        excluded_sources: Set of source names to skip
        paths: Dictionary with keys: nemeth_base, ueb_base, mathml_base,
               test_output_dir, example_output_dir, test_canonical_output_dir,
               example_canonical_output_dir
        sample_size: Number of samples to take from each source

    Returns:
        Dictionary with keys: test_nemeth, test_ueb, test_mathml,
        example_nemeth, example_ueb, example_mathml,
        canonical_test_mathml, canonical_example_mathml
    """
    # Initialize all lists
    data = {
        'test_nemeth': [],
        'test_ueb': [],
        'test_mathml': [],
        'example_nemeth': [],
        'example_ueb': [],
        'example_mathml': [],
        'canonical_test_mathml': [],
        'canonical_example_mathml': []
    }

    # Mapping for writing samples: (key, subfolder, extension)
    write_configs = [
        ('test_nemeth', 'Nemeth', '.brls'),
        ('test_ueb', 'UEB', '.brls'),
        ('test_mathml', 'MathML', '.mmls'),
        ('example_nemeth', 'Nemeth', '.brls'),
        ('example_ueb', 'UEB', '.brls'),
        ('example_mathml', 'MathML', '.mmls'),
    ]

    for category, source_name, _ in sources:
        if source_name in excluded_sources:
            print(f"Skipping {source_name}: excluded")
            continue

        nemeth_file = paths['nemeth_base'] / category / f"{source_name}-no-dups.brls"
        ueb_file = paths['ueb_base'] / category / f"{source_name}-no-dups.brls"
        mathml_file = paths['mathml_base'] / category / f"{source_name}-no-dups.mmls"

        # Sample disjoint aligned data
        samples_dict = split_aligned_data(
            nemeth_file, ueb_file, mathml_file, sample_size
        )

        # Write samples using loop
        for key, subfolder, ext in write_configs:
            output_dir = paths['test_output_dir'] if key.startswith('test') else paths['example_output_dir']
            write_samples(output_dir, subfolder, source_name, ext, samples_dict[key])

        # Canonicalize test and example MathML entries
        canonical_test_mathml = canonicalize_mathml_list(samples_dict['test_mathml'])
        canonical_example_mathml = canonicalize_mathml_list(samples_dict['example_mathml'])
        write_samples(paths['test_canonical_output_dir'], '', source_name, '.mmls', canonical_test_mathml)
        write_samples(paths['example_canonical_output_dir'], '', source_name, '.mmls', canonical_example_mathml)

        # Add canonical data to samples_dict for extending
        samples_dict['canonical_test_mathml'] = canonical_test_mathml
        samples_dict['canonical_example_mathml'] = canonical_example_mathml

        # Extend all lists using a loop
        for key in samples_dict:
            data[key].extend(samples_dict[key])

    return data


def canonicalize_mathml_list(mathml_list: list[str]) -> list[str]:
    """Canonicalize a list of MathML strings using libmathcat.setMathMLForMathCAT.
    Returns a list with one MathML expression per line (newlines in MathML replaced with spaces).
    """
    canonicalized = []
    for mathml in mathml_list:
        mathml_stripped = mathml.strip()
        if not mathml_stripped:
            canonicalized.append('\n')
            continue
        try:
            canonical = setMathMLForMathCAT(mathml_stripped)
            canonicalized.append(canonical + '\n')
        except Exception as e:
            print(f"Warning: Error canonicalizing MathML: {e}")
            canonicalized.append('\n')
    return canonicalized


def main():
    parser = argparse.ArgumentParser(description='Create test and example datasets')
    parser.add_argument(
        '-combined-only',
        action='store_true',
        help='Read existing files from test_data and example_data instead of processing source files'
    )
    args = parser.parse_args()

    random.seed(42)

    # Initialize MathCAT
    setMathCATPreferences({})

    project_root = Path(__file__).parent
    nemeth_base = project_root / 'BrailleData' / 'Braille' / 'Nemeth'
    ueb_base = project_root / 'BrailleData' / 'Braille' / 'UEB'
    mathml_base = project_root / 'SimpleSpeakData'
    test_output_dir = project_root / 'test_data'
    example_output_dir = project_root / 'example_data'
    test_canonical_output_dir = test_output_dir / 'CanonicalMathML'
    example_canonical_output_dir = example_output_dir / 'CanonicalMathML'

    if args.combined_only:
        # Collect existing data from subdirectories
        data = collect_existing_data(test_output_dir, example_output_dir)
    else:
        excluded_sources = {
            'Adventures of Small Number__58__ A collection of short stories',
            'The Story of 8',
        }

        # Get all source files from Nemeth directory (use as reference)
        sources = get_source_files(nemeth_base)

        print(f"Found {len(sources)} source files")

        # Prepare paths dictionary
        paths = {
            'nemeth_base': nemeth_base,
            'ueb_base': ueb_base,
            'mathml_base': mathml_base,
            'test_output_dir': test_output_dir,
            'example_output_dir': example_output_dir,
            'test_canonical_output_dir': test_canonical_output_dir,
            'example_canonical_output_dir': example_canonical_output_dir,
        }

        # Process all source files
        data = process_source_files(
            sources,
            excluded_sources,
            paths,
            sample_size=120
        )

    # Create a single randomized order and apply it to all lists to maintain alignment
    n_items = len(data['test_nemeth'])
    if n_items > 0:
        # Verify all lists have the same length
        for key in data:
            assert len(data[key]) == n_items, f"List {key} has length {len(data[key])}, expected {n_items}"

        # Create a single random permutation
        indices = list(range(n_items))
        random.shuffle(indices)

        # Apply the same permutation to all lists using a loop
        for key in data:
            data[key] = [data[key][i] for i in indices]

    # Write combined files using a loop
    # Mapping: (key, output_dir, filename)
    output_configs = [
        ('test_nemeth', test_output_dir, 'nemeth.brls'),
        ('test_ueb', test_output_dir, 'ueb.brls'),
        ('test_mathml', test_output_dir, 'mathml.mmls'),
        ('canonical_test_mathml', test_output_dir, 'canonical-mathml.mmls'),
        ('example_nemeth', example_output_dir, 'nemeth.brls'),
        ('example_ueb', example_output_dir, 'ueb.brls'),
        ('example_mathml', example_output_dir, 'mathml.mmls'),
        ('canonical_example_mathml', example_output_dir, 'canonical-mathml.mmls'),
    ]

    for key, output_dir, filename in output_configs:
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            f.writelines(data[key])

    print("\nCompleted!")
    print("Total samples:")
    print(f"  - Test Nemeth: {len(data['test_nemeth'])}")
    print(f"  - Test UEB: {len(data['test_ueb'])}")
    print(f"  - Test MathML: {len(data['test_mathml'])}")
    print(f"  - Test Canonical MathML: {len(data['canonical_test_mathml'])}")
    print(f"  - Example Nemeth: {len(data['example_nemeth'])}")
    print(f"  - Example UEB: {len(data['example_ueb'])}")
    print(f"  - Example MathML: {len(data['example_mathml'])}")
    print(f"  - Example Canonical MathML: {len(data['canonical_example_mathml'])}")
    print(f"\nTest data set saved to: {test_output_dir}")
    print(f"Example dataset saved to: {example_output_dir}")


if __name__ == '__main__':
    main()
