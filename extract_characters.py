import xml.etree.ElementTree as ET
import yaml
from typing import Any
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')  # in case print statements are used for debugging

# Python 3.12+ Type Alias
type CharMapping = dict[str, str]
type CharSet = set[str]


def get_unique_mathml_chars(file_path: str) -> CharSet:
    unique_chars = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                # Parse the MathML line
                # Note: This assumes each line is a valid XML fragment (e.g., starts with <math>)
                root = ET.fromstring(line)

                # itertext() extracts all text between tags and automatically
                # converts entities like &#x2211; to their Unicode equivalents
                for text_content in root.itertext():
                    unique_chars.update(text_content)

            except ET.ParseError:
                # If a line isn't a complete XML tree, you might need to wrap it
                # root = ET.fromstring(f"<root>{line}</root>")
                continue

    return unique_chars


def get_aggregated_mathml_chars(path_str: str) -> CharSet:
    """
    Takes a path to a file or directory, extracts unique MathML characters
    from all relevant files, and returns the union of those character sets.
    """
    path = Path(path_str)
    all_chars: set[str] = set()

    # If it's a single file, process it directly
    if path.is_file():
        return get_unique_mathml_chars(str(path))

    # If it's a directory, iterate through all files
    if path.is_dir():
        # rglob("*") finds all files in this dir and all sub-dirs
        # Use glob("*") if you only want the top-level directory
        for file_path in path.glob("*"):
            if file_path.is_file():
                # Union the existing set with the results from the new file
                all_chars |= get_unique_mathml_chars(str(file_path))

    return all_chars

# Example Usage:
# final_set = get_aggregated_mathml_chars("./mathml_data")
# print(f"Total unique characters across all files: {len(final_set)}")


UEB_REPLACEMENT_CHARS: CharMapping = {
    "S": "⠈⠼",
    "B": "⠘",
    "𝔹": "⠈",
    "T": "⠈",
    "I": "⠨",
    "R": "",
    "1": "⠰",
    "𝟙": "⠰⠰",
    "L": "",
    "D": "⠈",
    "G": "⠨",
    "V": "⠨⠈",
    "C": "⠠",
    "𝐶": "⠠",
    "N": "⠼",
    "t": "⠱",
    "W": "⠀",
    "𝐖": "⠀",
    "s": "⠆",
    "w": "⠂",
    "e": "⠄",
    "o": "",
    "c": "",
    "b": "",
    ",": "⠂",
    ".": "⠲",
    "-": "-",
    "—": "⠠⠤",
    "―": "⠐⠠⠤",
    "#": "",
}

NEMETH_REPLACEMENT_CHARS: dict[str, str] = {
    "S": "⠠⠨",
    "B": "⠸",
    "𝔹": "⠨",
    "T": "⠈",
    "I": "⠨",
    "R": "",
    "E": "⠰",
    "D": "⠸",
    "G": "⠨",
    "V": "⠨⠈",
    "H": "⠠⠠",
    "U": "⠈⠈",
    "C": "⠠",
    "P": "⠸",
    "𝐏": "⠸",
    "L": "",
    "l": "",
    "M": "",
    "m": "⠐",
    "N": "",
    "n": "⠼",
    "𝑁": "",
    "W": "⠀",
    "w": "⠀",
    ",": "⠠⠀",
    "b": "⠐",
    "𝑏": "⣐",
    "↑": "⠘",
    "↓": "⠰",
}


def replace_t_values_in_structure(value: Any, indicator_replacements: CharMapping) -> Any:
    """
    Recursively processes YAML structure and replaces all 't' key values with indicator-replaced versions.
    Preserves the original YAML structure.
    """
    if isinstance(value, dict):
        # Create a copy to avoid modifying the original
        result = {}
        for k, v in value.items():
            if k == 't' and isinstance(v, str):
                # Replace the 't' value with indicator-replaced version
                result[k] = "".join(indicator_replacements.get(ch, ch) for ch in v)
            else:
                # Recursively process nested structures
                result[k] = replace_t_values_in_structure(v, indicator_replacements)
        return result
    elif isinstance(value, list):
        # Process each item in the list
        return [replace_t_values_in_structure(item, indicator_replacements) for item in value]
    else:
        # Return primitive values as-is
        return value


def extract_braille_string(value: Any) -> str | None:
    """
    Extracts the first 't' value from a YAML structure as a string for output purposes.
    Returns None if no 't' value is found.
    """
    if isinstance(value, dict):
        if 't' in value and isinstance(value['t'], str):
            return value['t']
        # Recursively search nested dictionaries
        for v in value.values():
            result = extract_braille_string(v)
            if result is not None:
                return result
    elif isinstance(value, list):
        # Recursively search list items
        for item in value:
            result = extract_braille_string(item)
            if result is not None:
                return result
    return None


def generate_braille_mapping(
    yaml_file_path: str,
    indicator_replacements: CharMapping,
    chars_output_file: str,
    char_set: CharSet
) -> None:
    """
    Parses a YAML file and prints a mapping of keys to their 'else' or
    default braille values for all characters present in char_set.
    """
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            # Type checkers require Any here as YAML structure is dynamic
            data: Any = yaml.safe_load(f)
        with open(yaml_file_path.replace(".yaml", "-full.yaml"), "r", encoding="utf-8") as f:
            # Type checkers require Any here as YAML structure is dynamic
            data_all: Any = yaml.safe_load(f)

        if not isinstance(data, list) or not isinstance(data_all, list):
            return
        data.extend(data_all)
        mapping: CharMapping = {}

        for entry in data:
            if not isinstance(entry, dict):
                continue

            for key, value in entry.items():
                # match value:
                #     # Case 1: Simple list [t: "braille"]
                #     case [{"t": str(b)}]:
                #         mapping[key] = b

                #     # Case 2: Nested test structure with 'else' branch
                #     case [{"test": {"else": [{"t": str(b)}]}}]:
                #         mapping[key] = b

                #     case _:
                #         continue
                # replace the indicators with the replacement characters
                # if key == " ":
                #     del mapping[key]
                #     key = " "
                #     mapping[key] = "⠀"      # non breaking space -> empty braille dots
                # Replace 't' values in the YAML structure while preserving structure
                processed_value = replace_t_values_in_structure(value, indicator_replacements)
                # Handle special case for space character
                if key == " ":
                    # Use non-breaking space as the key
                    mapping[" "] = processed_value
                else:
                    mapping[key] = processed_value

        # Sort for deterministic output; prints only keys found in the YAML
        not_found = []
        with open(chars_output_file, "w", encoding="utf-8") as f:
            f.write("---\n")
            for char in sorted(char_set):
                if structure := mapping.get(char):
                    # Extract braille string from the structure for output
                    # braille = extract_braille_string(structure)
                    if structure:
                        f.write(f' - "{char if char != "\\" else "\\\\"}":  {structure} \n')
                    else:
                        not_found.append(char)
                else:
                    not_found.append(char)
        print(f"Not found: {not_found}")
    except FileNotFoundError:
        print(f"Error: {yaml_file_path} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_all_unique_chars():
    highschool_unique_chars = get_aggregated_mathml_chars("SimpleSpeakData/highschool")
    college_unique_chars = get_aggregated_mathml_chars("SimpleSpeakData/college")
    all_unique_chars = highschool_unique_chars | college_unique_chars
    return all_unique_chars


def get_example_test_unique_chars():
    example_unique_chars = get_unique_mathml_chars("example_data/mathml.mmls")
    test_unique_chars = get_unique_mathml_chars("test_data/mathml.mmls")
    return example_unique_chars | test_unique_chars


# Example usage:
# char_set: CharSet = {'≠', '≡', '≤'}
# generate_braille_mapping('symbols.yaml', char_set)
def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_characters.py <unicode_file.yaml> <chars_output_file.yaml>")
        print("\nArguments:")
        print("  unicode_file.yaml    - Path to the YAML file containing braille mappings")
        print("                        (filename should contain 'nemeth' or 'ueb' to determine type)")
        print("  chars_output_file.yaml - Path to the output file for character mappings")
        print("\nExample:")
        print("  python extract_characters.py MathCAT/Rules/Braille/Nemeth/unicode.yaml Nemeth_charmap.yaml")
        sys.exit(1)

    unicode_file_name = sys.argv[1]
    chars_output_file = sys.argv[2]
    if "nemeth" in unicode_file_name.lower():
        indicator_dict = NEMETH_REPLACEMENT_CHARS
    else:
        indicator_dict = UEB_REPLACEMENT_CHARS
    example_test_unique_chars = get_example_test_unique_chars()
    print(f"Unique characters: {len(example_test_unique_chars)}")
    print(sorted(list(example_test_unique_chars)))
    generate_braille_mapping(
        unicode_file_name,
        indicator_dict,
        chars_output_file,
        example_test_unique_chars
    )

    all_unique_chars = get_all_unique_chars()
    print(f"Unique characters: {len(all_unique_chars)}")
    print(sorted(list(all_unique_chars)))
    generate_braille_mapping(
        unicode_file_name,
        indicator_dict,
        chars_output_file.replace(".yaml", "_all.yaml"),
        all_unique_chars
    )

# Example usage:
# char_set = get_unique_mathml_chars('data.mml')
# print(f"Unique characters: {sorted(list(char_set))}")


if __name__ == '__main__':
    main()
