import pathlib
import sys
import re
FIX_MSPACE = re.compile(r"<mspace (.+?)=[\"\'](\d*?\.?\d*?)(.+?)[\"\']>&lt;mspace .+?=[\"\'].+?[\"\']/&gt;")
FIX_MSPACE_NAMEDSPACE = re.compile(r'<mspace (.+?)=[\"\']([a-z]+?space)[\"\']>&lt;mspace width=[\"\']([a-z]+?space)[\"\']/&gt;')


def cleanup(mathml: str) -> str:
    """
    There is a bug in some MathML data; this attempts to fix it.
    """
    fixed = FIX_MSPACE.sub(r'<mspace \1="\2\3">', mathml)
    fixed = FIX_MSPACE_NAMEDSPACE.sub(r'<mspace \1="\2">', fixed)
    fixed = (fixed.replace(r"<mspace>&lt;mspace/&gt;</mspace>", r"<mspace></mspace>")
                  .replace(r"<none>&lt;none/&gt;</none>", r"<none></none>"))
    return fixed


def process_mml_files(directory_path: str) -> None:
    base_dir = pathlib.Path(directory_path)
    mml_files = list(base_dir.glob("*.mmls"))  # Adjusted to .mmls per your request

    for file_path in mml_files:
        print(f"Processing: {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            cleaned_content = []
            for line in lines:
                cleaned = cleanup(line)
                cleaned_content.append(cleaned)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_content)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")


if __name__ == "__main__":
    # Check if directory argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <directory_path>")
    else:
        # Pass the first command line argument as the directory
        process_mml_files(sys.argv[1])
