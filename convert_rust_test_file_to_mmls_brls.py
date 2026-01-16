import re
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')  # in case print statements are used for debugging


def extract_from_files(file_list, mathml_output, braille_output):
    # Clear existing output files if they exist before starting the batch
    for out_file in [mathml_output, braille_output]:
        if os.path.exists(out_file):
            os.remove(out_file)

    total_matches = 0
    total_validated = 0

    # Patterns
    expr_part = r'let\s+expr\s*=\s*(?:r#*"(.*?)"#*|"(.*?)")\s*;'
    call_part = r'\s*test_braille(?:_prefs)?\s*\(\s*.*?\s*expr\s*,\s*"([^"]*)"\s*\)\s*;'
    combined_pattern = expr_part + call_part

    for filename in file_list:
        if not os.path.isfile(filename):
            print(f"Skipping: '{filename}' (File not found)")
            continue

        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Strip // comments
        content = "".join([line.split('//')[0] for line in lines])

        matches = re.findall(combined_pattern, content, re.DOTALL)

        file_mathml = []
        file_braille = []

        for raw_m, std_m, braille in matches:
            total_matches += 1
            mathml_raw = raw_m if raw_m else std_m
            clean_mathml = " ".join(mathml_raw.split()).strip()

            if clean_mathml.startswith("<math") and clean_mathml.endswith("</math>"):
                file_mathml.append(clean_mathml)
                file_braille.append(braille)
                total_validated += 1
            else:
                print(f"Validation failed in {filename} for mathml: {clean_mathml[:30]}...")

        # Append this file's results to the master lists
        with open(mathml_output, 'a', encoding='utf-8') as f:
            for item in file_mathml:
                f.write(f"{item}\n")

        with open(braille_output, 'a', encoding='utf-8') as f:
            for item in file_braille:
                f.write(f"{item}\n")

        print(f"Processed {filename}: Found {len(matches)} tests.")

    print("\n--- Final Batch Summary ---")
    print(f"Files scanned: {len(file_list)}")
    print(f"Total tests identified: {total_matches}")
    print(f"Total tests validated and saved: {total_validated}")


if __name__ == "__main__":
    # Define your list of files here
    my_files = [
        'C:/Users/neils/MathCAT/tests/braille/Nemeth/rules.rs',
        'C:/Users/neils/MathCAT/tests/braille/Nemeth/other.rs',
        'C:/Users/neils/MathCAT/tests/braille/Nemeth/chemistry.rs'
    ]

    extract_from_files(my_files, 'all_mathml.txt', 'all_braille.txt')

# if __name__ == "__main__":
#     # Replace 'tests.rs' with your actual filename
#     extract_rust_tests('C:/Users/neils/MathCAT/tests/braille/Nemeth/chemistry.rs',
#                       'RustTestData/mathml.txt', 'RustTestData/braille.txt')
