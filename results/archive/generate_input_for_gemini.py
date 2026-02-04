import itertools


def generateExamples(braille_path, mathml_path, out_path, start_line, end_line):
    """
    Zips lines from two files within a specified range (inclusive, 1-based indexing)
    and prints the combined result.
    """
    # Basic validation
    if start_line < 1 or end_line < start_line:
        print("Error: Invalid line range.")
        return

    try:
        with open(braille_path, 'r', encoding='utf-8') as braille, \
             open(mathml_path, 'r', encoding='utf-8') as mathml, \
             open(out_path, 'w', encoding='utf-8') as out_file:

            # Create iterators for the specific range
            # islice uses 0-based indexing, so we adjust start_line by -1
            brl_line = itertools.islice(braille, start_line - 1, end_line)
            mml_line = itertools.islice(mathml, start_line - 1, end_line)

            # Zip and print
            for i, (brl, mml) in enumerate(zip(brl_line, mml_line), start=start_line):
                # .strip() removes the trailing newline character for cleaner output
                quoted_mml = mml.strip().replace('"', '\\"')
                out_file.write(f'        "{brl.strip()} | {quoted_mml}\\n"\n')

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generateTestAndExpected(test_path, expected_path, out_path, start_line, end_line):
    """
    Zips lines from two files within a specified range (inclusive, 1-based indexing)
    and prints the combined result.
    """
    # Basic validation
    if start_line < 1 or end_line < start_line:
        print("Error: Invalid line range.")
        return

    try:
        with open(test_path, 'r', encoding='utf-8') as test, \
             open(expected_path, 'r', encoding='utf-8') as expected, \
             open(out_path, 'w', encoding='utf-8') as out_file:

            # Create iterators for the specific range
            # islice uses 0-based indexing, so we adjust start_line by -1
            lines = list(itertools.islice(test, start_line - 1, end_line))
            out_file.write("    tests = [\n")
            for line in lines:
                out_file.write(f"        '{line.strip()}',\n")
            out_file.write("    ]\n\n")
            lines = list(itertools.islice(expected, start_line - 1, end_line))
            out_file.write("    expected = [\n")
            for line in lines:
                line = line.replace('"', '\\"')
                out_file.write(f'        "{line.strip()}",\n')
            out_file.write("    ]\n\n")
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    generateExamples("RustTestData/Nemeth.brls", "RustTestData/Nemeth.mmls", "examples.txt", 1, 308)
    generateTestAndExpected("BrailleData/Braille/Nemeth/highschool/Algebra Toolkit-no-dups.brls",
                            "SimpleSpeakData/highschool/Algebra Toolkit-no-dups.mmls",
                            "test_and_expected.txt",
                            100, 299)
