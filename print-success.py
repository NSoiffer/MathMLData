import sys


def process_files(filenames):
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                check_count = 0
                cross_count = 0
                bad_mathml_count = 0
                is_to_mathml = False
                not_table_check_count = 0
                not_table_cross_count = 0

                for line in f:
                    # Stop processing if we hit the specific section header
                    if line.startswith("# Normalized MathML") or line.startswith("Normalized MathML"):
                        is_to_mathml = True
                        break

                    # Check the first character of the line
                    if line.startswith("✓"):
                        check_count += 1
                        if line.find("</mtable>") == -1:
                            not_table_check_count += 1
                    elif line.startswith("✗"):
                        cross_count += 1
                        if line.find("Bad MathML") != -1:
                            bad_mathml_count += 1
                        if line.find("</mtable>") == -1:
                            not_table_cross_count += 1

                # Calculate totals and percentage
                total = check_count + cross_count
                percentage = 0
                if total > 0:
                    percentage = int((check_count / total) * 100)
                not_table_total = not_table_check_count + not_table_cross_count
                not_table_percentage = 0
                if not_table_total > 0:
                    not_table_percentage = int((not_table_check_count / not_table_total) * 100)

                # Print: File Name, Count of ✓, Total (✓+✗), Percentage
                bad_mml_str = f", bad MML: {bad_mathml_count}" if is_to_mathml else ""
                print(
                    f"{percentage}%: {check_count}/{total}, "
                    f"not table: {not_table_percentage}% {not_table_check_count}/{not_table_total} "
                    f"{bad_mml_str} ({filename})"
                )

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"Error reading '{filename}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print-success.py <file1> <file2> ...")
    else:
        process_files(sys.argv[1:])
