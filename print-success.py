import sys


def process_files(filenames):
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                check_count = 0
                cross_count = 0

                for line in f:
                    # Stop processing if we hit the specific section header
                    if line.startswith("# Normalized MathML") or line.startswith("Normalized MathML"):
                        break

                    # Check the first character of the line
                    if line.startswith("✓"):
                        check_count += 1
                    elif line.startswith("✗"):
                        cross_count += 1

                # Calculate totals and percentage
                total = check_count + cross_count
                percentage = 0
                if total > 0:
                    percentage = int((check_count / total) * 100)

                # Print: File Name, Count of ✓, Total (✓+✗), Percentage
                print(f"{percentage}%: {check_count}/{total} ({filename})")

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"Error reading '{filename}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_marks.py <file1> <file2> ...")
    else:
        process_files(sys.argv[1:])
