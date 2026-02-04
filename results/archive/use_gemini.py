import time
import logging
import re
import argparse
import threading
from typing import Any, NamedTuple  # 'Any' is still imported from typing
from google import genai
from google.genai import types
from google.api_core import exceptions
from compare_mathml_in_csv import setMathCATPreferences, areCanonicallyEqual, CanonicalResults
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')   # Ensure UTF-8 output for Unicode Braille

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class RunConfig(NamedTuple):
    """Configuration values for a Gemini API run."""
    braille_code: str
    gen_braille: bool
    model: str
    apiKeyName: str
    batch_size: int
    n_examples: int
    instructions: str
    example_braille_file: str
    example_mathml_file: str
    input_braille_file: str
    input_mathml_file: str

    def print_config(self, n_tests: int | None = None, short: bool = False) -> str:
        """Return configuration values as a formatted string."""
        lines = []
        lines.append("\nConfiguration:")
        lines.append(f"  Braille Code: {self.braille_code}")
        lines.append(f"  Generate Braille: {self.gen_braille}")
        lines.append(f"  Model: {self.model}")
        lines.append(f"  API Key: {self.apiKeyName}")
        lines.append(f"  Batch Size: {self.batch_size}")
        lines.append(f"  Number of Examples: {self.n_examples}")
        if n_tests is not None:
            lines.append(f"  Number of Tests: {n_tests}")
        lines.append(f"  Example Braille File: {self.example_braille_file}")
        lines.append(f"  Example MathML File: {self.example_mathml_file}")
        lines.append(f"  Input Braille Dir: {self.input_braille_file}")
        lines.append(f"  Input MathML Dir: {self.input_mathml_file}")
        if short:
            instructions_preview = self.instructions[:150] + "..." if len(self.instructions) > 150 else self.instructions
            lines.append(f"  Instructions (preview): {instructions_preview}")
        else:
            lines.append(f"  Instructions: {self.instructions}")
        return "\n".join(lines)


def convert_input_with_model(
    instructions: str,
    input: list[str],
    model: str,
    apiKeyName: str,
    batch_size: int,
    run_info: str
) -> tuple[list[str], dict[str, int], float]:
    """
    Splits input into batches, processes them with retries/streaming,
    tracks token usage, and measures pure generation time.
    """

    # Setup Client
    api_key = os.environ.get(apiKeyName)
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    client = genai.Client(api_key=api_key, http_options={"timeout": 2400000})  # isn't allowed in generate_content

    # 1. Initialize accumulators
    all_results = ""
    total_tokens: dict[str, int] = {"prompt": 0, "candidates": 0, "total": 0}
    total_generation_time: float = 0.0
    first_attempt = True

    # 2. Loop through the data in chunks
    for i in range(0, len(input), batch_size):
        batch = input[i:i + batch_size]
        batch_id = (i // batch_size) + 1
        print(f"\n--- Processing Batch {batch_id} (Items {i+1} to {i+len(batch)}) {run_info} ---")

        # 3. Call helper (now returns duration too)
        # We use Any for usage_metadata to be safe across SDK versions
        try:
            batch_text, batch_usage, batch_time = generate_with_retry(client, model, instructions, batch, run_info, 3)
        except (exceptions.ServiceUnavailable, exceptions.ServerError) as e:
            if first_attempt:
                # reestablish connection and try one more time
                client = genai.Client(api_key=api_key, http_options={"timeout": 2400000})  # likely large than needed
                first_attempt = False
                batch_text, batch_usage, batch_time = generate_with_retry(client, model, instructions, batch, run_info, 3)
            else:
                print(f"Exception raised twice during generation: {e}")
                batch_text, batch_usage, batch_time = None, None, 0.0
                break

        except Exception as e:
            print(f"Exception raised during generation: {e}")
            batch_text, batch_usage, batch_time = None, None, 0.0
            break

        # 4. Process results
        if batch_text:
            all_results += '|next-item|' + batch_text

        # 5. Update stats
        if batch_usage:
            # Assuming batch_usage is a types.GenerateContentResponseUsageMetadata object
            total_tokens["prompt"] += batch_usage.prompt_token_count
            total_tokens["candidates"] += batch_usage.candidates_token_count
            total_tokens["total"] += batch_usage.total_token_count

        if batch_time:
            total_generation_time += batch_time
            print(f"   > Batch Time: {batch_time:.2f}s")
            # We use safe access or Any here since it's a specific SDK object
            if batch_usage:
                print(f"   > Batch Token Usage ({run_info}): {batch_usage.total_token_count} "
                      f"(Prompt: {batch_usage.prompt_token_count}, Output: {batch_usage.candidates_token_count})")

    # trim the start and end, then split the string at '|next-item|' and return a list of strings
    text = all_results
    if input[0].find("<math"):
        i_start = text.find("<math")
        i_end = text.rfind("</math>") + len("</math>")
        if i_start == -1 or i_end == -1:
            raise Exception("Could not find MathML tags in the response.")
    else:
        matches = list(re.finditer(r'[\u2800-\u28ff]', text))
        if not matches:
            raise Exception("Could not find braille chars in the response.")
        i_start = matches[0].start()
        i_end = matches[-1].end()

    # Return the substring including everything between the first and last Braille char
    as_list = text[i_start:i_end].split("|next-item|")
    as_list = [item.strip() for item in as_list if item.strip()]  # Clean up whitespace and remove empty strings
    return as_list, total_tokens, total_generation_time


def _sum_usage(usage1: Any, usage2: Any) -> Any:
    """Helper to sum two UsageMetadata objects."""
    if not usage1:
        return usage2
    if not usage2:
        return usage1

    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage1.prompt_token_count + usage2.prompt_token_count,
        candidates_token_count=usage1.candidates_token_count + usage2.candidates_token_count,
        total_token_count=usage1.total_token_count + usage2.total_token_count
    )


def generate_with_retry(
    client: genai.Client,
    model: str,
    instructions: str,
    content: list[str],
    run_info: str,
    max_retries: int = 3,
    depth: int = 0
) -> tuple[str, Any, float]:

    indent = "  " * depth
    t0 = time.perf_counter()
    time_to_first_token = -1000.0      # initialize to a negative value to indicate not yet received
    delay = 30

    # Attempt Loop (Handles 503s and standard failures)
    for attempt in range(1, max_retries + 1):
        try:
            payload_text = "\n".join(content)

            response_stream = client.models.generate_content_stream(
                model=model,
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE",
                        )
                    ],
                    system_instruction=instructions,
                    temperature=0.1,
                ),
                contents=payload_text,
            )

            full_text_list: list[str] = []
            first_token_received = False
            final_usage: Any = None
            finish_reason = None

            for chunk in response_stream:
                # Capture the exact moment the first chunk arrives
                if not first_token_received:
                    time_to_first_token = time.perf_counter() - t0
                    print(f"⚡ Time to First Token: {time_to_first_token:.4f} seconds")
                    first_token_received = True
                if chunk.text:
                    full_text_list.append(chunk.text)
                if chunk.usage_metadata:
                    final_usage = chunk.usage_metadata
                if chunk.candidates:
                    finish_reason = chunk.candidates[0].finish_reason

            # --- Validation ---
            if finish_reason == "MAX_TOKENS":
                raise ValueError("MAX_TOKENS")

            if finish_reason != "STOP":
                raise Exception(f"Incomplete generation: {finish_reason}")

            print(f"\n\n--- Performance Summary for {run_info} ---")
            total_time = time.perf_counter() - t0
            print(f"Total Latency:    {total_time:.2f} s")
            print(f"Time to 1st Token:{time_to_first_token:.2f} s")
            print(f"Generation Time:  {total_time - time_to_first_token:.2f} s (Streaming duration)")
            return "".join(full_text_list), final_usage, total_time - time_to_first_token

        # --- Recursive Split Logic (Divide and Conquer) ---
        except ValueError as e:
            # Only handle MAX_TOKENS here; re-raise actual ValueErrors
            if str(e) != "MAX_TOKENS":
                raise e

            print(f"{indent}[!] In {run_info}: MAX_TOKENS hit on {len(content)} lines.")

            if len(content) <= 1:
                print(f"{indent}[X] Critical in {run_info}: Single input line is too large.")
                raise e

            mid = len(content) // 2
            left_part = content[:mid]
            right_part = content[mid:]

            print(f"{indent}    -> Splitting: {len(left_part)} lines | {len(right_part)} lines")

            # Recursive calls (Depth + 1)
            # We don't need to catch exceptions here; let them bubble up
            text_a, usage_a, _ = generate_with_retry(
                client, model, instructions, left_part, run_info, max_retries, depth + 1
            )
            text_b, usage_b, _ = generate_with_retry(
                client, model, instructions, right_part, run_info, max_retries, depth + 1
            )

            if text_a is None or text_b is None:
                return "Error", dict(), time.perf_counter() - t0 - time_to_first_token

            return (text_a + '|next-item|' + text_b,
                    _sum_usage(usage_a, usage_b),
                    time.perf_counter() - t0 - time_to_first_token
                    )

        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"{indent}[!] 503 Unavailable (Attempt {attempt}/{max_retries}) {run_info}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"{indent}Exception Type: {type(e).__name__}")
                print(f"{indent}[X] Critical Error in {run_info}: {e}")
                if len(full_text_list) > 0:
                    total_time = time.perf_counter() - t0
                    return "".join(full_text_list), final_usage, total_time - time_to_first_token
                else:
                    raise e

    print(f"{indent}[X] Failed after max retries in {run_info}.")
    if len(full_text_list) > 0:
        total_time = time.perf_counter() - t0
        return "".join(full_text_list), final_usage, total_time - time_to_first_token
    else:
        return "Error", None, time.perf_counter() - t0 - time_to_first_token


def write_results_to_file(input: list[str],
                          computed_output: list[str],
                          expected_output: list[str],
                          info: dict[str, int],  # time is in ms
                          output_file: str,
                          config: RunConfig | None = None) -> None:
    """
    Write the results out after comparing the computed and expected MathML outputs.
    If show_normalized = True, computed_output and expected_output should both be MathML (=> input is braille)
    """
    usage_info = str(info)[1:-1].replace("'", "").replace(": ", "=")
    print(f"Generated {len(computed_output)} outputs. Stats: {usage_info}ms")
    is_mathml_output = expected_output[0].startswith('<math')
    if not isinstance(computed_output, list):
        print(f"Error: Computed output is a {type(computed_output)},\
              {len(computed_output) if isinstance(computed_output, list) else 0} items")
        return
    if is_mathml_output and not computed_output[0].startswith('<math'):
        print("Computed output does not appear to be MathML--first 5 lines:\n", computed_output[:5])
        return
    if not is_mathml_output and not re.match('[\u2800-\u28ff]', computed_output[0][0]):
        print("Computed output does not appear to be MathML--last 5 lines:\n", computed_output[len(computed_output)-5:])
        return

    # initial MathCAT
    setMathCATPreferences({})

    with open(output_file, "w", encoding="utf-8") as f:
        # Write variable values from main() at the start
        if config:
            # Write config to file with # prefix on each line
            config_str = config.print_config()
            for line in config_str.split('\n'):
                if line.strip():  # Skip empty lines
                    f.write(f"# {line}\n")
            f.write("#\n")

        match_count = 0
        f.write(f"# {len(computed_output)} items. "
                f"Usage info: {usage_info}ms, "
                f"TPS={(1000 * info['time']/info['candidates'])}.2f\n#\n")
        if is_mathml_output:
            f.write("\n# NOT Normalized MathML\n")
        f.write("# Match | Test Input | Expected | Computed\n")
        for tests, computed, expected in zip(input, computed_output, expected_output):
            try:
                if is_mathml_output:
                    checked = areCanonicallyEqual(expected, computed)
                else:
                    checked = CanonicalResults(expected.strip() == computed.strip(), "", "")
                if checked.isEqual:
                    match_count += 1
            except Exception:
                checked = CanonicalResults(False, "", "")
            match = "✓" if checked.isEqual else "✗"
            f.write(f"{match} | {tests} | {expected} | {computed}\n")
        if is_mathml_output:
            f.write("\n#===========\n")
            f.write("\n# Normalized MathML\n")
            f.write("Match | Test Input | Expected | Computed\n")
            for tests, computed, expected in zip(input, computed_output, expected_output):
                try:
                    checked = areCanonicallyEqual(expected, computed)
                except Exception as e:
                    print(f"areCanonicallyEqual error message:\n{e}", file=sys.stderr)
                    checked = CanonicalResults(False, expected, '<--bad MathML-->' + computed)

                match = "✓" if checked.isEqual else "✗"
                f.write(f"{match} | {tests} | {checked.canonicalOriginal} | {checked.canonicalComputed}\n")

        f.write(f"# Matches: {match_count} out of {len(computed_output)}: {(match_count/len(computed_output)*100):.0f}%.")
        print(f"Matches: {match_count} out of {len(computed_output)}: {(match_count/len(computed_output)*100):.0f}%. "
              f"Results written to {output_file}. ")


def readMatchingFiles(braille_path: str, mathml_path: str) -> tuple[list[str], list[str]]:
    """
    Reads lines from two files or directories and returns tuple of (braille_lines, mathml_lines).
    If directories are provided, matches files by base name (without extension) and combines all pairs.
    """
    braille_lines = []
    mathml_lines = []

    # Check if paths are directories
    if os.path.isdir(braille_path) and os.path.isdir(mathml_path):
        # Iterate through braille files and read matching pairs directly
        for filename in os.listdir(braille_path):
            if filename.endswith('.brls'):
                base_name = os.path.splitext(filename)[0]
                braille_file = os.path.join(braille_path, filename)
                mathml_file = os.path.join(mathml_path, base_name + '.mmls')

                # If matching mathml file exists, read both files
                if os.path.exists(mathml_file):
                    with open(braille_file, "r", encoding="utf-8") as f:
                        braille_lines.extend(f.read().splitlines())
                    with open(mathml_file, "r", encoding="utf-8") as f:
                        mathml_lines.extend(f.read().splitlines())
    else:
        # Handle as files (original behavior)
        with open(braille_path, "r", encoding="utf-8") as f:
            braille_lines = f.read().splitlines()
        with open(mathml_path, "r", encoding="utf-8") as f:
            mathml_lines = f.read().splitlines()

    return braille_lines, mathml_lines


def generateExamples(braille_path, mathml_path) -> str:
    """
    Zips lines from two files or directories and returns a string of the form "braille | mathml\n".
    If directories are provided, matches files by base name (without extension) and combines all pairs.
    """
    try:
        braille_lines, mathml_lines = readMatchingFiles(braille_path, mathml_path)

        # Zip and format
        result = ""
        for brl, mml in zip(braille_lines, mathml_lines):
            # .strip() removes the trailing newline character for cleaner output
            quoted_mml = mml.strip().replace('"', '\\"')
            result += f'"{brl.strip()} | {quoted_mml}\\n"\n'

        return result

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""


def getInstructionsProlog(gen_braille: bool, braille_code: str) -> str:
    """
    Returns the instructions prolog based on whether generating braille or MathML.

    Args:
        gen_braille: If True, generates braille from MathML. If False, generates MathML from braille.
        braille_code: The braille code to use (e.g., "Nemeth", "UEB")

    Returns:
        The instructions prolog string
    """
    if gen_braille:
        return (
            f"You are an expert braille translator specializing in {braille_code} braille. "
            "The user will provide a python list of strings, "
            "where each string is composed of represents math encoded in MathML "
            "(e.g., <math><mi>f</mi><mrow><mo>(</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>)</mo></mrow></math>). "
            f"Your task is to translate each MathML expression into valid {braille_code} braille "
            "using only Unicode braille characters. "
            "For each MathML input, output ONLY the raw Unicode braille characters. (e.g., '⠋⠷⠹⠂⠌⠆⠼⠾')"
            "It is important to pay attention to generating Unicode braille spaces when needed in the braille. "
            "It is also important to pay attention when to generate "
            f"{'the number sign indicator ⠼ and the English letter indicator' if braille_code == 'Nemeth' else
               'grade 1 indicators ⠰, grade 1 word indicators ⠰⠰, and grade 1 passage indicators ⠰⠰⠰ when appropriate. '
               'These are very common at the start of the translation.'}"
            "Do not include markdown formatting, explanations, or any other text."
            "Add '|next-item|' between each braille output. "
            "Below are some examples of braille/MathML pairs separated by '|' that should be considered the ground truth:\n"
        )
    else:
        return (
            f"You are an expert braille translator specializing in {braille_code} braille. "
            "The user will provide a python list of strings, "
            f"where each string is composed of Unicode {braille_code} braille characters.  "
            "Here are two examples: '⠠⠑⠀⠨⠅⠀⠍⠉⠘⠆' and  '⠎⠊⠝⠀⠷⠨⠹⠾'. "
            "Your task is to translate each exact {braille_code} braille sequence of characters into valid MathML code. "
            "For each braille input, output ONLY the raw MathML string starting with <math> and ending with </math>. "
            "Every element in the MathML must be properly closed and nested. "
            "Do not include markdown formatting, explanations, or any other text. "
            "Do not include any newlines or carriage returns. "
            "Do not include any braille unicode characters in the MathML output. "
            "Add '|next-item|' between each MathML output. "
            "Below are some examples of braille/MathML pairs separated by '|' "
            "that should be considered the ground truth:\n"
        )


def prepare_conversion_config(
    gen_braille: bool,
    braille_code: str,
    n_examples: int | None,
    n_tests: int | None,
    batch_size: int
) -> tuple[RunConfig, str, list[str], list[str], str, str]:
    """
    Prepare configuration and data for a conversion run.

    Returns:
        Tuple of (config, instructions, test_input, expected_output, model, apiKeyName)
    """
    # Get instructions prolog based on output type
    instructions_prolog = getInstructionsProlog(gen_braille, braille_code)

    # File paths for examples
    example_braille_file = f"RustTestData/{braille_code}.brls"
    example_mathml_file = f"RustTestData/{braille_code}.mmls"
    # example_mathml_file = f"RustTestData/{braille_code}-cnclz.mmls"

    if n_examples is None or n_examples > 0:
        examples = generateExamples(example_braille_file, example_mathml_file)
        n_test_examples = examples.count('\n')
        if n_examples is None:
            n_examples = n_test_examples
        else:
            n_examples = min(n_examples, n_test_examples)
            additional_examples = generateExamples(
                f"example_data/{braille_code.lower()}.brls",
                "example_data/mathml.mmls"
            )
            truncated_additional_examples = additional_examples.splitlines(keepends=True)
            additional_examples = "".join(truncated_additional_examples[:n_examples])
            examples += "\n" + additional_examples
            n_examples += n_test_examples
    else:
        examples = ""

    test_mathml_dir = "test_data/MathML"
    test_braille_dir = f"test_data/{braille_code}"

    # File paths for input - gather lines from directories
    braille, mathml = readMatchingFiles(test_braille_dir, test_mathml_dir)
    if len(braille) != len(mathml):
        print("Error: Number of test inputs does not match number of expected outputs.")
        sys.exit(1)

    # Use n_tests parameter, default to len(mathml) if not provided
    n_tests_actual = min(n_tests, len(mathml)) if n_tests is not None else len(mathml)
    braille = braille[:n_tests_actual]
    mathml = mathml[:n_tests_actual]
    # model = "gemini-3-flash-preview"
    # model = "gemini-2.5-flash"
    # model = "gemini-2.5-flash-lite"
    model = "gemini-2.5-pro"
    model = "gemini-3-pro-preview"
    apiKeyName = "GEMINI_API_KEY"
    apiKeyName = "GEMINI_PAID_API_KEY"

    # GENERATE either braille or MathML
    instructions = instructions_prolog + examples
    test_input, expected = (mathml, braille) if gen_braille else (braille, mathml)

    # Create config
    config = RunConfig(
        braille_code=braille_code,
        gen_braille=gen_braille,
        model=model,
        apiKeyName=apiKeyName,
        batch_size=batch_size,
        n_examples=n_examples,
        instructions=instructions,
        example_braille_file=example_braille_file,
        example_mathml_file=example_mathml_file,
        input_braille_file=test_braille_dir,
        input_mathml_file=test_mathml_dir
    )

    return config, instructions, test_input, expected, model, apiKeyName


def run_conversion(
    config: RunConfig,
    instructions: str,
    test_input: list[str],
    expected: list[str],
    model: str,
    apiKeyName: str,
    batch_size: int
) -> None:
    """
    Run the conversion process to generate braille or MathML.

    Args:
        config: Pre-configured RunConfig object
        instructions: Full instructions string for the model
        test_input: List of input strings to process
        expected: List of expected output strings
        model: Model name to use
        apiKeyName: API key environment variable name
        batch_size: Batch size for processing
    """
    print(f"Using API key: {apiKeyName}")
    print(f"Generating {'braille' if config.gen_braille else 'MathML'} with {config.n_examples} examples, "
          f"{len(test_input)} tests with {model} for {config.braille_code}.")

    try:
        run_info = f"{'to-' if config.gen_braille else 'from-'}{config.braille_code}"
        computed, total_tokens, total_generation_time = convert_input_with_model(
            instructions, test_input, model, apiKeyName, batch_size, run_info
        )
        if computed is None:
            computed = []
        total_tokens['time'] = round(1000 * total_generation_time)  # ms -- needs to be an int
        output_filename = (
            f"{'to-' if config.gen_braille else 'from-'}{config.braille_code}-{model}-"
            f"{config.n_examples}exs-{len(test_input)}tests.txt"
        )
        write_results_to_file(test_input, computed, expected, total_tokens,
                              output_filename, config=config)
    except Exception as e:
        print(f"Conversion error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate braille or MathML using Gemini API',
        epilog='''
Examples:
  # Generate MathML from Nemeth braille, use 100 additional examples and 200 tests:
  python use_gemini.py mathml Nemeth -e 100 -t 200

  # Generate braille from MathML using UEB, use all examples and all tests:
  python use_gemini.py braille UEB -e 9999 -t -1

  # Generate MathML from Nemeth braille, use only rust examples and 50 tests with batch size 40:
  python use_gemini.py mathml Nemeth -e -1 -t 50 -b 40

Note: Requires GEMINI_API_KEY or GEMINI_PAID_API_KEY environment variable.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-e', '--examples', type=int, required=True,
                        help=('Number of examples to use. A negative number means use all available examples.'))
    parser.add_argument('-t', '--tests', type=int, required=True,
                        help=('Number of tests to process. A negative number means use all available tests.'))
    parser.add_argument('-b', '--batch-size', type=int, default=80,
                        help='Batch size for processing (default: 80).')
    parser.add_argument('--config', nargs='*', metavar='CONFIG',
                        help='Select configurations to run (case-insensitive).\n'
                             'Options: to-nemeth, to-ueb, from-nemeth, from-ueb.\n'
                             'If not specified, all configurations are run.')

    args = parser.parse_args()

    # Convert negative numbers to None (meaning "all")
    n_examples = None if args.examples < 0 else args.examples
    n_tests = None if args.tests < 0 else args.tests

    # Map configuration strings to conversion parameters (case-insensitive)
    # Map: (gen_braille, braille_code)
    config_map = {
        'to-nemeth': (True, 'Nemeth'),
        'to-ueb': (True, 'UEB'),
        'from-nemeth': (False, 'Nemeth'),
        'from-ueb': (False, 'UEB'),
    }

    # All possible configurations
    all_conversion_params = [
        (True, 'Nemeth'),
        (True, 'UEB'),
        (False, 'Nemeth'),
        (False, 'UEB'),
    ]

    # Filter configurations based on provided arguments (case-insensitive)
    selected_configs = []
    if args.config:
        # Normalize to lowercase for case-insensitive matching
        provided_configs = [c.lower() for c in args.config]
        valid_configs = {k.lower(): v for k, v in config_map.items()}

        invalid_configs = []
        for config_str in provided_configs:
            if config_str in valid_configs:
                if valid_configs[config_str] not in selected_configs:
                    selected_configs.append(valid_configs[config_str])
            else:
                invalid_configs.append(config_str)

        if invalid_configs:
            print(f"Error: Invalid configuration(s): {', '.join(invalid_configs)}")
            print(f"Valid options are: {', '.join(config_map.keys())}")
            sys.exit(1)

        if not selected_configs:
            print("Error: No valid configurations selected.")
            sys.exit(1)
    else:
        # No configs specified, use all
        selected_configs = all_conversion_params

    conversion_params = selected_configs

    # Prepare all configurations before asking for confirmation
    print("\n=== Preparing Configurations ===")
    configs_data = []

    for gen_braille, braille_code in conversion_params:
        try:
            config_data = prepare_conversion_config(
                gen_braille, braille_code, n_examples, n_tests, args.batch_size
            )
            configs_data.append(config_data)
        except Exception as e:
            print(f"Error preparing config for {braille_code} ({'braille' if gen_braille else 'MathML'}): {e}")
            sys.exit(1)

    # Display first configuration
    print("\n=== Full Configuration ===")
    config, instructions, test_input, expected, model, apiKeyName = configs_data[0]
    conversion_type = f"{'Generate Braille' if config.gen_braille else 'Generate MathML'} ({config.braille_code})"
    print(f"\n--- Configuration 1/{len(configs_data)}: {conversion_type} ---")
    config_str = config.print_config(n_tests=len(test_input), short=True)
    print(config_str)

    # Ask for confirmation once
    print("\n=== Confirmation ===")
    print("Is this correct? (y/yes to proceed, anything else to exit): ", end='', flush=True)
    response = input().strip().lower()
    confirmed = response in ('y', 'yes')
    if not confirmed:
        print("Exiting without processing.")
        sys.exit(0)

    # Create threads for each conversion
    threads = []
    for config, instructions, test_input, expected, model, apiKeyName in configs_data:
        thread = threading.Thread(
            target=run_conversion,
            args=(config, instructions, test_input, expected, model, apiKeyName, args.batch_size)
        )
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
