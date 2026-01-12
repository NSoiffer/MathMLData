"""
Batch process MathML files to generate braille output using MathCAT.
To specify the source directory, output directory, and braille code, modify the variables in the main() function.
"""
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
import libmathcat_py as libmathcat

sys.stdout.reconfigure(encoding='utf-8')  # in case print statements are used for debugging


def setMathCATPreferences(braille_code):
    try:
        libmathcat.SetRulesDir("C:/Users/neils/MathCAT/Rules")
    except Exception as e:
        sys.exit(f"problem with finding the MathCAT rules: {e}")

    try:
        # libmathcat.SetPreference("BrailleNavHighlight", "Off")
        libmathcat.SetPreference("BrailleCode", braille_code)
    except Exception as e:
        sys.exit(f"problem with setting a preference: {e}")


def setMathMLForMathCAT(mathml: str):
    try:
        libmathcat.SetMathML(mathml)
    except Exception as e:
        raise e


def getSpeech():
    try:
        return libmathcat.GetSpokenText()
    except Exception as e:
        raise e


def getBraille():
    try:
        return libmathcat.GetBraille("")
    except Exception as e:
        raise e


# Configure logging
logging.basicConfig(filename='batch_process.log', level=logging.ERROR)



def ProcessFile(file_path: str, dest_folder: str, config: dict[str, str]) -> str:
    """
    Read all the MathML lines from file_path, convert to braille, and write the braille to dest_folder
    """
    file_path = Path(file_path)
    filename = os.path.basename(file_path)
    (batch_name, filename) = file_path.parts[-2:]
    brailleCode = config["BrailleCode"]
    output_path = os.path.join(dest_folder, brailleCode, batch_name, filename.replace('.mml', '.brl'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        setMathCATPreferences(brailleCode)
    except Exception as e:
        print(f"Can't set rules dir/preference: {e}")
    try:
        with open(file_path, 'r', encoding='utf8') as in_stream:
            with open(output_path, 'w', encoding='utf8') as out_stream:
                for line in in_stream.readlines():
                    try:
                        # print(f'Generating braille for "{line}"')
                        setMathMLForMathCAT(line)
                        braille = getBraille()
                        out_stream.write(braille)
                        out_stream.write("\n")
                    except Exception as e:
                        print(f"\n==== Error in {filename}: setting MathML:  {e}\n MathML={line}")
                        exit(0)

        return output_path
    except Exception as e:
        raise e


def main():
    source_dir = "../SimpleSpeakData"
    source_subdir = "highschool"
    output_dir = "./Braille"

    # Extra settings to pass to workers
    settings = {"BrailleCode": "UEB"}
    max_workers = 24   # set 24 for core 9 ultra

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths: list[str] = []
    for root, dirs, files in os.walk(f"{source_dir}/{source_subdir}"):
        # Add the files list to the all_files list
        file_paths.extend([f"{root}/{f}" for f in files if f.endswith('no-dups.mmls')])

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Pass extra arguments inside executor.submit
        # Syntax: executor.submit(fn, arg1, arg2, arg3...)
        future_to_file = {
            executor.submit(ProcessFile, path, output_dir, settings): path
            for path in file_paths
        }

        with tqdm(total=len(file_paths), desc="Batch Processing") as pbar:
            for future in as_completed(future_to_file):
                try:
                    future.result()
                except Exception as exc:
                    logging.error(f"Error on {future_to_file[future]}: {exc}")
                pbar.update(1)


if __name__ == "__main__":
    main()
