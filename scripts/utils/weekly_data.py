# weekly_data.py
import os
import numpy as np
import ast, re
# -----------------------------
# --- Detect repo root dynamically ---

# REPO_PATH = os.path.dirname(os.path.abspath(__file__))  # folder containing this script
# Or one level up:
# REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

BASE_FUNC_FOLDER = os.path.join(REPO_PATH, "data/original/function_{functionNo}")
BASE_UPDATES_FOLDER = os.path.join(REPO_PATH, "data/weeklyAddition/week{weekNo}SubmissionProcessed")


BASE_FUNC_FOLDER = os.path.join(REPO_PATH, "data/original/function_{functionNo}")
BASE_UPDATES_FOLDER = os.path.join(REPO_PATH, "data/weeklyAddition/week{weekNo}SubmissionProcessed")



def get_weekly_inputs(functionNo, weekNo):
    """
    Load initial inputs and append weekly update inputs.
    Handles multi-line array blocks safely.
    """


    # Load initial inputs
    base_func_folder = BASE_FUNC_FOLDER.format(functionNo=functionNo)
    initial_file = os.path.join(base_func_folder, "initial_inputs.npy")
    initial_inputs = [np.array(x, dtype=float) for x in np.load(initial_file, allow_pickle=True)]

    # Load weekly data
    updates_folder = BASE_UPDATES_FOLDER.format(weekNo=weekNo)
    weekly_file = os.path.join(updates_folder, "inputs.txt")

    func_weekly_data = []
    block = ""

    with open(weekly_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            block += line

            # block ends when we hit a closing bracket
            if line.endswith("]"):
                # extract all array(...) entries in this block
                arrays_raw = re.findall(r'array\((.*?)\)', block, flags=re.DOTALL)
                arrays = [np.array(ast.literal_eval(a), dtype=float) for a in arrays_raw]

                # pick the correct function data
                if functionNo - 1 < len(arrays):
                    func_weekly_data.append(arrays[functionNo - 1])

                block = ""

    return initial_inputs + func_weekly_data



def get_weekly_outputs(functionNo, weekNo):
   

    # --- Load initial outputs ---
    base_func_folder = BASE_FUNC_FOLDER.format(functionNo=functionNo)
    initial_file = os.path.join(base_func_folder, "initial_outputs.npy")
    raw_initial = np.load(initial_file, allow_pickle=True)
    flat_initial = np.array(list(flatten(raw_initial)), dtype=float)

    # --- Load weekly outputs ---
    updates_folder = BASE_UPDATES_FOLDER.format(weekNo=weekNo)
    weekly_file = os.path.join(updates_folder, "outputs.txt")

    all_weeks_array = []
    current_line = ""
    with open(weekly_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            current_line += stripped
            if stripped.endswith("]"):
                # --- Remove np.float64(...) ---
                cleaned_line = re.sub(r"np\.float64\((.*?)\)", r"\1", current_line)
                all_weeks_array.append(ast.literal_eval(cleaned_line))
                current_line = ""

    all_weeks_array = np.array(all_weeks_array, dtype=float)
    weekly_values = all_weeks_array[:, functionNo - 1]

    combined_outputs = np.concatenate([flat_initial, weekly_values])

    print("Initial count:", len(flat_initial))
    print("Weekly count:", len(weekly_values))
    print("Combined:", len(combined_outputs))

    return combined_outputs


def flatten(x):
    """Recursively flatten nested lists/arrays."""
    for item in x:
        if isinstance(item, (list, tuple, np.ndarray)):
            yield from flatten(item)
        else:
            yield item

