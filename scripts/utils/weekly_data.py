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

def load_weekly_function_data(function_folders, weekly_inputs_folder, weekly_outputs_folder):
    """
    Load weekly inputs and outputs for multiple functions.

    Parameters:
    - function_folders: list of function folder names ['function_1', ...]
    - weekly_inputs_folder: folder containing weekly input files
    - weekly_outputs_folder: folder containing weekly output files

    Returns:
    - data: dict with structure:
        {
            'function_1': [{'inputs': ..., 'outputs': ...}, ...],
            'function_2': [...],
            ...
        }
    """
    data = {f: [] for f in function_folders}

    # Sort weekly files (assumes naming like week1.txt, week2.txt...)
    weekly_input_files = sorted([f for f in os.listdir(weekly_inputs_folder) if f.endswith('.txt')])
    weekly_output_files = sorted([f for f in os.listdir(weekly_outputs_folder) if f.endswith('.txt')])

    for input_file, output_file in zip(weekly_input_files, weekly_output_files):
        # Read all function inputs for this week
        with open(os.path.join(weekly_inputs_folder, input_file), 'r') as f:
            weekly_inputs = eval(f.read())
        # Read all function outputs for this week
        with open(os.path.join(weekly_outputs_folder, output_file), 'r') as f:
            weekly_outputs = eval(f.read())

        for func_idx, func_name in enumerate(function_folders):
            record = {
                'inputs': weekly_inputs[func_idx],
                'outputs': weekly_outputs[func_idx]
            }
            data[func_name].append(record)

    return data


def flatten_outputs(data, function_folders):
    """
    Convert the data dict into a 2D numpy array suitable for plotting/scoring.
    
    Returns:
    - all_weeks_array: shape (num_weeks, num_functions)
    """
    num_functions = len(function_folders)
    num_weeks = len(next(iter(data.values())))  # assumes all functions have same number of weeks
    all_weeks_array = np.zeros((num_weeks, num_functions))

    for func_idx, func_name in enumerate(function_folders):
        for week_idx, record in enumerate(data[func_name]):
            # Here we take sum of outputs, mean, or first value? Choose your logic
            # Example: mean of outputs
            all_weeks_array[week_idx, func_idx] = np.mean(record['outputs'])

    return all_weeks_array


def get_weekly_inputs(functionNo, weekNo):
    """
    Combine initial inputs with weekly updates for the given function.
    Supports multi-line array definitions (one sample per block).
    """
    import ast, re, numpy as np, os

    # --- Load initial inputs ---
    base_func_folder = BASE_FUNC_FOLDER.format(functionNo=functionNo)
    initial_file = os.path.join(base_func_folder, "initial_inputs.npy")
    initial_inputs = [np.array(x, dtype=float) for x in np.load(initial_file, allow_pickle=True)]

    # --- Load weekly file ---
    updates_folder = BASE_UPDATES_FOLDER.format(weekNo=weekNo)
    weekly_file = os.path.join(updates_folder, "inputs.txt")

    func_weekly_data = []
    block = ""   # will accumulate lines until ] appears

    with open(weekly_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            block += stripped

            # full sample collected
            if stripped.endswith("]"):
                # extract all array(...) inside the block
                arrays_raw = re.findall(r'array\((.*?)\)', block)
                arrays = [np.array(ast.literal_eval(a), dtype=float) for a in arrays_raw]

                if functionNo - 1 < len(arrays):
                    func_weekly_data.append(arrays[functionNo - 1])

                block = ""  # reset

    # --- Combine ---
    return initial_inputs + func_weekly_data


def flatten(x):
    """Recursively flatten nested lists/arrays."""
    for item in x:
        if isinstance(item, (list, tuple, np.ndarray)):
            yield from flatten(item)
        else:
            yield item

import os
import numpy as np
import ast

def flatten(x):
    for item in x:
        if isinstance(item, (list, np.ndarray)):
            yield from flatten(item)
        else:
            yield item

def get_weekly_outputs(functionNo, weekNo):
    import re

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
