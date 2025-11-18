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
    Combine initial inputs from the original data with the weekly update for a given week.
    
    Parameters:
    - functionNo: int, e.g., 1
    - weekNo: int, e.g., 4
    
    Returns:
    - list of numpy arrays: combined inputs
    """
    
    base_func_folder = BASE_FUNC_FOLDER.format(functionNo=functionNo)
    updates_folder = BASE_UPDATES_FOLDER.format(weekNo=weekNo)
    
    # Load initial inputs
    initial_file = os.path.join(base_func_folder, "initial_inputs.npy")
    initial_inputs = [list(map(float, x)) for x in np.load(initial_file, allow_pickle=True)]
    
     # --- Load weekly updates ---
    
    weekly_file = os.path.join(updates_folder, "inputs.txt")
    with open(weekly_file, "r") as f:
        content = f.read().strip()
    
    # Split into separate sets
    raw_sets = re.split(r'\]\s*\[', content)
    raw_sets = [s.strip('[]') for s in raw_sets]
    
    all_sets = []
    for s in raw_sets:
        # Extract numbers inside array(...) and convert to plain lists
        arrays_raw = re.findall(r'array\((.*?)\)', s, re.DOTALL)
        instance_set = [list(ast.literal_eval(a)) for a in arrays_raw]
        all_sets.append(instance_set)
    
    
    func_weekly_data = [week_update[functionNo - 1] for week_update in all_sets]
    
   
    combined_inputs = initial_inputs + func_weekly_data
    
    return combined_inputs
def get_weekly_outputs(functionNo, weekNo):
    """
    Combine initial outputs with weekly updates for a given function.
    Returns a flat list of floats, one per sample.
    """
    base_func_folder = BASE_FUNC_FOLDER.format(functionNo=functionNo)
    updates_folder = BASE_UPDATES_FOLDER.format(weekNo=weekNo)

    # --- Load initial outputs ---
    initial_file = os.path.join(base_func_folder, "initial_outputs.npy")
    if not os.path.exists(initial_file):
        raise FileNotFoundError(f"Initial outputs not found: {initial_file}")
    
    raw_initial = np.load(initial_file, allow_pickle=True)
    
    # Ensure everything is a scalar float
  # Flatten initial outputs
    flat_initial = []
    for x in raw_initial:
       if isinstance(x, list) or isinstance(x, np.ndarray):
          flat_initial.extend(x)  # use extend, not append
       else:
          flat_initial.append(x)
    
    # --- Load weekly outputs ---
    weekly_file = os.path.join(updates_folder, "outputs.txt")
    if not os.path.exists(weekly_file):
        raise FileNotFoundError(f"Weekly outputs not found: {weekly_file}")
    
    func_weekly_outputs = []

    with open(weekly_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    first_line = lines[0]
    
    if first_line.startswith('[') and 'np.float64' in first_line:
        # New format: each line = one function
        if functionNo > len(lines):
            raise IndexError(f"Function {functionNo} not found in weekly outputs")
        line_for_function = lines[functionNo - 1]
        cleaned = line_for_function.replace('np.float64(', '').replace(')', '')
        func_weekly_outputs.extend([float(v) for v in ast.literal_eval(f"[{cleaned}]")])
    else:
        # Old format: one line contains all functions
        for line in lines:
            cleaned = line.replace('np.float64(', '').replace(')', '')
            values = ast.literal_eval(f"[{cleaned}]")
            func_weekly_outputs.append(float(values[functionNo - 1]))

    # Combine initial + weekly outputs into **flat list of floats**
    combined_outputs = flat_initial + func_weekly_outputs
    
    return combined_outputs




