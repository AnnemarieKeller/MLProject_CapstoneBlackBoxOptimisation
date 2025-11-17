# weekly_data.py
import os
import numpy as np
BASE_FUNC_FOLDER = "data/original/function_{functionNo}"
BASE_UPDATES_FOLDER = "data/weeklyAddition/week{weekNo}SubmissionProcessed"

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
    - function_no: int, e.g., 1
    - week: int, e.g., 4
    
    Returns:
    - list of numpy arrays: combined inputs
    """
    
    base_func_folder = BASE_FUNC_FOLDER.format(functionNo=functionNo)
    updates_folder = BASE_UPDATES_FOLDER.format(weekNo=weekNo)
    
    # Load initial inputs
    initial_file = os.path.join(base_func_folder, "inputs.npy")
    initial_inputs = list(np.load(initial_file, allow_pickle=True))
    
    # Load weekly update inputs
    weekly_file = os.path.join(updates_folder, f"function_{function_no}_inputs.npy")
    weekly_inputs = list(np.load(weekly_file, allow_pickle=True))
    
    # Combine (cumulative)
    combined_inputs = initial_inputs + weekly_inputs
    return combined_inputs

def get_weekly_outputs(functionNo, weekNo):
    """
    Combine initial outputs from the original data with the weekly update for a given week.
    
    Parameters:
    - function_no: int
    - week: int
    
    Returns:
    - list of numpy arrays: combined outputs
    """
    
    base_func_folder = BASE_FUNC_FOLDER.format(functionNo=functionNo)
    updates_folder = BASE_UPDATES_FOLDER.format(weekNo=weekNo)
    
    # Load initial outputs
    initial_file = os.path.join(base_func_folder, "outputs.npy")
    initial_outputs = list(np.load(initial_file, allow_pickle=True))
    
    # Load weekly update outputs
    weekly_file = os.path.join(updates_folder, f"function_{function_no}_outputs.npy")
    weekly_outputs = list(np.load(weekly_file, allow_pickle=True))
    
    # Combine (cumulative)
    combined_outputs = initial_outputs + weekly_outputs
    return combined_outputs
