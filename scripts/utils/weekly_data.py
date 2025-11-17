# data_utils.py
import os
import numpy as np

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

import numpy as np
import os

def get_weekly_inputs(func_folder, weekly_base_folder, week):
    """
    Combine the initial function input with the weekly update for a given week.

    Args:
        func_folder (str): Folder containing the initial function input (initial_input.npy)
        weekly_base_folder (str): Base folder containing weekly update folders like week1update, week2update, ...
        week (int): Week number to read

    Returns:
        list of numpy arrays: combined inputs
    """
    # Load initial input
    initial_input = np.load(os.path.join(func_folder, "initial_input.npy"), allow_pickle=True)

    # Load weekly input
    weekly_folder = os.path.join(weekly_base_folder, f"week{week}update")
    weekly_input = np.load(os.path.join(weekly_folder, "inputs.npy"), allow_pickle=True)

    # Combine
    combined = list(initial_input) + list(weekly_input)
    return combined
  
def get_weekly_outputs(weekly_base_folder, week):
    """
    Get the outputs for a given week.

    Args:
        weekly_base_folder (str): Base folder containing weekly update folders like week1update, week2update, ...
        week (int): Week number to read

    Returns:
        list of numpy arrays: weekly outputs
    """
    weekly_folder = os.path.join(weekly_base_folder, f"week{week}update")
    weekly_output = np.load(os.path.join(weekly_folder, "outputs.npy"), allow_pickle=True)
    
    return list(weekly_output)

