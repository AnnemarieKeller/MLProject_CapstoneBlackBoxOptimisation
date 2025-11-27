from scripts.utils.weekly_data import get_weekly_inputs, get_weekly_outputs
import numpy as np
def generate_data(functionNo,weekNo):
    """ Generates the X and y needed to fit/train the model
    from the original and weekly inputs for one function for the week specified
    """
    inputs = get_weekly_inputs(functionNo=functionNo,weekNo = weekNo)
    outputs = get_weekly_outputs(functionNo=functionNo,weekNo = weekNo)
    print(len(outputs))
    X_train = np.array(inputs)
    y_train = np.array(outputs)
    return X_train,y_train
