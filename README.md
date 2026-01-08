# MLProject_Imperial
This Project is the capstone project for the Professional ML and AI Certificate which utilises the learnt ML concepts to solve a problem 


### Project Summary

### Project Goal 

#### Dependencies used 
* Python,
* Pytorch, 
##### How to run the project 
### Project Overview
The Black-Box Optimization (BBO) capstone challenges us to maximise outputs of unknown functions using limited queries. Each function represents a real-world optimization problem, from hyperparameter tuning to chemical process optimisation. The purpose is to practice data-driven decision-making under uncertainty, combining exploration and exploitation strategies

### Inputs and Outputs
Inputs:  The initial dataset provided 10 inputs and 10 outputs, which serve as the starting point for model training or Bayesian optimisation:

 

A vector of numerical values between 0 and 1.

Dimensionality varies depending on the function:

Function 1 & 2 → 2D

Function 3 → 3D

Function 4 & 5 → 4D

Function 6 → 5D

Function 7 → 6D

Function 8 → 8D

Format for submission: x1-x2-x3-...-xn with six decimal places for each number.

Example for 2D: 0.123456-0.654321

Example for 4D: 0.234567-0.345678-0.456789-0.567890

### Outputs:

A single numeric value representing the function’s performance or “score.”

The higher the output, the better. Some outputs may be transformed (e.g., negative of a cost) so that maximisation is consistent across functions.

Each week after submission of inputs, a new set of input and outputs for each function is received
Challenge Objectives
     Goal: Maximise each function’s output using as few queries as possible.

     Constraints: Limited query budget (increasing from 10 to 22 points), unknown and noisy function surfaces, high-dimensional inputs.

Key Consideration: Balance exploration with exploitation 

 

### Technical Approach
Primarily use Gaussian Process regression to model the unknown functions. Query selection relies on:

UCB (Upper Confidence Bound): Suited for noisy or multi-modal functions, balancing exploration and exploitation.

PI (Probability of Improvement): Suitable for low-noise, unimodal functions.

SVR (Support Vector Regression): Helps approximate non-linear or high-dimensional surfaces when GP uncertainty estimates are unreliable.

| Function | Acquisition Function | Reasoning | SVR Suitability | Kernel |
|----------|--------------------|-----------|----------------|--------|
| 1 – 2D Contamination | UCB | Sparse outputs; exploration crucial to locate non-zero zones. | ☐ | - |
| 2 – 2D Noisy Log-Likelihood | UCB | Noisy and multi-modal; UCB balances exploration and exploitation. |  ☐ | - |
| 3 – 3D Drug Combination | UCB | Multi-modal; avoids getting trapped in local minima, prioritises safe regions. | ☑ | RBF |
| 4 – 4D Warehouse Placement | UCB | High-dimensional, dynamic system; exploration needed to find promising regions. | ☑ | RBF vs Poly |
| 5 – 4D Chemical Yield | PI | Unimodal and smooth; exploitation-focused acquisition quickly finds the peak. | ☑ | RBF vs Linear |
| 6 – 5D Cake Recipe | UCB | Complex, multi-factor landscape; exploration reduces risk of missing high-scoring regions. |☑  | RBF |
| 7 – 6D ML Hyperparameters | UCB | High-dimensional, unknown surface; exploration best option | ☑  | RBF |
| 8 – 8D ML Hyperparameters | UCB | Very high-dimensional; complex interactions among parameters; global exploration needed to find strong local maxima. |☑  | RBF |


## Documentation

To support transparency and reproducibility of the BBO Capstone Project, the following documentation is provided:

- **Datasheet for Dataset:** Describes the structure, collection process, and intended uses of the weekly input-output datasets for the 8 black-box functions.  
  [View Datasheet](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/Datasheet_BBO_Capstone.md)

- **Model Card:** Summarises the optimisation approach used, including strategy, performance, assumptions, limitations, and ethical considerations.  
  [View Model Card](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/ModelCard_BBO_Capstone.md)



