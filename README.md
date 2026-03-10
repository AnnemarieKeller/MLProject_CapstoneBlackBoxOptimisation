# MLProject_Imperial
### Black-Box Bayesian Optimisation Capstone Project

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow/Keras](https://img.shields.io/badge/TensorFlow%2FKeras-Deep%20Learning-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Bayesian%20Optimisation-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

This project was completed as the **Capstone project for the Professional Machine Learning and AI Certificate**.  
The goal was to apply **Bayesian Optimisation techniques** to maximise the outputs of several **unknown black-box functions** under limited query budgets.

The project is inspired by the **NeurIPS 2020 Black-Box Optimization Challenge**, simulating real-world optimisation tasks where the underlying function is unknown and costly to evaluate.

---

# Project Overview

Black-Box Optimisation (BBO) focuses on solving problems where:

- The **objective function is unknown**
- Only **input–output observations** are available
- The number of **function evaluations is limited**

To address this, the project builds **surrogate models** that approximate the unknown functions and guide the selection of new query points.

Each week, new candidate inputs were generated and submitted to a project portal, which returned the evaluated outputs. Over **13 weeks**, the dataset gradually expanded and the optimisation strategy improved.

The project explores the trade-off between:

- **Exploration** – sampling unknown regions  
- **Exploitation** – focusing on areas likely to produce high outputs  

---

# Project Goal

The objective was to:

- **Maximise the output of each black-box function**
- Learn about the **structure of the underlying functions**

Additional goals included:

- Implementing **Bayesian optimisation pipelines**
- Experimenting with **surrogate models**
- Managing and documenting an **end-to-end machine learning project**

---
# Project Run 



- Project can be run from this [notebook](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/notebooks/runfunctions.ipynb)
- Logs will be [here](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/tree/main/analysis/Functions/analysis/data/weeklybestpredictions/logs)
- Automatically generated Run reports will be found [here](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/tree/main/analysis/Functions/reports) (select the runreport name folder given as a parameter) 


---

# Dataset Structure

Each function initially provided:

- **input vectors**
- **corresponding output values**

The dataset expanded weekly with new evaluations.

## Input Characteristics

- Dimensionality varies per function

| Function | Dimensions |
|----------|------------|
| Function 1 | 2D |
| Function 2 | 2D |
| Function 3 | 3D |
| Function 4 | 4D |
| Function 5 | 4D |
| Function 6 | 5D |
| Function 7 | 6D |
| Function 8 | 8D |

## Submission Format

Inputs were submitted in the format:

```
x1-x2-x3-...-xn
```

with **six decimal places**.

Example:

```
0.234567-0.345678-0.456789-0.567890
```

---

# Technical Approach

The project primarily uses **Bayesian Optimisation** with **Gaussian Process regression** as the surrogate model.

The workflow consists of:

1. Fit a surrogate model to existing observations  
2. Use an **acquisition function** to identify promising points  
3. Submit new candidate inputs  
4. Observe outputs  
5. Update the model and repeat

A detailed explanation of the optimisation workflow, modelling choices, and supporting diagrams can be found in the **Concepts and Methodologies Overview**:

- [Diagrams and Technical Justification of Approach](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/analysis/research/general.md)

---
## Hyperparameter Tuning & Strategy Selection

Key **surrogate model parameters** were tuned for each function:

- **Gaussian Process Parameters:** kernel type, length-scale, ν, white noise α, normalization, n_restarts_optimizer  
- **Acquisition Function Parameters:** UCB β/κ, PI settings  
- **Weekly Strategies:** defined how acquisition functions were applied dynamically  

**Strategy Types:**  
- **Dense Exploit:** focus on high-value regions  
- **Global Explore:** explore widely for unknown peaks  
- **Refinement:** fine-tune in promising regions  
- **Mixed / Explore-then-Exploit:** balance exploration and exploitation  

**Summary Table: Hyperparameters & Strategies per Function**

| Function | Name | Kernel | Length-scale | White Noise | Acquisition Params | Strategy |
|----------|------|--------|-------------|------------|-----------------|---------|
| 1 | 2D Contamination | Matern | auto (1e-2–1e6) | No | UCB β | Dense Exploit / Global Explore |
| 2 | 2D Noisy Log-Likelihood | Matern | [1.0,1.0] (1e-2–1e6) | 1e-3 | UCB β / Portfolio | Noisy Explore / Refinement |
| 3 | 3D Drug Combination | Matern | [1.0]*3 (1e-5–1e8) | 1e-6 | PI / UCB β | Refinement |
| 4 | 4D Warehouse Placement | Matern | [1.0]*4 (1e-2–1e6) | 1e-3 | UCB β / EI | Mixed / Explore-then-Exploit |
| 5 | 4D Chemical Yield | Matern | [1.0]*4 (1e-2–1e6) | 1e-6 | UCB β / Portfolio / EI | Global Explore / Explore-then-Exploit |
| 6 | 5D Cake Recipe | Matern | [1.0]*5 (1e-2–1e6) | 1e-3 | UCB κ / PI / EI | Mixed / Local Exploit |
| 7 | 6D ML Hyperparameters | Matern | [1.0]*6 (1e-2–1e6) | 1e-6 | UCB β / Portfolio / EI | Refinement / Explore-then-Exploit |
| 8 | 8D ML Hyperparameters | Matern | [1.0]*8 (1e-2–1e6) | 1e-6 | UCB β / Portfolio / EI | Refinement |

> **Note:** Strategies indicate how acquisition functions were **applied dynamically** each week.

This can be found in the [function config](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/scripts/configs/functionConfig.py) and strategies section of [accquistions](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/1f2536b25a903fb2b36f4e5b7f54a1d7540ae987/scripts/exploration/accquistions.py#L308)

---

# Project Workflow

```
Initial Dataset
      ↓
Train Surrogate Model (Gaussian Process / SVR)
      ↓
Compute Acquisition Function
      ↓
Select Candidate Inputs
      ↓
Submit Inputs to Portal
      ↓
Receive Outputs
      ↓
Update Dataset
      ↓
Repeat
```



# Results and Analysis

Over the course of the **13-week optimisation process**, the models progressively improved their ability to identify high-performing input regions.

Key outcomes:

- Improved understanding of the behaviour of each function  
- Effective exploration of high-dimensional search spaces  
- Demonstration of Bayesian optimisation strategies under limited query budgets  

Detailed results can be found in the weekly reports.

- **Weekly Report**  
[View summarized report](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/analysis/Weekly%20Reports/report.md)

- **Weekly Predictions**  
[View the generated highest predictions per run](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/tree/main/analysis/Functions/analysis/data/weeklybestpredictions)

---

## Documentation

To support transparency and reproducibility of the BBO Capstone Project, the following documentation is provided:

- **Datasheet for Dataset:** Describes the structure, collection process, and intended uses of the weekly input-output datasets for the 8 black-box functions.  
  [View Datasheet](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/datasheet.md)

- **Model Card:** Summarises the optimisation approach used, including strategy, performance, assumptions, limitations, and ethical considerations.  
  [View Model Card](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/modelcard.md)
- **Concepts and Methodologies Overview**  
  Provides an overview of Bayesian optimisation concepts, modelling approaches, and workflow diagrams used in the project.  
  [View Technical Approach](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/analysis/research/general.md)

- **Starting Information**  
  [View Initial Information](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/analysis/Overview/startingpoint.md)

- **Glossary of Terms**  
  [View Glossary](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/main/analysis/research/glossary.md)


---

# Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- SciPy
- TensorFlow
---

# Learning Outcomes

This project provided hands-on experience with:

- Bayesian optimisation techniques  
- Surrogate modelling  
- Exploration vs exploitation strategies  
- High-dimensional optimisation  
- ML project documentation and reproducibility

  # Further Experimentation 
- [LLM Analysis](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/blob/llm_integration/src/readme.md) of PDF reports to suggest adjustments  ( this is meant to be simplified version of current research on the integration of [LARGE LANGUAGE MODELS TO ENHANCE BAYESIAN
OPTIMIZATION](https://proceedings.iclr.cc/paper_files/paper/2024/file/84b8d9fcb4e262fcd429544697e1e720-Paper-Conference.pdf) )
