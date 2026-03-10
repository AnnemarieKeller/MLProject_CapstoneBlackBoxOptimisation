# Datasheet for BBO Capstone Project Dataset

## 1. Motivation
This dataset was received to support a **Black-Box Optimization (BBO)** task in a capstone project. 
Initial Data was received in a zip file with folders for each of the corresponding functions inputs and outputs as npy format.   
- **Purpose:** To optimize unknown functions across multiple dimensions by submitting candidate inputs and observing outputs.  
- **Task supported:** Sequential optimization of 8 unknown functions over several weeks, evaluating candidate solutions based on output feedback.  

---

## 2. Composition
- **Contents:** Eight separate functions, each with varying input dimensions and initial dataset size:  


### Function Details
| Function | Input Dim | Initial Size |
|----------|-----------|--------------|
| 1        | 2         | 10           |
| 2        | 2         | 10           |
| 3        | 3         | 15           |
| 4        | 4         | 30           |
| 5        | 4         | 20           |
| 6        | 5         | 20           |
| 7        | 6         | 30           |
| 8        | 8         | 40           |
 
- **Format:** Each week contains a set of **submitted inputs** and corresponding **outputs returned by the black-box function**. Data is stored in `.txt` format.  
- **Size:** every week the additional inputs and outputs are added to the existing submissions  
- **Missing data:** None for submitted weeks. Unknown function outputs are only available after submission.

---

## 3. Collection Process
- **Query generation:** Candidate inputs are generated using optimization algorithms (e.g., Bayesian Optimization with GP surrogate).  
- **Strategy:** Inputs are selected each week to explore high-potential regions of the search space, balancing exploration and exploitation.  
- **Time frame:** Weekly submissions over the course of the capstone project. Outputs are received by email after each submission is processed.  

---

## 4. Preprocessing and Uses
- **Transformations:** Inputs for certain functions are normalized before submission. No other preprocessing is applied.  
- **Intended use:**  
  - Evaluating optimization algorithms on unknown black-box functions.  
  - Tracking optimization progress across multiple functions and weeks.  
- **Inappropriate uses:**  
  - Reverse-engineering the underlying function beyond submitted inputs.  
  - Using outputs outside of the weekly submissions context.  

---

## 5. Distribution and Maintenance
- **Availability:** Hosted in the [GitHub repository](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation/tree/architecture/data).  
- **Terms of use:** Open access for educational and research purposes. Attribution required.  
- **Maintainer:** Project author: Annemarie Keller. Updates follow weekly submissions.  
