# Model Card for BBO Capstone Project Optimisation Approach

## 1. Overview
- **Name:** Capstone Black-Box Optimisation (BBO) Strategy  
- **Type:** Sequential Optimisation via Surrogate Modelling (Gaussian Process-based Bayesian Optimisation)  
- **Version:** 1.0  

---

## 2. Intended Use
- **Primary task:** Optimising unknown black-box functions by sequentially submitting candidate inputs and receiving corresponding outputs.  
- **Suitable use cases:**  
  - Maximising performance metrics of unknown functions.  
  - Exploring high-dimensional parameter spaces efficiently.  
  - Comparing optimisation strategies on black-box benchmarks.  
- **Inappropriate uses:**  
  - Reverse-engineering the underlying function beyond observed outputs.  
  - Applying the model outside of submitted input-output pairs.  

---

## 3. Details
- **Strategy:**  
  - Gaussian Process (GP) surrogate model predicts function outputs.  
  - Candidate inputs selected weekly via acquisition functions (UCB, EI, PI, Thompson Sampling, or Portfolio strategies).  
  - GP kernel types experimented: RBF and Matern.  
- **Workflow across 10 rounds:**  
  1. Receive initial dataset for each function.  
  2. Normalize input data where appropriate.  
  3. Train GP on submitted input-output pairs.  
  4. Predict outputs across candidate inputs to guide next submission.  
  5. Submit new inputs to the black-box function.  
  6. Receive outputs and update GP model.  
  7. Repeat until maximum iterations or convergence.  

---

## 4. Performance
- **Functions optimised:** 8 black-box functions of varying dimensions (2D to 8D).  
- **Metrics:**  
  - Best output achieved per function.  
  - Iterative convergence across submissions.  
- **Results:**  
  - Function 5 (4D chemical yield) reached `Best Output ≈ 4440.52` at input `[0, 1, 1, 1]`.  
  - High-dimensional functions (6D–8D) showed slower convergence but improvements in predicted best outputs each week.  

---

## 5. Assumptions and Limitations
- **Assumptions:**  
  - Functions are smooth enough for GP surrogate modelling to capture trends.  
  - Weekly input-output feedback is accurate.  
  - The search space is bounded (inputs normalized to [0,1]).  
- **Limitations:**  
  - High-dimensional functions may stagnate due to sparse exploration.  
  - GP predictions may underperform if data is highly noisy.  
  - Cannot provide true function expressions, only predicted maxima.  

---

## 6. Ethical Considerations
- **Transparency:** Model decisions and best inputs are logged weekly.  
- **Reproducibility:** Submission and output history allow others to replicate optimisation results.  
- **Real-world adaptation:** Approach is safe for simulation and educational purposes; not for critical systems without domain validation.  

---

## 7. Contact and Maintenance
- **Maintainer:** Annemarie Keller  
- **Repository:** [GitHub BBO Capstone Project](https://github.com/AnnemarieKeller/MLProject_CapstoneBlackBoxOptimisation)  
- **Updates:** Weekly, as new input-output pairs are submitted and GP models retrained.
