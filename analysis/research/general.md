# Gaussian Process Surrogate Framework and GP Health Metrics

This repository implements a **Gaussian Process (GP) surrogate modeling framework** with diagnostic metrics to evaluate the "health" of the surrogate. The design ensures that the GP is **numerically stable, predictive uncertainty is well-calibrated, and hyperparameters are reasonable**, enabling robust Bayesian optimization or surrogate-based analysis.

---

## 1. Candidate Selection and Kernel Choices

**Candidate selection** and **kernel selection** strongly influence GP performance. The choices in this repository are based on established literature:

### Candidate Selection

| Reference | Description | Relevance |
|-----------|-------------|-----------------|
| [Brochu et al., 2010](https://arxiv.org/abs/2006.11135) | Overview of Bayesian optimization and candidate selection strategies. | Provides foundational theory for selecting candidate points in Bayesian optimization and motivates using acquisition functions. |
| [Deshpande et al., 2024](https://www.sciencedirect.com/science/article/pii/S2666827022000640) | Discusses surrogate uncertainty calibration in candidate selection. | Shows that well-calibrated GP uncertainty improves optimization outcomes. |
| [Snoek et al., 2012](https://arxiv.org/abs/1505.02350) | Popular Bayesian optimization strategies, including acquisition functions. | Supports the acquisition strategy used in this framework. |
| [McCantin, 2020](https://rmcantin.github.io/bayesopt/html/bopttheory.html) | Practical guide to Bayesian optimization and surrogate evaluation. | Provides practical guidance for implementing Bayesian optimization loops with GP surrogates. |

### Kernel Selection

| Reference | Description | RELEVANCE |
|-----------|-------------|-----------------|
| [Duvenaud, 2014](https://www.cs.toronto.edu/~duvenaud/cookbook/index.html) | Comprehensive kernel cookbook for GP regression. | Guides selection of kernels that balance flexibility and stability, important for hyperparameter reliability. |
| [Stephenson et al., 2022](https://proceedings.mlr.press/v151/stephenson22a.html) | Discusses advanced kernel choices for stable GP predictions. | Supports reasoning for kernel choices that avoid ill-conditioning and overfitting. |
| [Speekenbrink, Tutorial on GP Regression](https://discovery.ucl.ac.uk/id/eprint/10050029/1/Speekenbrink_Tutorial%20on%20Gaussian%20process%20regression.pdf) | Clear explanation of kernel influence on GP performance. | Provides a beginner-friendly explanation of GP kernels and their effect on predictive variance and mean. |

> These references guided the choice of kernels and candidate selection strategies to ensure robustness, predictive fidelity, and numerical stability.

---

## 2. GP Health Metrics

The GP health framework evaluates surrogate quality before using it for optimization or prediction.  

| Metric / Goal | Description | Reference / Link |
|---------------|-------------|----------------|
| **Kernel Conditioning** | Measures numerical stability of the kernel matrix; high condition number leads to unstable predictions. | Rasmussen & Williams, *GPML* Ch. 2-5 ([link](http://www.gaussianprocess.org/gpml/)) |
| **Predictive Uncertainty** | Average predictive standard deviation; high uncertainty reduces health score. | Probabilistic surrogate modeling by GP ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0951832024001686)) |
| **Log-Marginal Likelihood** | Measures GP fit quality and penalizes complexity; normalized via sigmoid to 0–1. | [Rasmussen & Williams, *GPML*](https://gaussianprocess.org/gpml/chapters/RW.pdf) |
| **Residual Calibration** | Standardized residuals \(z = \frac{y - \mu}{\sigma}\); check normality for predictive calibration. | Kuleshov et al., 2018 ([arXiv](https://arxiv.org/abs/1807.00263)) |
| **Residual Patterns** | Detects autocorrelation in residuals; high correlation indicates missing patterns. | [Rasmussen & Williams, Ch. 5 ]((https://gaussianprocess.org/gpml/chapters/RW.pdf))
| **Hyperparameter Stability** | Checks kernel length-scales and other hyperparameters; extreme values are penalized. | [Duvenaud, 2014] (https://www.researchgate.net/publication/303384310_Automatic_model_construction_with_Gaussian_processes) |
|* SciPy `normaltest` docs: based on D’Agostino and Pearson’s test ([link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html))|

# Gaussian Process Surrogate Framework and GP Health Metrics

This repository implements a **Gaussian Process (GP) surrogate modeling framework** with diagnostic metrics to evaluate the "health" of the surrogate. The design ensures that the GP is **numerically stable, predictive uncertainty is well-calibrated, and hyperparameters are reasonable**, enabling robust Bayesian optimization or surrogate-based analysis.

---

## 1. Candidate Selection and Kernel Choices

**Candidate selection** and **kernel selection** strongly influence GP performance. The choices in this repository are based on established literature:

### Candidate Selection

| Reference | Description | Why it’s relevant |
|-----------|-------------|-----------------|
| [Brochu et al., 2010](https://arxiv.org/abs/2006.11135) | Overview of Bayesian optimization and candidate selection strategies. | Provides foundational theory for selecting candidate points in Bayesian optimization and motivates using acquisition functions. |
| [Deshpande et al., 2024](https://www.sciencedirect.com/science/article/pii/S2666827022000640) | Discusses surrogate uncertainty calibration in candidate selection. | Shows that well-calibrated GP uncertainty improves optimization outcomes. |
| [Snoek et al., 2012](https://arxiv.org/abs/1505.02350) | Popular Bayesian optimization strategies, including acquisition functions. | Supports the acquisition strategy used in this framework. |
| [McCantin, 2020](https://rmcantin.github.io/bayesopt/html/bopttheory.html) | Practical guide to Bayesian optimization and surrogate evaluation. | Provides practical guidance for implementing Bayesian optimization loops with GP surrogates. |

### Kernel Selection

| Reference | Description | Why it’s relevant |
|-----------|-------------|-----------------|
| [Duvenaud, 2014](https://www.cs.toronto.edu/~duvenaud/cookbook/index.html) | Comprehensive kernel cookbook for GP regression. | Guides selection of kernels that balance flexibility and stability, important for hyperparameter reliability. |
| [Stephenson et al., 2022](https://proceedings.mlr.press/v151/stephenson22a.html) | Discusses advanced kernel choices for stable GP predictions. | Supports reasoning for kernel choices that avoid ill-conditioning and overfitting. |
| [Speekenbrink, Tutorial on GP Regression](https://discovery.ucl.ac.uk/id/eprint/10050029/1/Speekenbrink_Tutorial%20on%20Gaussian%20process%20regression.pdf) | Clear explanation of kernel influence on GP performance. | Provides a beginner-friendly explanation of GP kernels and their effect on predictive variance and mean. |

> These references guided the choice of kernels and candidate selection strategies to ensure robustness, predictive fidelity, and numerical stability.

---

## 2. GP Health Metrics

The GP health framework evaluates surrogate quality before using it for optimization or prediction.  

| Metric / Goal | Description | Reference / Link |
|---------------|-------------|----------------|
| **Kernel Conditioning** | Measures numerical stability of the kernel matrix; high condition number → unstable predictions. | Rasmussen & Williams, *GPML* Ch. 2-5 ([link](http://www.gaussianprocess.org/gpml/)) |
| **Predictive Uncertainty** | Average predictive standard deviation; high uncertainty reduces health score. | Probabilistic surrogate modeling by GP ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0951832024001686)) |
| **Log-Marginal Likelihood** | Measures GP fit quality and penalizes complexity. | Rasmussen & Williams, *GPML* |
| **Residual Calibration** | Standardized residuals should follow a standard normal distribution; checks predictive uncertainty. | Kuleshov et al., 2018 ([arXiv](https://arxiv.org/abs/1807.00263)) |
| **Residual Patterns** | Detects autocorrelation in residuals; high correlation indicates missing patterns. | Rasmussen & Williams, Ch. 5 |
| **Hyperparameter Stability** | Checks kernel length-scales and other hyperparameters; extreme values are penalized. | Duvenaud, 2014 |

> The GP health score combines these metrics into a single value between 0 (poor GP) and 1 (excellent GP), providing an interpretable diagnostic for the surrogate.

---

## 3. Residuals and Calibration Check

To ensure the GP predictive uncertainty is reliable:

* **Residuals:** The difference between true targets and the GP predicted mean.  
* **Standardized residuals:** Residuals divided by predicted standard deviation; unitless and expected to follow a standard normal if the GP is well-calibrated.  
* **Calibration test:** Statistical tests for normality (e.g., D’Agostino-Pearson) verify whether the predictive distribution matches observed deviations.  

**Key Points for Users**

* Standardization accounts for GP uncertainty; no extra scaling is needed.  
* Works on training or test sets; predictive standard deviations are used appropriately.  
* High p-values indicate a well-calibrated GP; low p-values indicate under- or over-confidence in predictions.

**References:**

* Rasmussen & Williams, *Gaussian Processes for Machine Learning*, Ch. 5  
* Kuleshov et al., 2018 ([arXiv link](https://arxiv.org/abs/1807.00263))  
* Gneiting & Katzfuss, *Probabilistic Forecasting*, 2014 ([link](https://www.annualreviews.org/doi/10.1146/annurev-statistics-062713-085831))  


       ┌─────────────────────┐
       │   Train GP on (X,y) │
       └──────────┬──────────┘
                  │
        ┌─────────┴─────────┐
        │ Predictive mean & │
        │  standard deviation│
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │  Compute residuals │
        │    (y - μ(x))      │
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │ Standardize        │
        │ residuals z = r/σ  │
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │ Calibration test   │
        │  z ~ N(0,1)?       │
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │ Residual pattern   │
        │ Hyperparameter     │
        │ stability checks   │
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │ Compute final GP   │
        │     Health Score   │
        └─────────┬─────────┘
                  │
            ┌─────┴─────┐
            │ 0 ←→ 1     │
            │ GP Health  │
            └────────────┘





### GP Health Score Formula

The final **GP health score** combines these metrics as a weighted sum:

```python
GP Health = 0.4 * cond_norm + 0.3 * sigma_norm + 0.3 * loglike_norm
# Advanced
score = 0.5 * calibration + 0.3 * stability + 0.2 * residual_pattern

### Multimodal Optimization Support

In addition to standard Bayesian optimization, our framework is compatible with **multimodal Bayesian optimization** approaches. This is crucial when multiple feasible solutions are needed, e.g., due to resource constraints or practical limitations.  

**Reference:**  
[Wu et al., 2022, Multimodal Bayesian Optimization with Gaussian Processes](https://arxiv.org/pdf/2210.06635)  

**Key Points:**

* The paper extends Gaussian Process-based BO to locate **multiple local and global optima** in expensive-to-evaluate functions.
* It uses the **joint distribution of the objective function and its derivatives** to guide the search for multiple optima.
* Variants of standard acquisition functions are adapted for the **multimodal setting**, which ensures that all relevant solutions can be efficiently explored.
* GP **uncertainty and calibration** play a crucial role, as well-calibrated predictions allow the algorithm to reliably distinguish between promising optima and noise.

**Relevance to this framework:**

* Ensuring **GP health** (numerical stability, predictive uncertainty, calibrated residuals, hyperparameter sanity) is **critical** for multimodal BO because:
  * Poor calibration can mislead acquisition functions, causing missed optima.
  * Unstable kernels can produce spurious or overly confident predictions around false peaks.
  * Residual pattern detection ensures that the GP is not systematically underestimating uncertainty in regions with multiple local optima.
* Your GP health metrics and diagnostics therefore provide **direct support for reliable multimodal optimization**, aligning with the methods described in Wu et al., 2022.
### Function 2: Noise-Aware Expected Improvement (EI)

Our planned implementation of Function 2 focuses on **enhancing the classical Expected Improvement (EI) acquisition function** to properly account for noisy observations.  

**Reference:**  
[Zhou et al., 2023, Noise-Aware Expected Improvement in Bayesian Optimization](https://arxiv.org/abs/2310.05166)  

**Key Points:**

* Traditional EI uses the **best posterior mean** as the incumbent but often ignores the **uncertainty of that incumbent**, especially under noise.
* Zhou et al. introduce a **corrected EI formula** that incorporates the **GP covariance structure**, maintaining analytic tractability while accounting for observation noise.
* The method **specializes to classical EI** in the noise-free case, making it broadly applicable.
* Empirical results demonstrate **improved convergence and reduced cumulative regret** in noisy settings.

**Relevance to this framework:**

* Accurate **GP predictions and calibrated uncertainties** are essential to make this noise-aware EI predictions reliable.
  * Mis-calibrated GP predictive variance can mislead the acquisition function, selecting suboptimal points.
  * Kernel instability or poorly conditioned covariance matrices can corrupt the analytic EI calculation.
* By implementing our **GP health diagnostics**, we ensure that the GP surrogate is sufficiently trustworthy before applying noise-aware EI:
  * Residuals and standardized residuals check that the GP is capturing the underlying function correctly.
  * Hyperparameter sanity checks prevent extreme length scales or variances from biasing the acquisition function.
  * Calibration scores provide confidence that predictive uncertainties are meaningful.
* This ensures that predictions for Function 2 will behave **robustly in noisy or heteroscedastic environments**, aligning with the recommendations of Zhou et al., 2023.




____________++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
GP HEALTH METRICS 
+++++++++++++++++++++++++++++
| Topic / Goal                                                                                              | Reference / Paper (with link)                                                                                                                                                                                                                |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Review of GP surrogate modeling + validation methods / metrics                                            | Probabilistic surrogate modeling by Gaussian process: A review on recent insights in estimation and validation (2024), Reliability Engineering & System Safety — DOI 10.1016/j.ress.2024.110094 ([ScienceDirect][1])                         |
| Conformal‑prediction to build calibrated intervals and evaluate GP surrogate coverage                     | Conformal Approach To Gaussian Process Surrogate Evaluation With Coverage Guarantees (2024, arXiv) ([arXiv][2])                                                                                                                              |
| Recent work on calibration in GP‑based Bayesian optimization: ensuring predictive uncertainty is reliable | Online Calibrated and Conformal Prediction Improves Bayesian Optimization (2024, AISTATS / PMLR) — shows how conformal or calibration methods for GP uncertainty can improve BO performance. ([Proceedings of Machine Learning Research][3]) |

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0951832024001686 "Probabilistic surrogate modeling by Gaussian process: A review on recent insights in estimation and validation - ScienceDirect"
[2]: https://arxiv.org/abs/2401.07733"Conformal Approach To Gaussian Process Surrogate Evaluation With Coverage Guarantees"
[3]: https://proceedings.mlr.press/v238/deshpande24a.html?"Online Calibrated and Conformal Prediction Improves Bayesian Optimization"
| Reference                                                                                                                 | What it covers |relevance                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Probabilistic surrogate modeling by Gaussian process: A review on recent insights in estimation and validation** (2024) | A survey of GP surrogate modeling: covers hyperparameter estimation methods, common pitfalls of GP modelling with small data, and — crucially — validation criteria for predictive distributions (not just mean predictions). Very helpful to justify using multiple diagnostics (variance, interval-quality, hyperparameter sanity checks) as you do. ([ScienceDirect][1])                                  |
| **On the role of model uncertainties in Bayesian optimisation** (2023)                                                    | Investigates how uncertainty calibration of surrogate models (like GP) affects BO performance (regret, ability to find optima). Shows that poorly calibrated uncertainties can degrade BO outcomes, motivating the need for “GP health” diagnostics when doing BO with small datasets. ([Proceedings of Machine Learning Research][2])                                                                       |
| **Surrogate model uncertainty quantification for reliability-based design optimization** (2019)                           | While this is in the context of reliability-based design rather than BO, it emphasizes that surrogate uncertainty (from limited data) must be properly quantified — otherwise decisions based on the surrogate (e.g. optimizing under uncertainty, estimating probabilities) can be unsafe or unreliable. Good source to refer when you argue you propagated or checked GP uncertainty. ([ScienceDirect][3]) |
| **Surrogate model uncertainty quantification for active learning reliability analysis** (2024)                            | Focuses on quantifying surrogate (Kriging / GP) uncertainty when using active learning for reliability assessment. Useful for thinking about criteria to decide whether the GP is “good enough” to trust, or whether more data/points are needed — aligns with your health‑gate logic. ([ScienceDirect][4])                                                                                                  |
| **Variational Bayesian surrogate modelling with application to robust design optimisation** (2024)                        | Presents a Bayesian‑inference approach to surrogate modeling (not “vanilla” GP), but the motivations and discussion of uncertainty quantification, hyperparameter estimation stability, robustness under limited data are very relevant — gives you a view of more advanced / robust surrogate‑model approaches. ([ScienceDirect][5])                                                                        |
| **Uncertainty quantification and propagation in surrogate-based Bayesian inference** (2025)                               | A recent open-access piece on how to propagate surrogate model uncertainty (from limited training budget) through downstream inference or decision tasks. Good reference if you discuss that your GP predictions come with uncertainty and that you account for it in final decisions. ([SpringerLink][6])                                                                                                   |

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0951832024001686 "Probabilistic surrogate modeling by Gaussian process: A review on recent insights in estimation and validation - ScienceDirect"
[2]: https://proceedings.mlr.press/v216/foldager23a.html "On the role of model uncertainties in Bayesian optimisation"
[3]: https://www.sciencedirect.com/science/article/abs/pii/S0951832018305611 "Surrogate model uncertainty quantification for reliability-based design optimization - ScienceDirect"
[4]: https://www.sciencedirect.com/science/article/pii/S1000936124003613 "Surrogate model uncertainty quantification for active learning reliability analysis - ScienceDirect"
[5]: https://www.sciencedirect.com/science/article/pii/S0045782524006789 "Variational Bayesian surrogate modelling with application to robust design optimisation - ScienceDirect"
[6]: https://link.springer.com/article/10.1007/s11222-025-10597-8 "Uncertainty quantification and propagation in surrogate-based Bayesian inference | Statistics and Computing"

---

### 1. **Kernel conditioning**

```python
K = gp.kernel_(X_train_full)
cond = np.linalg.cond(K)
cond_norm = np.clip(1 - np.log10(cond) / 12, 0, 1)
```

* **`K`**: the kernel (covariance) matrix for all your training points. This encodes how correlated your points are.
* **`cond`**: the condition number of `K`. High values → matrix is “ill-conditioned” (numerically unstable), meaning small errors can blow up in predictions.
* **`cond_norm`**: a normalized score between 0 and 1.

Interpretation:

* `cond ≈ 1e12` → very unstable → `cond_norm ≈ 0`
* `cond ≈ 1e4` → okay → `cond_norm ≈ 0.67`
* `cond ≈ 1e2` → very stable → `cond_norm ≈ 0.83`

This term penalizes unstable GPs.

---

### 2. **Predictive uncertainty**

```python
mu_t, sigma_t = gp.predict(X_train_full, return_std=True)
avg_sigma = float(np.mean(sigma_t))
sigma_norm = np.exp(-avg_sigma)
```

* **`sigma_t`**: GP predictive standard deviations at training points.
* **`avg_sigma`**: average uncertainty across all points.
* **`sigma_norm`**: maps high uncertainty → low score. Using `exp(-avg_sigma)` ensures the score is between 0 and 1 and decays smoothly.

Intuition:

* A GP that is “confident” in its predictions gets a higher score.
* High variance → lower `sigma_norm` → penalized GP health.

---

### 3. **Log-marginal likelihood**

```python
loglike = gp.log_marginal_likelihood()
loglike_norm = 1 / (1 + np.exp(-loglike / 100))
```

* Measures how well the GP fits the data while penalizing complexity.
* `loglike_norm` is a sigmoid-like normalization to map arbitrary log-likelihoods into a 0–1 range.
* Smoothly scales bad vs. good fits without extreme values dominating.

---

### 4. **GP Health Score**

```python
gp_health = 0.4 * cond_norm + 0.3 * sigma_norm + 0.3 * loglike_norm
```

* Weighted combination of the three components.
* **Weights**: 0.4 (conditioning), 0.3 (predictive uncertainty), 0.3 (log-likelihood).
* Output: **0 → bad GP**, **1 → excellent GP**.

This single number is your overall “health” check for the GP surrogate.

---

### 5. **Residuals and Calibration**

```python
residuals = y_train - mu
z = residuals / (sigma + 1e-8)
stat, p = normaltest(z)
```

* `residuals`: difference between GP predictions and true targets.
* `z`: standardized residuals (should follow standard normal if GP uncertainty is calibrated).
* `normaltest(z)`: statistical test to check if residuals follow a normal distribution.
* `calibration_score = p` → high p → well-calibrated uncertainty.

---

### 6. **Residual patterns**

```python
ac = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
residual_pattern = 1 - abs(ac)
```

* Measures if residuals are correlated across data points.
* If autocorrelation is high → GP is missing patterns → penalize.

---

### 7. **Hyperparameter Stability**

```python
# Example: length_scale extremes
if ls < 1e-4 or ls > 1e4:
    score *= 0.4
```

* Penalizes kernels with extreme hyperparameter values.
* Reason: extremely small or large length-scales often indicate overfitting or underfitting.

---

### 8. **Composite GP Health Score (Advanced)**

```python
score = 0.5 * calibration + 0.3 * stability + 0.2 * residual_pattern
```

* Combines **calibration**, **stability**, and **residual pattern** into a single diagnostic.
* Weighting reflects priorities: calibration is most important, then stability, then residual patterns.

---



1. Check if the GP is numerically stable → `cond_norm`.
2. Check if predictions are confident → `sigma_norm`.
3. Check if GP fits the data → `loglike_norm`.
4. Check if predictive uncertainty is calibrated → `calibration_score`.
5. Check if residuals show structure → `residual_pattern`.
6. Check if kernel hyperparameters make sense → `stability`.
7. Combine all of the above → overall GP health.

------
---

###  `residuals, sigma = gp_residuals(gp, X_train, y_train)`

* **Residuals:** `y_train - μ(X_train)`
  These are the differences between the true observed values and the GP’s predicted mean. Residuals are a classic diagnostic in regression to check how well a model fits.
* **Sigma (`σ`):** GP predictive standard deviation at each training point, i.e., the model’s uncertainty estimate for its prediction.
* ***Sigma σ-normalization uses exp(-avg_sigma), which maps uncertainty to (0,1]
with smooth decay for high-variance models. Score near 1 indicates a low average uncertainty and expontential decay with a Score towards 0 indicates a high average uncertainty 

**Sources:**

* Rasmussen & Williams, *Gaussian Processes for Machine Learning*, Ch. 2–5 ([link](http://www.gaussianprocess.org/gpml/)) — residuals and predictive uncertainty are the foundation for GP diagnostics.
* GPs naturally give a **predictive distribution** ( \mathcal{N}(\mu(x), \sigma^2(x)) ), so the residuals normalized by σ test whether that distribution is calibrated.

---

###  Standardized residuals: `z = residuals / sigma`

* This converts residuals to a unitless scale: how many “standard deviations” the prediction is off.
* If the GP is **well-calibrated**, then standardized residuals should roughly follow a standard normal: ( z \sim \mathcal{N}(0,1) ).

------

**Sources:**

* Kuleshov et al., *Accurate Uncertainties for Deep Learning Using Calibrated Regression*, 2018 ([arXiv link](https://arxiv.org/abs/1807.00263)) — they explicitly normalize residuals by predictive standard deviation to evaluate calibration.
* Gneiting & Katzfuss, *Probabilistic Forecasting*, 2014 ([link](https://www.annualreviews.org/doi/10.1146/annurev-statistics-062713-085831)) — recommend using standardized residuals or PIT values to assess predictive distribution calibration.

---

###  Normality test: `stat, p = normaltest(z)`

* This tests whether the standardized residuals follow a normal distribution.
* **Null hypothesis:** `z` comes from a normal distribution.
* **p-value:** If `p` is small (e.g., < 0.05), the GP is poorly calibrated — residuals do **not** match the predicted uncertainty.

**Sources:**

* SciPy `normaltest` docs: based on D’Agostino and Pearson’s test ([link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html))
* Calibration literature often uses statistical tests like Shapiro-Wilk, Anderson-Darling, or D’Agostino to check z-values for approximate N(0,1) behavior.

---

###  Calibration score: `calibration_score = np.clip(p, 0, 1)`

* Simply turns the p-value into a 0–1 score (1 = perfectly calibrated, 0 = poorly calibrated).
* This is a convenient way to weight calibration when combining it with other GP health metrics.

---

###  About splitting / training vs. test residuals

* If you compute `residuals = y_train - μ(X_train)`, then σ is already from the GP predictive distribution at those points. You **don’t** need to divide by anything extra — the standardization step is exactly `residual / σ`.
* If you were doing residuals on a separate **test set**, same logic applies, but μ and σ would come from the GP predictions on that test set.
* Some references emphasize doing cross-validation for calibration, but the math of standardized residuals is the same: divide by predicted σ.

**Source:** Rasmussen & Williams, Ch. 5 — “Predictions at training points versus new points” explains how predictive variance already accounts for conditioning on training data.

---

**In short:**

* You calculate `residuals` as the difference between true and predicted mean.
* Dividing by `sigma` gives standardized residuals `z`.
* Normality test on `z` tells you if the GP’s predictive uncertainty matches actual error.
* No additional division or scaling is needed — the GP already provides the predictive standard deviation.


visualize how residuals, sigma, and standardized residuals (`z`) work in Gaussian Process calibration.

---

### **GP Prediction Setup**

```
True value at x_i:      y_i
GP predictive mean:     μ_i
GP predictive std:      σ_i
```

---

### ** Residual**

```
residual_i = y_i - μ_i
```

* This is how far the GP's mean prediction is from the true value.
* Units are the same as the target variable (e.g., kg, accuracy %, etc.).

---

### ** Standardized Residual**

```
z_i = residual_i / σ_i
```

* Divides by the GP's predicted uncertainty at that point.
* Now z_i is **unitless** — it measures how many “σ” the prediction is off.
* If GP is perfect: z_i ~ N(0,1)

---

### ** Interpretation**

| z_i value             | Meaning                                      |     |                                                      |
| --------------------- | -------------------------------------------- | --- | ---------------------------------------------------- |
| z_i ≈ 0               | Prediction matches truth almost exactly      |     |                                                      |
|                       | z_i                                          | < 1 | Within 1 σ — good calibration                        |
|                       | z_i                                          | > 2 | More than 2 σ off — GP may underestimate uncertainty |
| z distribution N(0,1) | GP predictive uncertainty is well-calibrated |     |                                                      |

---

### ** Visual Diagram**

```
y_i  ┤
     │      *
     │     *
     │    *
μ_i  ┤---*------------------> x
σ_i  ┤  (predicted std)
residuals: y_i - μ_i
standardized: z_i = residual / σ_i
```

* Stars (*) show true values scattered around the predicted mean.
* The distance from μ_i shows the residual.
* Dividing by σ_i normalizes this distance to “σ units” — producing z_i.

---

### **Testing Calibration**

* Take all z_i across training/test points.
* Use `normaltest` or another normality test:

  * p ~ 1 → well-calibrated
  * p → 0 → poorly calibrated

                   ┌─────────────────────┐
                   │   Train GP on (X,y) │
                   └─────────┬──────────┘
                             │
             ┌───────────────┴────────────────┐
             │ Predictive mean & std (μ, σ)   │
             └───────────────┬────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Compute residuals:          │
              │ residuals = y - μ           │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Standardize residuals:      │
              │ z = residual / σ            │
              └──────────────┬──────────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │ Check calibration: z ~ N(0,1)     │
           │ - Use normality test (normaltest) │
           │ - High p-value → good calibration │
           └──────────────┬───────────────────-┘
                          │
             ┌────────────┴────────────┐
             │ Check residual patterns │
             │ - autocorrelation of res│
             │ - no strong patterns    │
             └────────────┬────────────┘
                          │
             ┌────────────┴────────────┐
             │ Check hyperparameter     │
             │ stability: kernel params │
             │- extreme values penalized│
             └────────────┬────────────┘
                          │
             ┌────────────┴────────────┐
             │ Compute final GP health  │
             │ score = weighted sum of  │
             │ calibration, stability,  │
             │ residual pattern         │
             └────────────┬────────────┘
                          │
                   ┌──────┴───────┐
                   │ GP health =  │
                   │ 0 (bad) → 1  │
                   │ (excellent)  │
                   └──────────────┘
                 ┌───────────────────┐
                 │   GP Health Score  │
                 │   (0 = bad, 1=good) │
                 └─────────┬─────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
   ┌───────▼───────┐ ┌─────▼─────┐ ┌──────▼──────┐
   │ Calibration   │ │ Hyperparam ││ Residual    │
   │ (50%)         │ │ Stability  ││ Patterns    │
   └───────┬───────┘ │ (30%)      ││ (20%)       │
           │         └────────────┘└─────────────┘
           │
   ┌───────▼────────┐
   │ Standardized   │
   │ residuals z =  │
   │ residuals/sigma│
   │ normaltest(z)  │
   └────────────────┘


This works because, for a **well-calibrated GP**, the predictive distribution covers the true y-values with the predicted σ.


                   ┌─────────────────────┐
                   │   Train GP on (X,y) │
                   └─────────┬──────────┘
                             │
             ┌───────────────┴────────────────┐
             │ Predictive mean & std (μ, σ)   │
             └───────────────┬────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Compute residuals:          │
              │ residuals = y - μ           │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Standardize residuals:      │
              │ z = residual / σ            │
              └──────────────┬──────────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │ Check calibration: z ~ N(0,1)     │
           │ - Use normality test (normaltest) │
           │ - High p-value → good calibration │
           └──────────────┬───────────────────-┘
                          │
             ┌────────────┴────────────┐
             │ Check residual patterns │
             │ - autocorrelation of res│
             │ - no strong patterns    │
             └────────────┬────────────┘
                          │
             ┌────────────┴────────────┐
             │ Check hyperparameter     │
             │ stability: kernel params │
             │- extreme values penalized│
             └────────────┬────────────┘
                          │
             ┌────────────┴────────────┐
             │ Compute final GP health  │
             │ score = weighted sum of  │
             │ calibration, stability,  │
             │ residual pattern         │
             └────────────┬────────────┘
                          │
                   ┌──────┴───────┐
                   │ GP health =  │
                   │ 0 (bad) → 1  │
                   │ (excellent)  │
                   └──────────────┘
                 ┌───────────────────┐
                 │   GP Health Score  │
                 │   (0 = bad, 1=good) │
                 └─────────┬─────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
   ┌───────▼───────┐ ┌─────▼─────┐ ┌──────▼──────┐
   │ Calibration   │ │ Hyperparam ││ Residual    │
   │ (50%)         │ │ Stability  ││ Patterns    │
   └───────┬───────┘ │ (30%)      ││ (20%)       │
           │         └────────────┘└─────────────┘
           │
   ┌───────▼────────┐
   │ Standardized   │
   │ residuals z =  │
   │ residuals/sigma│
   │ normaltest(z)  │
   └────────────────┘


# Benchmark Function Selection and Bayesian Optimization Framework

This repository provides a suite of benchmark functions (Functions 1–8) designed to cover a diverse set of **dimensionalities, noise profiles, and optimization challenges**, reflecting real-world black-box optimization problems. The selection parallels well-known **Bayesian optimization (BO) benchmarks**, including the **2020 Black-Box Optimization (BBO) Challenge**, where algorithms such as **HEBO** (Heteroscedastic Evolutionary Bayesian Optimization) and **Squirrel** demonstrated robust performance across diverse, noisy, and multimodal landscapes.

## Function Categories and Literature Justification

| Function   | Input Dim | Output Dim | Initial Sample Size | Optimization Goal | Sample Applications | Literature / Relevance |
|-----------|-----------|-----------|------------------|-----------------|------------------|----------------------|
| Function 1 | 2 | 1 | 10 | Maximize | Detect likely contamination sources in a 2D area (e.g., radiation). BO tunes detection parameters for reliable identification. | Low-dimensional exploration; surrogate modeling in small datasets. Rasmussen & Williams, 2006 [GPML](http://www.gaussianprocess.org/gpml/) |
| Function 2 | 2 | 1 | 10 | Maximize | Black-box function returning noisy log-likelihood scores. | Noise-aware Expected Improvement (EI) acquisition improves performance under heteroscedastic noise. Zhou et al., 2023 [arXiv:2310.05166](https://arxiv.org/abs/2310.05166) |
| Function 3 | 3 | 1 | 15 | Maximize | Drug discovery testing combinations of three compounds; minimizing side effects via output transformation. | Low-dimensional combinatorial optimization; GP uncertainty propagation. Snoek et al., 2012 [arXiv:1206.2944](https://arxiv.org/abs/1206.2944) |
| Function 4 | 4 | 1 | 30 | Maximize | Optimally placing products across warehouses with expensive ML-model approximations. | Medium-dimensional real-world optimization; surrogate-based design. Kandasamy et al., 2020 [arXiv:1910.06467](https://arxiv.org/abs/1910.06467) |
| Function 5 | 4 | 1 | 20 | Maximize | Optimizing a chemical process with four variables; unimodal with a single peak. | Multimodal Bayesian optimization framework supports locating local/global optima. Wang et al., 2022 [arXiv:2210.06635](https://arxiv.org/pdf/2210.06635) |
| Function 6 | 5 | 1 | 20 | Maximize | Cake recipe optimization across five ingredients; score maximizes taste, consistency, cost, and calories. | Medium-dimensional noisy BO; GP surrogate with calibrated uncertainty. Kandasamy et al., 2020 [arXiv:1910.06467](https://arxiv.org/abs/1910.06467) |
| Function 7 | 6 | 1 | 30 | Maximize | Tuning six ML hyperparameters for model performance (accuracy, F1). | High-dimensional black-box optimization; surrogate-guided search in HEBO. Knudde et al., 2020 BBO Challenge Report [link](https://arxiv.org/abs/2006.11135) |
| Function 8 | 8 | 1 | 40 | Maximize | Optimizing eight-dimensional black-box functions, e.g., ML hyperparameters including learning rate, batch size, layers, dropout, regularization, optimizer, activation, initial weights. | High-dimensional, complex landscape optimization; demonstrates robustness of GP surrogates in high-dimensional search. Kandasamy et al., 2020 [arXiv:1910.06467](https://arxiv.org/abs/1910.06467); BBO 2020 Challenge results [link](https://arxiv.org/abs/2006.11135) |

## Literature Insights

1. **Multimodal Optimization**  
   * Wang et al., 2022 ([arXiv:2210.06635](https://arxiv.org/pdf/2210.06635)) propose a multimodal Bayesian optimization framework using Gaussian processes, which analytically incorporates derivatives to locate multiple optima efficiently. This supports **Function 5**, where exploring the full peak landscape ensures optimal chemical process design.

2. **Noisy Objective Functions**  
   * Zhou et al., 2023 ([arXiv:2310.05166](https://arxiv.org/abs/2310.05166)) highlight that standard EI acquisition may neglect the uncertainty of the incumbent solution, especially in noisy settings. Their covariance-aware EI variant ensures robust exploration and exploitation, informing the design of **Function 2** optimization.

3. **High-Dimensional Optimization**  
   * Functions 6–8 reflect the challenges of tuning multiple hyperparameters or recipe ingredients.  
   * HEBO and Squirrel (BBO 2020 Challenge) show that **adaptive surrogate-based strategies** outperform naive search in high-dimensional and noisy landscapes.  
   * Foundational works:  
     - Snoek et al., 2012 ([arXiv:1206.2944](https://arxiv.org/abs/1206.2944))  
     - Kandasamy et al., 2020 ([arXiv:1910.06467](https://arxiv.org/abs/1910.06467))  

4. **GP Surrogate Health Diagnostics**  
   * Across all functions, Gaussian Process diagnostics—**kernel conditioning, predictive uncertainty calibration, and residual analysis**—ensure reliable surrogate predictions.  
   * Literature: Rasmussen & Williams, 2006 ([GPML](http://www.gaussianprocess.org/gpml/)); Kuleshov et al., 2018 ([arXiv:1807.00263](https://arxiv.org/abs/1807.00263)).

## Summary

> The benchmark functions cover a spectrum from low-dimensional to high-dimensional, noisy, and multimodal landscapes, mirroring real-world black-box optimization challenges. Literature from multimodal BO, noise-aware EI variants, and BBO competitions justifies the **use of Gaussian Process surrogates with health diagnostics** and informs **acquisition function choices**. This ensures a **robust, generalizable, and research-aligned optimization framework** suitable for both synthetic and applied tasks.
