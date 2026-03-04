
# Bayesian Black-Box Optimization Glossary

## Acquisition Function
A function that combines the surrogate model prediction and uncertainty to determine the next point to sample.

## Alpha (α, alpha)
A parameter used in Gaussian Process models to represent noise level or regularization. In some libraries it is used in the kernel or GP settings to control how much noise is assumed in the data. A larger alpha implies higher noise and a smoother model fit, while a smaller alpha implies less noise and a tighter fit.

## Anomaly
Unexpected behavior in predictions or uncertainty such as sudden spikes or collapses.

## Bayesian Optimization
An optimization framework that uses a probabilistic surrogate model to guide efficient sampling of an unknown function.

## Black-Box Function
A function where the internal structure is unknown or expensive to evaluate. Only input and output values are observable.

## Candidate Points
Potential input values evaluated by the acquisition function.

## Convergence
A state where additional queries produce little improvement or new information. Convergence is indicated when the best observed value stops improving over multiple iterations, when the acquisition function selects points with low expected gain, or when the model uncertainty (sigma) becomes consistently low across the search space. Convergence can also be judged by the stability of the Gaussian Process predictions and the absence of new high variance regions.

## Dimensions (dim / D)
The number of input parameters in the search space.dim is used in code.
D is used in mathematical notation.

## Exploration
Sampling points with high σ to reduce uncertainty in the model.

## Exploration–Exploitation Trade-off
The balance between sampling uncertain regions and improving known good regions.

## Exploitation
Sampling points with high μ and low σ to refine the optimum.

## Expected Improvement (EI)
An acquisition function that uses μ and σ to estimate expected improvement over the current best value.

## Gaussian Process (GP)
A probabilistic surrogate model that represents a distribution over possible functions and provides both predictions and uncertainty estimates.  
Learn more:  
• Bayesian Optimization package guide — Gaussian processes and posterior explained (interactive intro) [Basic tour of the Bayesian Optimization package](https://bayesian-optimization.github.io/BayesianOptimization/master/basic-tour.html?utm_source=chatgpt.com)  
• Intuitive Gaussian Process tutorial & code examples [An Intuitive Tutorial to Gaussian Process Regression (GitHub)](https://github.com/jwangjie/Gaussian-Process-Regression-Tutorial?utm_source=chatgpt.com)

## Gaussian Process (GP) Posterior
The conditional distribution over functions obtained after incorporating observed data into a Gaussian Process, providing updated predictions and uncertainty estimates.  
Learn more:  
• Gentle conceptual explanation of GP posterior and predictive distributions (with formulas) [Intro to Gaussian processes and posterior prediction](https://jejjohnson.github.io/gp_model_zoo/intro)  
• University lecture notes on GP regression and posterior inference (PDF slides) [Gaussian Processes and Bayesian Optimization (lecture)](https://tamids.tamu.edu/wp-content/uploads/2020/09/TAMIDS-Tutorial-Rui-Tuo-2020.09.04_Lecture.pdf)

## Global Optimum
The best achievable value of an objective function across the entire input space.  
Learn more:  
• Mathematical overview of finding global optima in optimization problems [Global optimization (Wikipedia)](https://en.wikipedia.org/wiki/Global_optimization)

## Heteroscedasticity 
condition where the variance of errors or noise varies across different levels of the input or output.

## High-Dimensional Space
A search space with many input variables which increases optimization complexity.

## Hyperparameters
Parameters that define the behavior of the surrogate model such as kernel type, length scale, regularization strength, and noise level.

## Information Gain
The reduction in uncertainty gained by sampling new points.

## Initialization
The initial data points used to train the surrogate model before optimization begins.

## Kernel (Covariance Function)
A function that defines similarity between points in the input space and controls the smoothness of the GP or SVR model.

## Kappa (κ, kappa, beta)
A parameter used in the Upper Confidence Bound acquisition function to control the exploration–exploitation trade off. In code it may appear as `kappa` or `beta` depending on the library. A larger value increases exploration by giving more weight to uncertainty, while a smaller value prioritizes exploitation.

## Length Scale
A kernel hyperparameter that determines how rapidly the function is allowed to change across the input space.

## Local Optimum
A best value within a limited region but not globally optimal.
### Log-likelihood  
A measure of how likely the observed data is given the model parameters. In Gaussian Process models it evaluates how well the GP explains the observed outputs, and is used to tune kernel hyperparameters. Higher log-likelihood means the model fits the data better.
### Log marginal likelihood  
Also known as the evidence, the log marginal likelihood measures how probable the observed data is under a given model and its hyperparameters after integrating over all possible functions. In Gaussian Process training it is maximized to select kernel hyperparameters that best explain the data. Higher values indicate a better fit while avoiding overfitting.


## Mean Prediction (μ, mu)
The predicted mean value of the surrogate model at a given input point. In code this is often written as `mu`, while in mathematical notation it is written as μ. In Gaussian Processes, this represents the expected value of the objective function at that point.

## Model Health
An assessment of surrogate model stability including reasonable μ and σ behavior and prediction consistency.

## Neural Network Surrogate (NN Surrogate)
A neural network used as an alternative surrogate model to approximate the black-box function. NN surrogates can handle high-dimensional inputs and complex nonlinear relationships. Unlike Gaussian Processes, neural networks do not natively provide uncertainty estimates, so uncertainty must be approximated or handled separately.

## Noise
A random error or variability in data that is not explained by the model.
A noisy function has large random fluctuations and unpredictable behavior, while a smooth function changes gradually and predictably with inputs.

## Overconfidence
A condition where σ becomes unrealistically small despite limited or poor data.

## Objective Function
The unknown or expensive function being optimized.

## Probability of Improvement (PI)
An acquisition function that uses μ and σ to compute the probability of outperforming the current best result.

## p-value
The probability, under a specified null hypothesis, of observing a result at least as extreme as the one obtained. In surrogate modeling, p-values are often computed from standardized residuals to assess whether observed errors are consistent with the assumed noise model (e.g., Gaussian). Small p-values indicate potential model misfit, miscalibrated uncertainty, or unmodeled structure in the data.

## Redundant Sampling
Sampling points that provide little new information or improvement.

## Residuals
The difference between an observed value \(y_i\) and the surrogate model’s predicted mean \(μ_i\):
\[
r_i = y_i - μ_i
\]
Residuals measure local model error at a given input. Large residuals indicate poor fit, model misspecification, or unmodeled noise.

## Standardized Residuals
Residuals normalized by the predicted uncertainty σ,:
\[
z_i = \frac{y_i - μ_i}{σ_i}
\]
Standardized residuals express error in units of predicted standard deviation and are useful for assessing surrogate calibration and detecting overconfidence.


## Restarts
A technique used in optimization where the search process is restarted from different initial points or random seeds. Restarts help avoid getting stuck in local optima and improve the chance of finding the global optimum. They are especially useful when the search space is complex or multi modal, and when the optimization appears to converge too early.


## Scikit-learn
A Python machine learning library used to implement Gaussian Processes, Support Vector Regression, kernels, and model fitting.

## Sigma (σ, sigma)
The standard deviation of the GP prediction at a given input point. In code this is often written as `sigma`, while in mathematical notation it is written as σ. Sigma represents model uncertainty.

## Support Vector Regression (SVR)
A supervised learning model used as an alternative surrogate model to approximate the black-box function. Unlike Gaussian Processes, SVR does not natively provide uncertainty estimates, so uncertainty must be approximated or handled separately.

## Upper Confidence Bound (UCB)
An acquisition function commonly defined as μ + κσ. It balances exploitation using the mean prediction μ and exploration using the uncertainty σ, with κ controlling the strength of exploration.
## Uncertainty (Variance)
A measure of confidence in a model’s prediction. Variance quantifies how much the predicted output is expected to vary if new data were observed or the sampling process were repeated. In surrogate models, predictive variance is commonly represented by σ², and standard deviation by σ.

## Variance (σ²)
The square of sigma. Variance measures uncertainty in squared units.


Libraries : 
## NumPy
A Python library used for numerical operations, vectorized computations, and array manipulation.

## Pandas
A Python library used for data manipulation, logging, and result analysis.

## SciPy
A Python library used for optimization, statistical functions, and scientific computing.

## Matplotlib
A Python library used for visualizing optimization results and model behavior.

## Global Optimum
The best achievable value across the entire input space.

## Local Optimum
A best value within a limited region but not globally optimal.
