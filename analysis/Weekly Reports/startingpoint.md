| Function   | Input Dim | Output Dim | Initial Sample Size | Optimisation Goal | Sample Applications |
|-----------|-----------|-----------|------------------|-----------------|------------------|
| Function 1 | 2 | 1 | 10 | Maximise | Detect likely contamination sources in a 2D area, e.g., radiation field. Bayesian optimisation tunes detection parameters to reliably identify strong and weak sources. |
| Function 2 | 2 | 1 | 10 | Maximise | Black-box function or ML model returning noisy log-likelihood scores. Bayesian optimisation balances exploration with exploitation to maximise the score and avoid local optima. |
| Function 3 | 3 | 1 | 15 | Maximise | Drug discovery project testing combinations of three compounds. Initial experiments record side effects; optimisation minimises side effects (framed as maximisation via output transformation). |
| Function 4 | 4 | 1 | 30 | Maximise | Optimally placing products across warehouses with high online sales. ML model approximates expensive biweekly calculations to find near-optimal decisions. |
| Function 5 | 4 | 1 | 20 | Maximise | Optimising a chemical process with four variables to maximise yield. Typically unimodal with a single peak. |
| Function 6 | 5 | 1 | 20 | Maximise | Optimising a cake recipe with five ingredients (flour, sugar, eggs, butter, milk). Score is negative; optimisation maximises score by minimising negative contributions from taste, consistency, calories, waste, and cost. |
| Function 7 | 6 | 1 | 30 | Maximise | Tuning six hyperparameters of an ML model (learning rate, regularisation, hidden layers, etc.) to maximise performance metrics like accuracy or F1 score. Treated as a black-box function. |
| Function 8 | 8 | 1 | 40 | Maximise | Optimising an eight-dimensional black-box function. Example: tuning ML model hyperparameters (learning rate, batch size, layers, dropout, regularisation, activation, optimiser, initial weights) to maximise validation accuracy. High-dimensional optimisation is challenging, local maxima are practical targets. |

