# Bayesian Optimization

This is the final group project for the course Computational Statistics (STAT 545) in Purdue University. 
The project was completed in a group size of 3. Two other group members are: Mengjie Chen, and Shishang Wu.

In this project, Bayesian Optimization (BO) and the random search method were implemented and compared for tuning one hyperparameter.
Both the two algorithms were implemented to maximize the function `f(x)`.

The surrogate function and the acquisition function used in the implementation of BO were Gaussian Process (GP) Regression and Expected Information (EI), respectively.

For more than one hyperparameters, the existing package 
[rBayesianOptimization](https://cran.r-project.org/web/packages/rBayesianOptimization/index.html)
was applied on tuning hyperparameters of XGboost to give a general view of results.

## Algorithm for Bayesian Optimization
![Algorithm for Bayesian Optimization](https://github.com/tan263/Bayesian-Optimization/blob/master/BO_algorithm.png)


## Algorithm for Random Search
1. Initialize a point `x0` from `runif(1)`
2. Obtain a new point `x1` from the interval around the current point `x0` with radius `r` by `runif(1, min=max(c(x0-r,0)), max=min(c(x0+r),1))`. The default setting of `r` is 0.333.
3. If `f(x1)>f(x0)`, then update `x0` by `x0 <- x1`. 
3. Repeat step 2 and 3 until the iteration number attains `iter_max`.

## Examples (One Hyperparameter)

#### A Toy Example
A toy example was given to examine the implemented BO algorithm by maximize 
![equation](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cexp%28-%2810x%20-%202%29%5E2%29%20&plus;%20%5Cexp%28-%5Cfrac%7B%2810x%20-%206%29%5E2%7D%7B10%7D%29%20&plus;%20%5Cfrac%7B1%7D%7B%2810x%29%5E2%20&plus;%201%7D)

Plot of predicted f(x) and the acquisition function at iteration = 1:
![Plot of predicted f(x) and the acquisition function at iteration = 1](https://github.com/tan263/Bayesian-Optimization/blob/master/toy_iter_1.png)

Plot of predicted f(x) and the acquisition function at iteration = 15
![Plot of predicted f(x) and the acquisition function at iteration = 15](https://github.com/tan263/Bayesian-Optimization/blob/master/toy_iter_15.png)

Comparison of Results by Bayesian Optimization and Random Search
![Evolution of max f(x) by BO and Random Search](https://github.com/tan263/Bayesian-Optimization/blob/master/toy_compare.png)

#### Hyperparameter Tuning on a XGboost model 

The algorithm was examined by hyperparameter tuning on a XGboost model on the 
[EMPRES](https://www.kaggle.com/tentotheminus9/empres-global-animal-disease-surveillance) 
dataset for predicting if an animal disease is disease-leading to human.

Plot of predicted accuracy and the acquisition function at iteration = 1
![Plot of predicted accuracy and the acquisition function at iteration = 1](https://github.com/tan263/Bayesian-Optimization/blob/master/1dim_iter_1.png)

Plot of predicted accuracy and the acquisition function at iteration = 10
![Plot of predicted accuracy and the acquisition function at iteration = 10](https://github.com/tan263/Bayesian-Optimization/blob/master/1dim_iter_10.png)

Comparison of Results by Bayesian Optimization and Random Search
![Evolution of maximzed predicted accuracy by BO and Random Search](https://github.com/tan263/Bayesian-Optimization/blob/master/1dim_compare.png)

### Example (More Than One Hyperparameters)
For more than one hyperparameters, the package 
[rBayesianOptimization](https://cran.r-project.org/web/packages/rBayesianOptimization/index.html) 
was applied on tuning hyperparameters of XGboost. The dataset 
[agaricus.train](https://www.rdocumentation.org/packages/xgboost/versions/0.90.0.2/topics/agaricus.train), 
which was the training part from [the mushroom data set](https://archive.ics.uci.edu/ml/datasets/mushroom) 
originally by UCI Machine Learning Repository, was used in this example.

Evolution of maximzed predicted accuracy by BO
![Evolution of maximzed predicted accuracy by BO](https://github.com/tan263/Bayesian-Optimization/blob/master/multi_evolution.png)
