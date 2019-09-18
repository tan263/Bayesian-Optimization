# Bayesian Optimization

This is the final group project for the course Computational Statistics (STAT 545) in Purdue University. 
The project was completed in a group size of 3. Two other group members are: Mengjie Chen, and Shishang Wu.

In this project, Bayesian Optimization (BO) and the random search method were implemented and compared for tuning one hyperparameter. 
For more than one hyperparameters, the existing package 
[rBayesianOptimization](https://cran.r-project.org/web/packages/rBayesianOptimization/index.html)
was applied on tuning hyperparameters of XGboost to give a general view of results.

## Bayesian Optimization:

### Surrogate function: Gaussian Process (GP) Regression


### Acquisition function: Expected Information (EI)

## Examples (One Hyperparameter)

#### A Toy Example
a toy example was given to examine the implemented BO algorithm. With
this algorithm, hyperparameter tuning was performed for a XGboost model to classify real data.

#### Hyperparameter Tuning on a XGboost model 

The algorithm was examined by hyperparameter tuning on a XGboost model on the 
[EMPRES](https://www.kaggle.com/tentotheminus9/empres-global-animal-disease-surveillance) 
dataset for predicting if an animal disease is disease-leading to human.

### Example (More Than One Hyperparameters)
For more than one hyperparameters, the package 
[rBayesianOptimization](https://cran.r-project.org/web/packages/rBayesianOptimization/index.html) 
was applied on tuning hyperparameters of XGboost. The dataset 
[agaricus.train](https://www.rdocumentation.org/packages/xgboost/versions/0.90.0.2/topics/agaricus.train), 
which was the training part from [the mushroom data set](https://archive.ics.uci.edu/ml/datasets/mushroom) 
originally by UCI Machine Learning Repository, was used in this example.
