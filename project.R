library('GPfit')

### Optipization by random search -- uniform distribution (one hyperparameter)
## r: radius of searching for next point
## epsilon: for finding the smallest point that attains max f(x)

random_search = function(f, max_iter, r=0.333, epsilon = 0.01, Seed){
  set.seed(Seed)
  x0 = runif(1)
  y0 = f(x0)
  
  X = x0
  Y = y0
  
  for (i in 1:max_iter) {
    x1 = runif(1, min=max(c(x0-r,0)), max=min(c(x0+r),1))
    y1 = f(x1)
    
    if(y1 > y0){
      x0 = x1
      y0 = y1
    }
    
    X = c(X, x0)
    Y = c(Y, y0)
  }
  optimizer = min(X[Y>(Y[length(Y)]-epsilon)])
  
  return(list("X"=X, "Y"=Y, "optimizer"=optimizer))
}

### One-parameter Bayesian optimization for f(x); range of x should be (0,1)
## spacing: distance between two initial points
## epsilon: for finding the smallest point that attains max f(x)
## acq_f: acquisition function.  "EI" -- expected improvement

bayesian_opt = function(f, spacing=0.25, xi=0.03, max_iter,Seed, epsilon=0.01, acq_f="EI"){
  x_axis = seq(from = 0, to = 1, by = 0.01)
  
  X<- seq(0,1,spacing)
  Y<-  unlist(lapply(X,f))
  
  Models = NULL
  GPmodel <- c()
  
  if(acq_f == "EI"){
    
    Acq = NULL
    
    for(i in 1:max_iter){
      tryCatch({
        
        set.seed(Seed)
        GPmodel <- GP_fit(X,Y)
        Models = c(Models, list(GPmodel))
        
        #acquisition function: Expected improvement
        min_obj <- function(x){
          GPprediction = predict.GP(GPmodel,x);
          mu <- GPprediction$Y_hat
          sigma <- sqrt(GPprediction$MSE)
          
          #acq_constant: indicate whether the acquisition function is constant, i.e., 0.
          
          acq_constant <<- F
          
          if(sigma==0){
            acq_constant <<- T
            return(0)
          }
          
          f_smaple_opt <- max(Y)
          
          delta <- mu-f_smaple_opt-xi
          delta_plus <- max(delta,0)
          ei <- delta_plus + sigma*dnorm(delta/sigma) - abs(delta)*pnorm(delta/sigma)
          
          #xi <- 0.01
          #Z <- (mu-f_smaple_opt-xi)/sigma
          #ei <- (mu-f_smaple_opt-xi)*pnorm(Z) + sigma*dnorm(Z)
          
          return(-ei)
        }
        
        y_acq = unlist(lapply(x_axis,min_obj))
        Acq = c(Acq, list(y_acq))
        
        x <- optimize(min_obj, c(0, 1))
        x = x$minimum
        #print(x$minimum)
        
        if(acq_constant == T) x=runif(1)
        
        X <- c(X, x)
        Y <- c(Y, f(x))
      }, error = function(e) { print(paste("i =", i, "failed.")) })
    }
  }
  
  temp = floor(1/spacing)
  distance = (X[(temp+1):length(X)] - X[temp:(length(X)-1)])^2
  
  optimizer = min(X[Y>(max(Y)-epsilon)])
  
  if(acq_f == "EI"){
    return(list("GPmodel_evolution" = Models, "optimizer" = optimizer, "acquisition_evolution"=Acq, 
                "evaluated_x"=X, "evaluated_y"=Y, "Distance_evolution"=distance))
  }
}

#####################################################################################
### A toy example function -- one parameter 

#f <- function(x){return(sin(x))}

f <- function(xx){
  x = xx*10
  return(exp(-(x - 2)^2) + exp(-(x - 6)^2/10) + 1/ (x^2 + 1))
}

x_axis = seq(from = 0, to = 1, by = 0.01)
y_truth = f(x_axis)
plot(x_axis,y_truth,type="l")



### New point selected by random search
max_iter <- 20
result_toy_unif = random_search(f, max_iter=max_iter, Seed=2018)
print(result_toy_unif$optimizer)


### New point selected by Expected Improvements
max_iter = 20 
result_toy = bayesian_opt(f, max_iter=max_iter, Seed=10003, acq_f="EI")

for(i in 1:length(result_toy$GPmodel_evolution)){
  
  GPmodel <- result_toy$GPmodel_evolution[[i]]
  
  par(mfrow=c(1,2))
  
  plot.GP(GPmodel, surf_check = TRUE, response = FALSE, shade = TRUE, legends = F, ylim=c(0,3)) 
  lines(x_axis,y_truth,col="green")
  legend("topright",legend=c("Surrogate function", "Uncertainty bounds", "Evaluated points", "Objective"),
         col=c("blue", "red", "black", "green"), lty=(c(1:2,19,1)), cex=0.6, text.width = 0.15)
  
  plot(x_axis,-result_toy$acquisition_evolution[[i]], type = "l", xlab="x",ylab="Utility")
}

par(mfrow=c(1,2),par(mar=c(5.1,4.1,4.1,2.1)))
plot(1:length(result_toy$Distance_evolution),result_toy$Distance_evolution, type = "b", col="blue",
     xlab="Iteration",ylab="Distance")

Y_eval = result_toy$evaluated_y
Y_evolution = rep(0,length(Y_eval)-4)
for (i in 1:length(Y_evolution)) {
  Y_evolution[i] = max(Y_eval[1:(4+i)])
}
plot(1:(length(Y_evolution)),Y_evolution, type = "b", col="red", xlab="Iteration",ylab="max f(x)")

print(result_toy$optimizer)


## Comparison of the two methods for toy example
par(mfrow=(c(1,1)))
plot(1:length(Y_evolution),Y_evolution, type = "b", col="red", xlab="Iteration",ylab="max f(x)", ylim=c(0.9,1.5))
lines(1:length(Y_evolution), result_toy_unif$Y[1:length(Y_evolution)], col="blue", type="b")
legend("topleft",legend=c("Bayesian optimization", "random search"),
       col=c("red", "blue"), lty=(c(1:2,19,1)), cex=0.65, text.width = 6.5)

######################################################################################
### Example on tuning hyperparameters for XGboost -- one hyperparameter
library(xgboost)
library(tidyverse)
library(rBayesianOptimization)

### Data processing

proj<-read.csv('Outbreak_240817.csv')
proj<-proj[sample(1:nrow(proj)), ]
set.seed(100)

## Training labels
Labels <-  proj%>%
  select(humansAffected) %>% # get the column with the # of humans affected
  is.na() %>% # is it NA?
  magrittr::not() 

## Remove information about target variables
proj_nohuman<-proj%>%
  select(-contains("human"))

## extract the least word in each row as species
species_List <- proj$speciesDescription %>%
  str_replace("[[:punct:]]", "") %>% 
  str_extract("[a-z]*$") 
species_List <- tibble(species = species_List)

## Convert to a matrix using 1 hot encoding
options(na.action='na.pass') 
species <- model.matrix(~species-1,species_List)
region <- model.matrix(~country-1,proj)

new_num<-proj_nohuman[,16:20]
new_num$domestic<-str_detect(proj$speciesDescription, "domestic")

## Split traning and validation set
new_proj_matrix <- data.matrix(cbind(new_num, region, species))
n_training <- 11000
my_train<- new_proj_matrix[1:n_training,]
train_label <- Labels[1:n_training]
my_test <- new_proj_matrix[-(1:n_training),]
test_label <- Labels[-(1:n_training)]
train_data <- xgb.DMatrix(data=my_train, label=train_label)
validation_data <- xgb.DMatrix(data = my_test, label= test_label)

##################################################################################

### Tuning hyperparameters-- one hyperparameter: eta

## Objective function
xg_bayes_1<-function(eta){
  mod <- xgboost(data = train_data, # the data           
                 max.depth = 3, # the maximum depth of each decision tree
                 nround = 10, # number of boosting rounds
                 eta = eta,  #step size of each boosting step
                 objective = "binary:logistic") # the objective function 
  
  # generate predictions for validation set
  pred <- predict(mod, validation_data)
  
  # classification error
  rslt <- ifelse(pred > 0.5, 1, 0)
  err <- mean(rslt!= test_label)
  
  return(1-err)
}

## Tuning by random search-- one hyperparameter: eta
max_iter = 10
result_unif = random_search(xg_bayes_1, max_iter = max_iter, Seed = 2018)
print(result_unif$optimizer)

## Tuning by Bayesian optimization-- one hyperparameter: eta
max_iter = 10
x_axis = seq(from = 0, to = 1, by = 0.01)

result = bayesian_opt(xg_bayes_1, max_iter=max_iter, Seed=2018, xi=0.01, acq_f="EI")
for(i in 1:length(result$GPmodel_evolution)){
  
  GPmodel <- result$GPmodel_evolution[[i]]
  
  par(mfrow=c(1,2))
  
  plot.GP(GPmodel, surf_check = TRUE, response = FALSE, shade = TRUE, legends = F, ylim=c(0,3)) 
  legend("bottomright",legend=c("Surrogate function", "Uncertainty bounds", "Evaluated points"),
         col=c("blue", "red", "black"), lty=(c(1:2,19,1)), cex=0.7, text.width = 0.4)
  
  plot(x_axis,-result$acquisition_evolution[[i]], type = "l", xlab="x",ylab="Utility")
}

par(mfrow=c(1,2),par(mar=c(5.1,4.1,4.1,2.1)))
plot(1:length(result$Distance_evolution),result$Distance_evolution, type = "b", col="blue",
     xlab="Iteration",ylab="Distance")

Y_eval = result$evaluated_y
Y_evolution = rep(0,length(Y_eval)-4)
for (i in 1:length(Y_evolution)) {
  Y_evolution[i] = max(Y_eval[1:(4+i)])
}
plot(1:(length(Y_evolution)),Y_evolution, type = "b", col="red", xlab="Iteration", ylab="max f(x)")

print(result$optimizer)

## Comparison of the two methods for XGboost
par(mfrow=(c(1,1)))
plot(1:length(Y_evolution),Y_evolution, type = "b", col="red", xlab="Iteration",ylab="max f(x)", ylim=c(0.98,0.99))
lines(1:length(Y_evolution), result_unif$Y[1:length(Y_evolution)], col="blue", type="b")
legend("bottomright",legend=c("Bayesian optimization", "random search"),
       col=c("red", "blue"), lty=(c(1:2,19,1)), cex=0.65, text.width = 3.5)

##############################################################################
### Example on tuning hyperparameters for XGboost -- multiple hyperparameters

# Adapted from the offcial example 
#(https://cran.r-project.org/web/packages/rBayesianOptimization/rBayesianOptimization.pdf)

library(xgboost)
data(agaricus.train, package = "xgboost")
dtrain <- xgb.DMatrix(agaricus.train$data,
                      label = agaricus.train$label)
cv_folds <- KFold(agaricus.train$label, nfolds = 5,
                  stratified = TRUE, seed = 0)
xgb_cv_bayes <- function(max.depth, min_child_weight, subsample, gamma) {
  cv <- xgb.cv(params = list(booster = "gbtree", eta = 0.01,
                             max_depth = max.depth,
                             min_child_weight = min_child_weight,
                             subsample = subsample, colsample_bytree = 0.3,
                             lambda = 1, alpha = 0, gamma=gamma,
                             objective = "binary:logistic",
                             eval_metric = "auc"),
               data = dtrain, nround = 100,
               folds = cv_folds, prediction = TRUE, showsd = TRUE,
               early_stop_round = 5, maximize = TRUE, verbose = 0)
  list(Score = cv$evaluation_log$test_auc_mean[length(cv$evaluation_log$test_auc_mean)],
       Pred = cv$pred)
}

set.seed(10002)
OPT_Res <- BayesianOptimization(xgb_cv_bayes,bounds = list(max.depth = c(2L, 6L),
                                                           min_child_weight = c(1L, 10L),
                                                           subsample = c(0, 1),
                                                           gamma = c(0.01,1000)),
                                init_grid_dt = NULL, init_points = 4, n_iter = 10,
                                acq = "ei", kappa = 2.576, eps = 0.01,
                                verbose = TRUE)

Y_eval = OPT_Res$Value
Y_evolution = rep(0,length(OPT_Res$History$Value)-4) # number of init points
for (i in 1:length(Y_evolution)) {
  Y_evolution[i] = max(OPT_Res$History$Value[1:(4+i)])
}
plot(1:(length(Y_evolution)),Y_evolution, type = "b", col="red", xlab="Iteration",ylab="max f(x)")
