#########################################################
## data Mining I HW3
## Subhashchandra Babu Madineni   UBIT = 50373860
## Created on 7 th october
## Edited: 
#########################################################
rm(list = ls())
#install.packages("ISLR")
#install.packages("glmnet")
#install.packages("tidyverse")
library("ISLR")
library("tidyverse")
library(glmnet)
#install.packages('leaps')
library(leaps)
#setwd('C:/University At Buffalo Fall 2020 Classes/EAS 506-CDA 541Stastical Data Mining/R for STA/homework_3')

#install.packages('Metrics')
library(Metrics)

#########################################################
# Loading th dataset
#########################################################
train_data = read.csv("C:/University At Buffalo Fall 2020 Classes/EAS 506-CDA 541Stastical Data Mining/R for STA/homework_3/ticdata2000.txt",sep ="\t", header = FALSE)
X_test_data  = read.csv("C:/University At Buffalo Fall 2020 Classes/EAS 506-CDA 541Stastical Data Mining/R for STA/homework_3/ticeval2000.txt",sep ="\t", header = FALSE)
Y_test_data  =read.csv("C:/University At Buffalo Fall 2020 Classes/EAS 506-CDA 541Stastical Data Mining/R for STA/homework_3/tictgts2000.txt",sep ="\t", header = FALSE)

dim(train_data)
dim(X_test_data)
dim(Y_test_data)


X=data.matrix(train_data[,-86])
Y=data.matrix(train_data[,86])
X_test = data.matrix(X_test_data)
Y_test = data.matrix(Y_test_data)

#########################################################
# Applying OLS model
#########################################################


ols = lm(formula =train_data$V86 ~.,data= train_data)
summary(ols)

pred_train = predict(ols,train_data[,-86])
pred_test = predict(ols,X_test_data)

train_mse = mean((pred_train - train_data$V86)^2)     #train_MSE = 0.0521
test_mse = mean((pred_test - Y_test_data$V1)^2)           #test_MSE  = 0.05398

#########################################################
#  Applying The Forward  Subset Selection Method
#########################################################
regfit.fwd <- regsubsets(train_data$V86~., data = train_data, nvmax = 86, method = "forward")

#########################################################
# 1)A Analysing the summaries for Forward Selection
#########################################################
summary(regfit.fwd)

which((summary(regfit.fwd))$cp == min((summary(regfit.fwd))$cp))

(summary(regfit.fwd))$outmat[23,]

test_matrix = model.matrix(Y_test_data$V1~.,data=X_test_data)
coef=coef(regfit.fwd ,id=23)
pred=test_matrix [,names(coef)]%*% coef
error_fwd= sum(( Y_test_data$V1-pred)^2)/length(pred)

reg.summary_fwd = summary(regfit.fwd)
par(mfrow = c(1,1))
plot(regfit.fwd,scale="bic")

#########################################################
# 1)B Applying The Backward  Subset Selection Method
#########################################################

regfit.bwd <- regsubsets(train_data$V86~., data = train_data, nvmax = 86, method = "backward")

#########################################################
# 1)A Analysing the summaries for Backward Selection
#########################################################
summary(regfit.bwd)
which((summary(regfit.bwd))$cp == min((summary(regfit.bwd))$cp))        #29 variables model is the best model

(summary(regfit.bwd))$outmat[29,]

test_matrix = model.matrix(Y_test_data$V1~.,data=X_test_data)
coef=coef(regfit.bwd ,id=29)
pred=test_matrix [,names(coef)]%*% coef
error= sum(( Y_test_data$V1-pred)^2)/length(pred)
summary(pred)
error

reg.summary_fwd = summary(regfit.fwd)
par(mfrow = c(1,1))
plot(regfit.bwd,scale="bic")

#########################################################
#  Fitting a Ridge REGRESSION to The dataset
#########################################################

set.seed(125)
ridge.fit = glmnet(X,Y,type.measure = "mse", alpha = 1,family = "gaussian")
ridge.predicted <- as.matrix( predict(ridge.fit, s=ridge.fit$lambda.min, newx=X_test))
par(mfrow = c(1,1))
plot(ridge.fit)
mean((Y_test - ridge.predicted))                                          
mse(Y_test,ridge.predicted)                                          


#########################################################
#  Fitting a Lasso REGRESSION to The dataset
#########################################################

set.seed(12345)
lasso.fit = glmnet(X,Y,type.measure = "mse", alpha = 0,family = "gaussian")

lasso.predicted <- as.matrix( predict(lasso.fit, s=lasso.fit$lambda.min, newx=X_test))
par(mfrow = c(1,1))
plot(lasso.fit)
mean((Y_test - lasso.predicted))                                          
mse(Y_test,lasso.predicted)              
