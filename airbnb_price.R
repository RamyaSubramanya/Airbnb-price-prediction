setwd(dir = "D:\\IIM - K\\SUPERVISED MACHINE LEARNING\\REGRESSION\\REG-PRACTICALS\\airbnb-madrid-ironhack")

#reading the data

train<-read.csv("train.csv")
test<-read.csv("test.csv")

-------------------------------------------------------------------------------------

#combine train and test data for data pre-processing

install.packages("dplyr")
library(dplyr)

data<-bind_rows(train, test)

data.frame(colnames(data))

#feature selection

data<-data[,c(1,12,15,22,23,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,50,51,
              52,53,54,56,57,58,61,62,63,64,65,66,67,69,70,71,72,73,74)]


#excluding host_since, bathrooms(entire column has null values)

data<-data[,-c(2,8)]

---------------------------------------------------------------------------------
  
#handle missing values using median/ mode approach
  
colSums(is.na(data))

data$bedrooms[which(is.na(data$bedrooms))]<-0
data$beds[which(is.na(data$beds))]<-0

data$minimum_minimum_nights[which(is.na(data$minimum_minimum_nights))]<-0
data$maximum_minimum_nights[which(is.na(data$maximum_minimum_nights))]<-0
data$minimum_maximum_nights[which(is.na(data$minimum_maximum_nights))]<-0
data$maximum_maximum_nights[which(is.na(data$maximum_maximum_nights))]<-0

data$minimum_nights_avg_ntm[which(is.na(data$minimum_nights_avg_ntm))]<-0
data$maximum_nights_avg_ntm[which(is.na(data$maximum_nights_avg_ntm))]<-0


data$review_scores_rating[which(is.na(data$review_scores_rating))]<-0
data$review_scores_accuracy[which(is.na(data$review_scores_accuracy))]<-0
data$review_scores_cleanliness[which(is.na(data$review_scores_cleanliness))]<-0
data$review_scores_checkin[which(is.na(data$review_scores_checkin))]<-0
data$review_scores_communication[which(is.na(data$review_scores_communication))]<-0
data$review_scores_location[which(is.na(data$review_scores_location))]<-0
data$review_scores_value[which(is.na(data$review_scores_value))]<-0
data$reviews_per_month[which(is.na(data$reviews_per_month))]<-0

  
--------------------------------------------------------------------------------
#chi-square test to check the relationship between categorical data and target variable
#if p-value<0.05, then we retain the variable i.e., as per chi-square lesser the p-value more significant it is to the model

str(data)

chisq.test(data$host_response_time, data$price, simulate.p.value = TRUE)

chisq.test(data$room_type, data$price, simulate.p.value = TRUE)

chisq.test(data$bathrooms_text, data$price, simulate.p.value = TRUE)

chisq.test(data$has_availability, data$price, simulate.p.value = TRUE)

chisq.test(data$instant_bookable, data$price, simulate.p.value = TRUE)

#as per chi-square test results, has_availability has no significance to the model as p-value>alpha (0.27>0.05)

data<-data[,-c(19)]

--------------------------------------------------------------------------------
#check correlation between continuous variables
  
install.packages("psych")
library(psych)

data.frame(colnames(data))
str(data)

#select continuous data in paris and check correlation

pairs.panels(data[,c(2,3)], scale = TRUE, digits = 2, method = "pearson")
pairs.panels(data[,c(9,10,11,12,13,14,15,16)], scale = TRUE, digits = 2, method = "pearson", cex.cor = 0.5)
pairs.panels(data[,c(17,18,19,20)], scale = TRUE, digits = 2, method = "pearson", hist.col = "green",cex.cor = 0.7)
pairs.panels(data[,c(21,22,23,24,36)], scale = TRUE, digits = 2, method = "pearson", hist.col = "red", cex.cor = 0.6)
pairs.panels(data[,c(25,26,27,28,29,30)], scale = TRUE, digits = 2, method = "pearson", hist.col = "purple", cex.cor = 0.6)
pairs.panels(data[,c(32,33,34,35)], scale = TRUE, digits = 2, method = "pearson", hist.col = "yellow", cex.cor = 0.6)

--------------------------------------------------------------------------------
#create dummy variables for categorical data

install.packages("fastDummies")
library(fastDummies)
  
str(data)

#dropping bathrooms text

data<-data[,-c(7)]


data<-dummy_cols(data, select_columns = "host_response_time", remove_first_dummy = TRUE)
data<-dummy_cols(data, select_columns = "room_type", remove_first_dummy = TRUE)
data<-dummy_cols(data, select_columns = "instant_bookable", remove_first_dummy = TRUE)

data.frame(colnames(data))
data<-data[,-c(2,5,32)]

---------------------------------------------------------------------------------
#split the data into train and test

  
data<-data[,-c(2,3,4,8,9,10,11,13,15,18,19,21,22,25,26,27,28,30,31,33)]

data<-data[,-c(5,6,8,14,15,16,18,19,21)]

data<-subset(data, bedrooms>=1)
data<-subset(data, beds>=1)
data<-subset(data, number_of_reviews>=1)
data<-subset(data, review_scores_rating>=1)
data<-subset(data, review_scores_accuracy>=1)
data<-subset(data, availability_30>=1)

  
Traindata<-data[1:4167,]
Testdata<-data[4168:5556,]
  
-------------------------------------------------------------------------------

#Linear Reg

install.packages("caret",dependencies = TRUE)
library(caret)

install.packages("glmnet",dependencies = TRUE)
library(glmnet)

install.packages("car")
library(car)

library(ggplot2)

set.seed(123)
reg<-lm(price~., data = Traindata)
summary(reg)

set.seed(123)
reg1<-step(reg, direction = "backward", trace = 0)

summary(reg1)
vif(reg1)

prediction<-predict(reg, Traindata, type = "response")
View(prediction)

write.csv(prediction, "P.csv")

result<-data.frame(Actuals = Traindata$price, Predicted = prediction) 

install.packages("Metrics")  
library(Metrics)  

RMSE<-rmse(result$Actuals, result$Predicted)  
RMSE


-------------------------------------------------------------------------------
#cross validation method

?train
custom<-trainControl(method = "repeatedcv", number = 10, repeats = 5)
cv<-train(price~., data = Traindata, method = "lm", trControl = custom)

summary(cv)
plot(varImp(cv, scale = TRUE))

---------------------------------------------------------------------------------

#ridge

set.seed(123)
ridge<-train(price~., data = Traindata, method = "glmnet", trControl = custom,
             tuneGrid = expand.grid(alpha = 0, lambda = seq(0.0001, 1, length = 5)))

ridge$results
ridge$bestTune

summary(ridge)

plot(varImp(ridge, scale = TRUE))

--------------------------------------------------------------------------------
#lasso

set.seed(123)
lasso<-train(price~., data = Traindata, method = "glmnet", trControl = custom,
             tuneGrid = expand.grid(alpha = 1, lambda = seq(0.1, 1, length = 5)))

lasso$results
lasso$bestTune

summary(lasso)

plot(varImp(lasso, scale = TRUE))

---------------------------------------------------------------------------------
#compare the models

modellist<-list(linear = cv, ridge = ridge, lasso = lasso)
compare<-resamples(modellist)
summary(compare)

-----------------------------------------------------------------------------------
  
p<-predict(ridge, Traindata)

result<-data.frame(Actuals = Traindata$price, Predicted = p) 

install.packages("Metrics")  
library(Metrics)  
  
RMSE<-rmse(result$Actuals, result$Predicted)  
RMSE

--------------------------------------------------------------------------------

#decision tree

install.packages("rpart")
library(rpart)

library(rpart.plot)
library(rattle)

?rpart
tree<-rpart(price~.,data = Traindata)
tree
tree$splits  

fancyRpartPlot(tree) 


p1<-predict(tree, Traindata)  

result1<-data.frame(Actuals=Traindata$price,Predicted=p1)

RMSE1<-rmse(result1$Actuals,result1$Predicted)
RMSE1

----------------------------------------------------------------------------------
#RandomForest

install.packages("randomForest")
library(randomForest)

install.packages("janitor")
library(janitor)

Traindata<-clean_names(Traindata)
Testdata<-clean_names(Testdata)


?randomForest
forest<-randomForest(price~.,data = Traindata)
print(forest)
plot(forest)

#observed from plot, ntree: 35

?tuneRF

tuneRF(Traindata[,-c(4)], Traindata[,4], ntreeTry=300, stepFactor=2, improve=0.05,
       trace = FALSE, plot=TRUE, doBest=FALSE)

set.seed(123)
forest1<-randomForest(price~.,data = Traindata, ntree=300, mtry=2)
print(forest1)
summary(forest1)

ForestPredicted<-predict(forest1,Traindata,type = "response")

ForestResult3<-data.frame(Actuals=Traindata$price,Predicted=ForestPredicted)

RMSE2<-rmse(ForestResult3$Actuals, ForestResult3$Predicted)
RMSE2

varImpPlot(forest1,sort = TRUE,n.var = 10, main = "Top 10")
options(scipen = 10)


##predicting the model on test data

finalresult<-predict(forest1,Testdata,type = "response")
prediction_file<-data.frame(Price = finalresult, Id = Testdata$id)

write.csv(prediction_file,"Predicted.csv")
-----------------------------------------------------------------------------------
#boosting
  
install.packages("xgboost")
library(xgboost)
  
train_label<-as.numeric(Traindata[,4])
test_label<-as.numeric(Testdata[,4])
test_label

train_features<-Traindata[,-c(4)]
test_features<-Testdata[,-c(4)]


?as.matrix
train_features<-as.matrix(train_features)
test_features<-as.matrix(test_features)


?xgboost
parameters<-list(eta = 0.3, max_depth = 2, subsample = 1, colsample_bytree = 1, min_child_weight = 1, 
                 gamma = 0, eval_metric = "rmse", objective = "reg:squarederror", booster = "gblinear",
                 scale_pos_weight = 0)


model1 <- xgboost(data = train_features,
                  label = train_label,
                  set.seed(1234),
                  nthread = 6,
                  nround = 200,
                  params = parameters,
                  print_every_n = 50,
                  early_stopping_rounds = 20,
                  verbose = 1)


train_label<-as.matrix(train_label)
pb<-predict(model1,train_label)

view(pb)

write.csv(pb,"predictedboosting.csv")

boostingresult<-data.frame(Actuals=train_label,Predicted=pb)

RMSE3<-rmse(boostingresult$Actuals, boostingresult$Predicted)
RMSE3


--------------------------------------------------------------------------------

#following to be dropped
  

#to be dropped

#host_listings_count 
#host_total_listings_count
#accommodates
#minimum_nights
#maximum_nights
#minimum_minimum_nights
#maximum_minimum_nights
#maximum_maximum_nights
#maximum_nights_avg_ntm
#availability_90
#availability_365
#number_of_reviews_ltm
#number_of_reviews_l30d
#review_scores_cleanliness
#review_scores_checkin
#review_scores_communication
#review_scores_location
#calculated_host_listings_count
#calculated_host_listings_count_entire_homes
#calculated_host_listings_count_shared_rooms

str(data)

data<-data[,-c(2,3,4,8,9,10,11,13,15,18,19,21,22,25,26,27,28,30,31,33)]

