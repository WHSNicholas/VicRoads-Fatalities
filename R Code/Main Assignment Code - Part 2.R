################################################################################
########################## ACTL3142 - Main Assignment ##########################
######################## Fatal Road Crashes in Victoria ########################
###################### Part II - Modelling Fatal Crashes #######################
################################################################################
library(MASS)
library(tidyverse)
library(ggmosaic)
library(caret)
library(pROC)
library(MLmetrics)
library(glmnet)

################################ Reading in Data ###############################
Data = read.csv("VicRoadFatalData.csv")

#Creating Owner State Factor
OWNER_STATE = rep(1, dim(Data)[1])
for (i in 1:length(OWNER_STATE)) {
  if ( Data$OWNER_POSTCODE[i] < 200 ) {
    OWNER_STATE[i] = "Other"
    
  } else if ( Data$OWNER_POSTCODE[i] < 800 | 
              ( Data$OWNER_POSTCODE[i] >= 2600 ) & 
              ( Data$OWNER_POSTCODE[i] <= 2618 ) | 
              ( Data$OWNER_POSTCODE[i] >= 2900 ) & 
              ( Data$OWNER_POSTCODE[i] <= 2920 ) ) {
    OWNER_STATE[i] = "ACT"
    
  } else if ( ( Data$OWNER_POSTCODE[i] >= 1000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 1999 ) | 
              ( Data$OWNER_POSTCODE[i] >= 2000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 2599 ) |
              ( Data$OWNER_POSTCODE[i] >= 2619 ) & 
              ( Data$OWNER_POSTCODE[i] <= 2898 ) |
              ( Data$OWNER_POSTCODE[i] >= 2921 ) & 
              ( Data$OWNER_POSTCODE[i] <= 2999 ) ) {
    OWNER_STATE[i] = "NSW"
    
  } else if ( ( Data$OWNER_POSTCODE[i] >= 3000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 3999 ) | 
              ( Data$OWNER_POSTCODE[i] >= 8000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 8999 ) ) {
    OWNER_STATE[i] = "VIC" 
    
  } else if ( ( Data$OWNER_POSTCODE[i] >= 4000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 4999 ) | 
              ( Data$OWNER_POSTCODE[i] >= 9000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 9999 ) ) {
    OWNER_STATE[i] = "QLD" 
    
  } else if ( ( Data$OWNER_POSTCODE[i] >= 5000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 5999 ) ) {
    OWNER_STATE[i] = "SA" 
    
  } else if ( ( Data$OWNER_POSTCODE[i] >= 6000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 6999 ) ) {
    OWNER_STATE[i] = "WA" 
    
  } else if ( ( Data$OWNER_POSTCODE[i] >= 7000 ) & 
              ( Data$OWNER_POSTCODE[i] <= 7999 ) ) {
    OWNER_STATE[i] = "TAS" 
    
  } else if ( ( Data$OWNER_POSTCODE[i] >= 800 ) & 
              ( Data$OWNER_POSTCODE[i] <= 999 ) ) {
    OWNER_STATE[i] = "VIC" 
    
  } 
}  

Data = Data %>%
  add_column(OWNER_STATE = OWNER_STATE, .after = "OWNER_POSTCODE")

#Creating Time of Day Factor
Data = Data %>%
  add_column(TIME_OF_DAY = cut(strptime(Data$ACCIDENTTIME, 
                                        format = "%H:%M:%S")$hour, 
                               breaks = c(0, 4, 8, 12, 16, 20, 24),
                               labels = c("Dawn", "Early Morning", "Midday", 
                                          "Afternoon", "Evening", "Nighttime"), 
                               include.lowest = TRUE), .after = "DAY_OF_WEEK")  

# Factoring Dates by Month
Data$ACCIDENTDATE = as.Date(Data$ACCIDENTDATE)
Data = Data %>%
  add_column(MONTH = as.integer(factor(format(Data$ACCIDENTDATE, "%B"), 
                                       levels = month.name)), 
             .after = "DAY_OF_WEEK")

#Changing Covariates to Factors
Data$SEX = as_factor(make.names(Data$SEX))
Data$Age.Group = as_factor(make.names(Data$Age.Group))
Data$LICENCE_STATE = as_factor(Data$LICENCE_STATE)
Data$HELMET_BELT_WORN = as_factor(make.names(Data$HELMET_BELT_WORN))
Data$VEHICLE_BODY_STYLE = as_factor(make.names(Data$VEHICLE_BODY_STYLE))
Data$VEHICLE_MAKE = as_factor(make.names(Data$VEHICLE_MAKE))
Data$VEHICLE_TYPE = as_factor(make.names(Data$VEHICLE_TYPE))
Data$FUEL_TYPE = as_factor(Data$FUEL_TYPE)
Data$DAY_OF_WEEK = as_factor(Data$DAY_OF_WEEK)
Data$TIME_OF_DAY = as_factor(make.names(Data$TIME_OF_DAY))
Data$ACCIDENT_TYPE = as_factor(make.names(Data$ACCIDENT_TYPE))
#Data$SPEED_ZONE = as_factor(make.names(Data$SPEED_ZONE))
Data$LIGHT_CONDITION = as_factor(make.names(Data$LIGHT_CONDITION))
Data$ROAD_GEOMETRY = as_factor(make.names(Data$ROAD_GEOMETRY))
Data$SURFACE_COND = as_factor(make.names(Data$SURFACE_COND))
Data$ATMOSPH_COND = as_factor(make.names(Data$ATMOSPH_COND))
Data$ROAD_SURFACE_TYPE = as_factor(make.names(Data$ROAD_SURFACE_TYPE))
Data$OWNER_STATE = as_factor(Data$OWNER_STATE)
Data$fatal = as_factor(make.names(Data$fatal))
#Data$fatal = as.numeric(Data$fatal)

#Deleting Useless Columns
#Data = select(Data, c(2, 4, 6, 8, 10, 11, 15, 20, 22, 23, 24, 25, 26, 27, 29, 30))

#One Hot Encoding
#Data$fatal = as.numeric(Data$fatal)
#dummy = dummyVars(" ~ .", data = Data)
#Data = data.frame(predict(dummy, newdata = Data))


#Splitting Data into Training, Validation and Testing Sets
set.seed(1)
isTrain = createDataPartition(Data$fatal, p = 0.9, list = F)
DataValidate = Data[-isTrain,]
DataTrain = Data[isTrain,]

isValidate = createDataPartition(DataValidate$fatal, p = 0.5, list = F)
DataTest = DataValidate[isValidate, ]
DataValidate = DataValidate[-isValidate, ]





######################## Fitting using KNN #########################


######################## Fitting using Validation Sets #########################
#Building an Initial GLM Model
glmControl = trainControl(method = "none",  classProbs = TRUE, 
                         summaryFunction = defaultSummary)
fitglm = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + VEHICLE_BODY_STYLE +
                 VEHICLE_MAKE + VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + DAY_OF_WEEK + 
                 TIME_OF_DAY + ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + ATMOSPH_COND + 
                 SPEED_ZONE + SURFACE_COND + ROAD_SURFACE_TYPE, 
               data = DataTrain, method = "glm", family = binomial(), 
               metric = "AUC", trControl = glmControl)
summary(fitglm)
fitglm
response = predict(fitglm, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitglm, newdata = DataValidate)
CM = confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
F1 = CM$byClass[7]
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
CM
F1
as.numeric(test_roc$auc)


#Performing Backward Selection
glmControl = trainControl(method = "none",  classProbs = TRUE, 
                          summaryFunction = defaultSummary)
fitback = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + VEHICLE_BODY_STYLE +
                  VEHICLE_MAKE + VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + DAY_OF_WEEK + 
                  TIME_OF_DAY + ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + ATMOSPH_COND + 
                  SPEED_ZONE + SURFACE_COND + ROAD_SURFACE_TYPE, 
               data = DataTrain, method = "glmStepAIC", direction = "backward", 
               family = binomial(), metric = "AUC", trControl = glmControl)
summary(fitback)
fitback
response = predict(fitback, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitback, newdata = DataValidate)
CM = confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
F1 = F1_Score(DataValidate$fatal, prediction)
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
CM
F1
as.numeric(test_roc$auc)
#This was only run for a couple minutes in order to identify which variables to take out


#Cross Validation
cvControl = trainControl(method = "cv", number = 5, classProbs = TRUE, 
                          summaryFunction = prSummary)
fitcv5 = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + 
                 VEHICLE_TYPE + OWNER_STATE + DAY_OF_WEEK + TIME_OF_DAY + 
                 ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE + 
                 SURFACE_COND, 
                data = DataTrain, method = "glm", family = binomial(), 
                metric = "AUC", trControl = cvControl)
summary(fitcv5)
fitcv5
response = predict(fitcv5, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitcv5, newdata = DataValidate)
CM = confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
F1 = CM$byClass[7]
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
CM
F1
as.numeric(test_roc$auc)


#Random Oversampling Methods
overControl = trainControl(method = "boot", number = 5, classProbs = TRUE, 
                         summaryFunction = prSummary, sampling = "up", 
                         savePredictions = "final")
fitover = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + 
                  VEHICLE_TYPE + OWNER_STATE + DAY_OF_WEEK + TIME_OF_DAY + 
                  ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE + 
                  SURFACE_COND, 
               data = DataTrain, method = "glm", family = binomial(), 
               metric = "AUC", trControl = overControl)
summary(fitover)
fitover
response = predict(fitover, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitover, newdata = DataValidate)
CM = confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
F1 = F1_Score(DataValidate$fatal, prediction)
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
CM
F1
as.numeric(test_roc$auc)

#Testing the Model
responseTest = fitover %>%
  predict(newdata = DataTest, type = "prob", list = F)
responseTest = cbind(responseTest, rownames(responseTest))
colnames(responseTest) = c("NonFatal", "Fatal", "ID")
top25 = head(arrange(responseTest, desc(Fatal),), 2500)

table(DataTest[top25$ID, 30])




#Random Oversampling Methods
overControl = trainControl(method = "boot", number = 5, classProbs = TRUE, 
                           summaryFunction = prSummary, sampling = "up", 
                           savePredictions = "final")
fitover = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + 
                  VEHICLE_TYPE + OWNER_STATE + DAY_OF_WEEK + TIME_OF_DAY + 
                  ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE + 
                  SURFACE_COND, 
                data = DataTrain, method = "glm", family = binomial(), 
                metric = "AUC", trControl = overControl)
summary(fitover)
fitover
response = predict(fitover, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitover, newdata = DataValidate)
CM = confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
F1 = F1_Score(DataValidate$fatal, prediction)
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
CM
F1
as.numeric(test_roc$auc)

#Testing the Model
responseTest = fitover %>%
  predict(newdata = DataTest, type = "prob", list = F)
responseTest = cbind(responseTest, rownames(responseTest))
colnames(responseTest) = c("NonFatal", "Fatal", "ID")
top25 = head(arrange(responseTest, desc(Fatal),), 2500)

table(DataTest[top25$ID, 30])






######################## OLD STUFF #########################
#Cross Validation
cvControl = trainControl(method = "cv", number = 5, classProbs = TRUE, 
             summaryFunction = prSummary)
fitcv5 = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + 
                 VEHICLE_MAKE + VEHICLE_TYPE + OWNER_STATE + DAY_OF_WEEK + 
                 TIME_OF_DAY + ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + 
                 SPEED_ZONE + SURFACE_COND + ATMOSPH_COND, 
           data = DataTrain, method = "glm", family = binomial(link = "logit"), 
           metric = "ROC", trControl = cvControl)

summary(fitcv5)
fitcv5
response = predict(fitcv5, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitcv5, newdata = DataValidate)
confusionMatrix(prediction, DataValidate$fatal)
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc)


#Using Upsampling to Combat Class Imbalances
upsControl = trainControl(method = "boot", number = 5, savePredictions = "final",
                          classProbs = TRUE, summaryFunction = prSummary,
                          sampling = "up")
fitups = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + 
                 VEHICLE_MAKE + VEHICLE_TYPE + OWNER_STATE + DAY_OF_WEEK + 
                 TIME_OF_DAY + ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + 
                 SPEED_ZONE + SURFACE_COND + ATMOSPH_COND, 
               data = DataTrain, method = "glm", family = binomial(link = "logit"), 
               metric = "ROC", trControl = upsControl)

summary(fitups)
fitups
response = predict(fitups, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitups, newdata = DataValidate)
confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc)


#Upsampling with Different Variables
upsControl = trainControl(method = "boot", number = 5, savePredictions = "final",
                          classProbs = TRUE, summaryFunction = prSummary,
                          sampling = "up")
fitups = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_MAKE + VEHICLE_BODY_STYLE +
                 VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + DAY_OF_WEEK + TIME_OF_DAY + 
                 ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE + 
                 SURFACE_COND + ATMOSPH_COND + ROAD_SURFACE_TYPE, 
               data = DataTrain, method = "glm", family = binomial(link = "logit"), 
               metric = "ROC", trControl = upsControl)

summary(fitups)
fitups
response = predict(fitups, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitups, newdata = DataValidate)
confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc)

#Downsampling
downControl = trainControl(method = "boot", number = 5, savePredictions = "final",
                          classProbs = TRUE, summaryFunction = prSummary,
                          sampling = "down")
fitdown = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_MAKE + VEHICLE_BODY_STYLE +
                 VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + DAY_OF_WEEK + TIME_OF_DAY + 
                 ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE + 
                 SURFACE_COND + ATMOSPH_COND + ROAD_SURFACE_TYPE, 
               data = DataTrain, method = "glm", family = binomial(link = "logit"), 
               metric = "ROC", trControl = upsControl)

summary(fitdown)
fitdown
response = predict(fitdown, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitdown, newdata = DataValidate)
confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc)


#Treebagging
Control = trainControl(method = "cv", number = 5)
fittree = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_MAKE + VEHICLE_BODY_STYLE +
                  VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + DAY_OF_WEEK + TIME_OF_DAY + 
                  ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE + 
                  SURFACE_COND + ATMOSPH_COND + ROAD_SURFACE_TYPE, 
                data = DataTrain, method = "treebag", trControl = Control)

summary(fitdown)
fitdown
response = predict(fitdown, newdata = DataValidate, type = "prob", list = F)
prediction = predict(fitdown, newdata = DataValidate)
confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc)


#Changing Thresholds
upsControl = trainControl(method = "boot", number = 5, savePredictions = "final",
                          classProbs = TRUE, summaryFunction = prSummary,
                          sampling = "up")
fitups = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_MAKE + VEHICLE_BODY_STYLE +
                 VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + DAY_OF_WEEK + TIME_OF_DAY + 
                 ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE + 
                 SURFACE_COND + ATMOSPH_COND + ROAD_SURFACE_TYPE, 
               data = DataTrain, method = "glm", family = binomial(link = "logit"), 
               metric = "ROC", trControl = upsControl)
summary(fitups)
fitups
response = predict(fitups, newdata = DataValidate, type = "prob", list = F)
altResp = response$FALSE.
altResp = factor(ifelse(altResp >= 0.02, "FALSE.", "TRUE."))

prediction = predict(fitups, newdata = DataValidate)
CM = confusionMatrix(prediction, DataValidate$fatal, positive = "TRUE.")
test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
CM
as.numeric(test_roc$auc)


alpha = rep(1, 50)
Accuracy = rep(0, 50)
F1 = rep(0, 50)
Sensitivity = rep(0, 50)
Specificity = rep(0, 50)

Threshold = cbind.data.frame(alpha, AUC, Accuracy, F1, Sensitivity, Specificity)

for (i in 1:50) {
  response = predict(fitups, newdata = DataValidate, type = "prob", list = F)
  altResp = response$FALSE.
  altResp = factor(ifelse(altResp >= i / 100, "FALSE.", "TRUE."))
  
  prediction = predict(fitups, newdata = DataValidate)
  CM = confusionMatrix(altResp, DataValidate$fatal, positive = "TRUE.")
  test_roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
  as.numeric(test_roc$auc)
  
  Threshold$alpha[i] = i / 100
  Threshold$Accuracy[i] = as.numeric(CM$overall[1])
  Threshold$F1[i] = as.numeric(CM$byClass[7])
  Threshold$Sensitivity[i] = as.numeric(CM$byClass[1])
  Threshold$Specificity[i] = as.numeric(CM$byClass[2])
}



#Test

upsControl = trainControl(method = "boot", number = 5, savePredictions = "final",
                          classProbs = TRUE, summaryFunction = prSummary,
                          sampling = "up")
fitups = train(fatal ~ SEX + Age.Group + HELMET_BELT_WORN + VEHICLE_MAKE + VEHICLE_BODY_STYLE +
                 VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + DAY_OF_WEEK + TIME_OF_DAY + 
                 ACCIDENT_TYPE + LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE + 
                 SURFACE_COND + ATMOSPH_COND + ROAD_SURFACE_TYPE, 
               data = DataTrain, method = "glm", family = binomial(link = "logit"), 
               metric = "ROC", trControl = upsControl)

fitups
response = predict(fitups, newdata = DataValidate, type = "prob", list = F)
altResp = response$TRUE.
altResp = factor(ifelse(altResp >= 0.2, "TRUE.", "FALSE."))

prediction = predict(fitups, newdata = DataValidate)
CM = confusionMatrix(altResp, DataValidate$fatal, positive = "TRUE.")
CM
roc = roc(DataValidate$fatal ~ response$TRUE., plot = TRUE, print.auc = TRUE)
as.numeric(roc$auc)






################################################################################
########################## ACTL3142 - Main Assignment ##########################
######################## Fatal Road Crashes in Victoria ########################
###################### Part II - Modelling Fatal Crashes #######################
################################################################################

