################################################################################
########################## ACTL3142 - Main Assignment ##########################
######################## Fatal Road Crashes in Victoria ########################
################## Part III - Predictive Modelling of Drivers ################## 
################################################################################
library(MASS)
library(tidyverse)
library(caret)
library(pROC)
library(MLmetrics)
library(glmnet)


############################# Data without Accident ############################
Data = read.csv("VicRoadFatalData.csv")
DriversEval = read.csv("Drivers_Eval.csv")
SelectedDrivers = read.csv("selected_drivers.csv")

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

#Changing Covariates to Factors
Data$SEX = as_factor(make.names(Data$SEX))
Data$Age.Group = as_factor(make.names(Data$Age.Group))
Data$LICENCE_STATE = as_factor(Data$LICENCE_STATE)
Data$HELMET_BELT_WORN = as_factor(make.names(Data$HELMET_BELT_WORN))
Data$VEHICLE_BODY_STYLE = as_factor(make.names(Data$VEHICLE_BODY_STYLE))
Data$VEHICLE_MAKE = as_factor(make.names(Data$VEHICLE_MAKE))
Data$VEHICLE_TYPE = as_factor(make.names(Data$VEHICLE_TYPE))
Data$FUEL_TYPE = as_factor(Data$FUEL_TYPE)
Data$OWNER_STATE = as_factor(Data$OWNER_STATE)
Data$fatal = as_factor(make.names(Data$fatal))
#Data$fatal = as.numeric(Data$fatal)

#Deleting Unwanted Columns
Data = select(Data, c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28))

#Creating Dummy Variables
dummy = dummyVars( ~ SEX + Age.Group + LICENCE_STATE + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + VEHICLE_BODY_STYLE + VEHICLE_MAKE +
                     VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + fatal, data = Data, fullRank = T)
FactorMat = as.matrix(data.frame(predict(dummy, newdata = Data)))
Factor = data.frame(predict(dummy, newdata = Data))
Factor$fatal.TRUE. = as.factor(Factor$fatal.TRUE.)

Factor$SEX.M = as.factor(make.names(Factor$SEX.M))
Factor$SEX.U = as.factor(make.names(Factor$SEX.U))
Factor$Age.Group.X30.39 = as.factor(make.names(Factor$Age.Group.X30.39))
Factor$Age.Group.X18.21 = as.factor(make.names(Factor$Age.Group.X18.21))
Factor$Age.Group.X40.49 = as.factor(make.names(Factor$Age.Group.X40.49))
Factor$Age.Group.X70. = as.factor(make.names(Factor$Age.Group.X70.))
Factor$Age.Group.X50.59 = as.factor(make.names(Factor$Age.Group.X50.59))
Factor$Age.Group.X26.29 = as.factor(make.names(Factor$Age.Group.X26.29))
Factor$Age.Group.X22.25 = as.factor(make.names(Factor$Age.Group.X22.25))
Factor$Age.Group.X64.69 = as.factor(make.names(Factor$Age.Group.X64.69))
Factor$Age.Group.X16.17 = as.factor(make.names(Factor$Age.Group.X16.17))
Factor$HELMET_BELT_WORN.Other = as.factor(make.names(Factor$HELMET_BELT_WORN.Other))
Factor$HELMET_BELT_WORN.Seatbelt.not.worn = as.factor(make.names(Factor$HELMET_BELT_WORN.Seatbelt.not.worn))
Factor$VEHICLE_TYPE.Utility = as.factor(make.names(Factor$VEHICLE_TYPE.Utility))
Factor$VEHICLE_TYPE.Panel.Van = as.factor(make.names(Factor$VEHICLE_TYPE.Panel.Van))
Factor$VEHICLE_TYPE.Station.Wagon = as.factor(make.names(Factor$VEHICLE_TYPE.Station.Wagon))
Factor$VEHICLE_TYPE.Heavy.Vehicle..Rigid....4.5.Tonnes = as.factor(make.names(Factor$VEHICLE_TYPE.Heavy.Vehicle..Rigid....4.5.Tonnes))
Factor$VEHICLE_TYPE.Other = as.factor(make.names(Factor$VEHICLE_TYPE.Other))
Factor$VEHICLE_TYPE.Taxi = as.factor(make.names(Factor$VEHICLE_TYPE.Taxi))
Factor$VEHICLE_TYPE.Prime.Mover...Single.Trailer = as.factor(make.names(Factor$VEHICLE_TYPE.Prime.Mover...Single.Trailer))
Factor$OWNER_STATE.SA = as.factor(make.names(Factor$OWNER_STATE.SA))
Factor$OWNER_STATE.NSW = as.factor(make.names(Factor$OWNER_STATE.NSW))
Factor$OWNER_STATE.Other = as.factor(make.names(Factor$OWNER_STATE.Other))
Factor$OWNER_STATE.QLD = as.factor(make.names(Factor$OWNER_STATE.QLD))
Factor$OWNER_STATE.ACT = as.factor(make.names(Factor$OWNER_STATE.ACT))
Factor$OWNER_STATE.WA = as.factor(make.names(Factor$OWNER_STATE.WA))
Factor$OWNER_STATE.TAS = as.factor(make.names(Factor$OWNER_STATE.TAS))
Factor$LICENCE_STATE.Other = as.factor(make.names(Factor$LICENCE_STATE.Other))
Factor$VEHICLE_BODY_STYLE.DC.UTE = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.DC.UTE))
Factor$VEHICLE_BODY_STYLE.SEDAN = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.SEDAN))
Factor$VEHICLE_BODY_STYLE.VAN = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.VAN))
Factor$VEHICLE_BODY_STYLE.S.WAG = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.S.WAG))
Factor$VEHICLE_BODY_STYLE.WAGON = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.WAGON))
Factor$VEHICLE_BODY_STYLE.Other = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.Other))
Factor$VEHICLE_BODY_STYLE.UTIL = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.UTIL))
Factor$VEHICLE_BODY_STYLE.TRAY = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.TRAY))
Factor$VEHICLE_BODY_STYLE.P.MVR = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.P.MVR))
Factor$VEHICLE_BODY_STYLE.SED = as.factor(make.names(Factor$VEHICLE_BODY_STYLE.SED))
Factor$VEHICLE_MAKE.TOYOTA = as.factor(make.names(Factor$VEHICLE_MAKE.TOYOTA))
Factor$VEHICLE_MAKE.SUBARU = as.factor(make.names(Factor$VEHICLE_MAKE.SUBARU))
Factor$VEHICLE_MAKE.FORD = as.factor(make.names(Factor$VEHICLE_MAKE.FORD))
Factor$VEHICLE_MAKE.HOLDEN = as.factor(make.names(Factor$VEHICLE_MAKE.HOLDEN))
Factor$VEHICLE_MAKE.MITSUB = as.factor(make.names(Factor$VEHICLE_MAKE.MITSUB))
Factor$VEHICLE_MAKE.MERC.B = as.factor(make.names(Factor$VEHICLE_MAKE.MERC.B))
Factor$VEHICLE_MAKE.NISSAN = as.factor(make.names(Factor$VEHICLE_MAKE.NISSAN))
Factor$VEHICLE_MAKE.HYNDAI = as.factor(make.names(Factor$VEHICLE_MAKE.HYNDAI))
Factor$VEHICLE_MAKE.MAZDA = as.factor(make.names(Factor$VEHICLE_MAKE.MAZDA))
Factor$VEHICLE_MAKE.KENWTH = as.factor(make.names(Factor$VEHICLE_MAKE.KENWTH))
Factor$VEHICLE_MAKE.HONDA = as.factor(make.names(Factor$VEHICLE_MAKE.HONDA))
Factor$VEHICLE_MAKE.VOLKS = as.factor(make.names(Factor$VEHICLE_MAKE.VOLKS))
Factor$FUEL_TYPE.Diesel = as.factor(make.names(Factor$FUEL_TYPE.Diesel))
Factor$FUEL_TYPE.Gas = as.factor(make.names(Factor$FUEL_TYPE.Gas))
Factor$FUEL_TYPE.Multi = as.factor(make.names(Factor$FUEL_TYPE.Multi))
Factor$FUEL_TYPE.Petrol = as.factor(make.names(Factor$FUEL_TYPE.Petrol))
Factor$fatal.TRUE. = as.factor(make.names(Factor$fatal.TRUE.))


#Splitting Data into Training, Validation and Testing Sets
#set.seed(1)
isTrain = createDataPartition(Data$fatal, p = 0.9, list = F)
DataValidate = Data[-isTrain, ]
DataTrain = Data[isTrain, ]
FactorValidate = Factor[-isTrain, ]
FactorTrain = Factor[isTrain, ]
FactorMatValidate = FactorMat[-isTrain, ]
FactorMatTrain = FactorMat[isTrain, ]

isValidate = createDataPartition(DataValidate$fatal, p = 0.5, list = F)
DataTest = DataValidate[isValidate, ]
DataValidate = DataValidate[-isValidate, ]
FactorTest = FactorValidate[isValidate, ]
FactorValidate = FactorValidate[-isValidate, ]
FactorMatTest = FactorMatValidate[isValidate, ]
FactorMatValidate = FactorMatValidate[-isValidate, ]

#Upsampling the Data
DataTrainUp = downSample(x = DataTrain[, -ncol(DataTrain)], y = DataTrain[, ncol(DataTrain)])
dummy = dummyVars( ~ SEX + Age.Group + LICENCE_STATE + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + VEHICLE_BODY_STYLE + VEHICLE_MAKE +
                     VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE + Class, data = DataTrainUp, fullRank = T)
FactorMatTrainUp = as.matrix(data.frame(predict(dummy, newdata = DataTrainUp)))
Factor$fatal.TRUE. = as.factor(Factor$fatal.TRUE.)


############################ Classification Problem ############################ 
#Best Subset Selection
#glmControl = trainControl(method = "none",  classProbs = TRUE, 
#                          summaryFunction = prSummary)
#fitback = train(fatal.TRUE. ~ . -VEHICLE_YEAR_MANUF, 
#                data = FactorTrain, method = "glmStepAIC", direction = "backward", 
#                family = binomial(), metric = "AUC", trControl = glmControl)


#Taking away
#VEHICLE_BODY_STYLE.WAGONX1 
#VEHICLE_BODY_STYLE.SEDANX1 
#VEHICLE_BODY_STYLE.VANX1 
#VEHICLE_MAKE.HYNDAIX1
#Age.Group.X64.69X1 
#VEHICLE_MAKE.MERC.BX1  
#VEHICLE_MAKE.MAZDAX1 
#VEHICLE_MAKE.MITSUBX1
#VEHICLE_BODY_STYLE.SEDX1
#VEHICLE_BODY_STYLE.DC.UTEX1 
#OWNER_STATE.WAX1
#VEHICLE_BODY_STYLE.TRAYX1
#VEHICLE_MAKE.VOLKSX1 
#Age.Group.X18.21X1
#Age.Group.X50.59X1  
#VEHICLE_TYPE.Panel.VanX1  
#FUEL_TYPE.DieselX1
#VEHICLE_MAKE.HONDAX1
#Age.Group.X26.29X1 
#LICENCE_STATE.OtherX1 
#SEX.UX1  
#OWNER_STATE.OtherX1 
#OWNER_STATE.ACTX1 
#VEHICLE_MAKE.SUBARUX1
#FUEL_TYPE.GasX1
#FUEL_TYPE.MultiX1 



#Logistic Ridge Regression
fitlasso = cv.glmnet(FactorMatTrain[, c(-20, -17, -18, -33, -10, -31, -34, -30, -25, -16, -54, -23, -37, -4, -7, -39, -47, -36, -8, -12, -2, -51, -53, -27, -48, -45, -56)], 
                     FactorMatTrain[,c(56)], alpha = 1, type.measure = "deviance", family = "binomial", standardise = T)
plot(fitlasso)

#Diagnostics
summary(fitlasso)
fitlasso
probabilitiesLasso = fitlasso %>% 
  predict(newx = FactorMatValidate[, c(-20, -17, -18, -33, -10, -31, -34, -30, -25, -16, -54, -23, -37, -4, -7, -39, -47, -36, -8, -12, -2, -51, -53, -27, -48, -45, -56)], 
          s = fitlasso$lambda.min, type = "response")
predictionLasso = rep("FALSE.", dim(FactorValidate)[1])

for (i in 1:9999) {
  if (probabilitiesLasso[i] > 0.05) {
    predictionLasso[i] = "TRUE."
  }
}

table(predictionLasso, FactorMatValidate[, c(56)])
round(coef(fitlasso, s = fitlasso$lambda.min),4)
fitlasso$lambda.min

#Make prediction on Validation data
probabilitiesLasso = fitlasso %>% 
  predict(newx = FactorMatValidate[, c(-20, -17, -18, -33, -10, -31, -34, -30, -25, -16, -54, -23, -37, -4, -7, -39, -47, -36, -8, -12, -2, -51, -53, -27, -48, -45, -56)], 
          s = fitlasso$lambda.min, type = "response")

probabilitiesLasso = as.data.frame(probabilitiesLasso)
colnames(probabilitiesLasso) = c("Fatal")
top25 = head(arrange(probabilitiesLasso, desc(Fatal),), 2500)
table(FactorValidate[rownames(top25), 56])

#Make prediction on Test data
probabilitiesLasso = fitlasso %>% 
  predict(newx = FactorMatTest[, c(-20, -17, -18, -33, -10, -31, -34, -30, -25, -16, -54, -23, -37, -4, -7, -39, -47, -36, -8, -12, -2, -51, -53, -27, -48, -45, -56)], 
          s = fitlasso$lambda.min, type = "response")

probabilitiesLasso = as.data.frame(probabilitiesLasso)
colnames(probabilitiesLasso) = c("Fatal")
top25 = head(arrange(probabilitiesLasso, desc(Fatal),), 2500)
table(FactorTest[rownames(top25), 56])



#Logistic Ridge Regression with Upsampling
fitLassoOver = cv.glmnet(FactorMatTrainUp[, c(-20, -17, -18, -33, -10, -31, -34, -30, -25, -16, -54, -23, -37, -4, -7, -39, -47, -36, -8, -12, -2, -51, -53, -27, -48, -45, -56)], 
                     FactorMatTrainUp[,c(56)], alpha = 1, type.measure = "deviance", family = "binomial", standardise = T)
plot(fitLassoOver)





# Make prediction on Validation data
probabilitiesLassoOver = fitLassoOver %>% 
  predict(newx = FactorMatValidate[, c(-20, -17, -18, -33, -10, -31, -34, -30, -25, -16, -54, -23, -37, -4, -7, -39, -47, -36, -8, -12, -2, -51, -53, -27, -48, -45, -56)], 
          s = fitLassoOver$lambda.min, type = "response")

probabilitiesLassoOver = as.data.frame(probabilitiesLassoOver)
colnames(probabilitiesLassoOver) = c("Fatal")
top25 = head(arrange(probabilitiesLassoOver, desc(Fatal),), 2500)
table(FactorValidate[rownames(top25), 56])

#Make prediction on Test data
probabilitiesLassoOver = fitLassoOver %>% 
  predict(newx = FactorMatValidate[, c(-20, -17, -18, -33, -10, -31, -34, -30, -25, -16, -54, -23, -37, -4, -7, -39, -47, -36, -8, -12, -2, -51, -53, -27, -48, -45, -56)], 
          s = fitLassoOver$lambda.min, type = "response")

probabilitiesLassoOver = as.data.frame(probabilitiesLassoOver)
colnames(probabilitiesLassoOver) = c("Fatal")
top25 = head(arrange(probabilitiesLassoOver, desc(Fatal),), 2500)
table(FactorValidate[rownames(top25), 56])



############################ DriversEval Prediction ############################ 
#Creating Owner State Factor
OWNER_STATE = rep(1, dim(DriversEval)[1])
for (i in 1:length(OWNER_STATE)) {
  if ( DriversEval$OWNER_POSTCODE[i] < 200 ) {
    OWNER_STATE[i] = "Other"
    
  } else if ( DriversEval$OWNER_POSTCODE[i] < 800 | 
              ( DriversEval$OWNER_POSTCODE[i] >= 2600 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 2618 ) | 
              ( DriversEval$OWNER_POSTCODE[i] >= 2900 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 2920 ) ) {
    OWNER_STATE[i] = "ACT"
    
  } else if ( ( DriversEval$OWNER_POSTCODE[i] >= 1000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 1999 ) | 
              ( DriversEval$OWNER_POSTCODE[i] >= 2000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 2599 ) |
              ( DriversEval$OWNER_POSTCODE[i] >= 2619 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 2898 ) |
              ( DriversEval$OWNER_POSTCODE[i] >= 2921 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 2999 ) ) {
    OWNER_STATE[i] = "NSW"
    
  } else if ( ( DriversEval$OWNER_POSTCODE[i] >= 3000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 3999 ) | 
              ( DriversEval$OWNER_POSTCODE[i] >= 8000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 8999 ) ) {
    OWNER_STATE[i] = "VIC" 
    
  } else if ( ( DriversEval$OWNER_POSTCODE[i] >= 4000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 4999 ) | 
              ( DriversEval$OWNER_POSTCODE[i] >= 9000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 9999 ) ) {
    OWNER_STATE[i] = "QLD" 
    
  } else if ( ( DriversEval$OWNER_POSTCODE[i] >= 5000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 5999 ) ) {
    OWNER_STATE[i] = "SA" 
    
  } else if ( ( DriversEval$OWNER_POSTCODE[i] >= 6000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 6999 ) ) {
    OWNER_STATE[i] = "WA" 
    
  } else if ( ( DriversEval$OWNER_POSTCODE[i] >= 7000 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 7999 ) ) {
    OWNER_STATE[i] = "TAS" 
    
  } else if ( ( DriversEval$OWNER_POSTCODE[i] >= 800 ) & 
              ( DriversEval$OWNER_POSTCODE[i] <= 999 ) ) {
    OWNER_STATE[i] = "VIC" 
    
  } 
}  

DriversEval = DriversEval %>%
  add_column(OWNER_STATE = OWNER_STATE, .after = "OWNER_POSTCODE")

#Changing Covariates to Factors
DriversEval$SEX = as_factor(make.names(DriversEval$SEX))
DriversEval$Age.Group = as_factor(make.names(DriversEval$Age.Group))
DriversEval$LICENCE_STATE = as_factor(DriversEval$LICENCE_STATE)
DriversEval$HELMET_BELT_WORN = as_factor(make.names(DriversEval$HELMET_BELT_WORN))
DriversEval$VEHICLE_BODY_STYLE = as_factor(make.names(DriversEval$VEHICLE_BODY_STYLE))
DriversEval$VEHICLE_MAKE = as_factor(make.names(DriversEval$VEHICLE_MAKE))
DriversEval$VEHICLE_TYPE = as_factor(make.names(DriversEval$VEHICLE_TYPE))
DriversEval$FUEL_TYPE = as_factor(DriversEval$FUEL_TYPE)
DriversEval$OWNER_STATE = as_factor(DriversEval$OWNER_STATE)

#Deleting Unwanted Columns
DriversEval = select(DriversEval, c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

#Creating Dummy Variables
dummy = dummyVars( ~ SEX + Age.Group + LICENCE_STATE + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + VEHICLE_BODY_STYLE + VEHICLE_MAKE +
                     VEHICLE_TYPE + FUEL_TYPE + OWNER_STATE, data = DriversEval, fullRank = T)
FactorDrEval = as.matrix(data.frame(predict(dummy, newdata = DriversEval)))


#Make prediction on Test data
probabilitiesLasso = fitlasso %>% 
  predict(newx = FactorDrEval[, c(-20, -17, -18, -33, -10, -31, -34, -30, -25, -16, -54, -23, -37, -4, -7, -39, -47, -36, -8, -12, -2, -51, -53, -27, -48, -45, -56)], 
          s = fitlasso$lambda.min, type = "response")

probabilitiesLasso = as.data.frame(probabilitiesLasso)
colnames(probabilitiesLasso) = c("Fatal")
top25 = head(arrange(probabilitiesLasso, desc(Fatal),), 2500)

Selected = DriversEval[rownames(top25),]
table(Selected$SEX) 
hist(Selected$AGE)
table(Selected$Age.Group)
table(Selected$HELMET_BELT_WORN)
table(DriversEval$HELMET_BELT_WORN)
table(Selected$VEHICLE_BODY_STYLE)
table(DriversEval$VEHICLE_BODY_STYLE)
table(Selected$VEHICLE_MAKE)
table(DriversEval$VEHICLE_MAKE)
table(Selected$VEHICLE_TYPE)
table(DriversEval$VEHICLE_TYPE)

SelectedID = as.data.frame(Selected$DRIVER_ID)
colnames(SelectedID) = "DRIVER_ID"
write_csv(SelectedID, "selectedDr.csv")



################################################################################
########################## ACTL3142 - Main Assignment ##########################
######################## Fatal Road Crashes in Victoria ########################
################## Part III - Predictive Modelling of Drivers ################## 
################################################################################
