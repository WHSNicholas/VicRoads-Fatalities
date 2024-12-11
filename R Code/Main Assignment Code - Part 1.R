################################################################################
########################## ACTL3142 - Main Assignment ##########################
######################## Fatal Road Crashes in Victoria ########################
###################### Part I - Exploratory Data Analysis ######################
################################################################################
library(MASS)
library(tidyverse)
library(ggmosaic)
library(caret)


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
Data$SEX = as_factor(Data$SEX)
Data$Age.Group = as_factor(Data$Age.Group)
Data$Age.Group = ordered(Data$Age.Group, 
                         levels = c("16-17", "18-21", "22-25", "26-29", "30-39", 
                                    "40-49", "50-59", "60-64", "64-69", "70+"))
Data$LICENCE_STATE = as_factor(Data$LICENCE_STATE)
Data$HELMET_BELT_WORN = as_factor(Data$HELMET_BELT_WORN)
Data$VEHICLE_BODY_STYLE = as_factor(Data$VEHICLE_BODY_STYLE)
Data$VEHICLE_MAKE = as_factor(Data$VEHICLE_MAKE)
Data$VEHICLE_TYPE = as_factor(Data$VEHICLE_TYPE)
Data$FUEL_TYPE = as_factor(Data$FUEL_TYPE)
Data$DAY_OF_WEEK = as_factor(Data$DAY_OF_WEEK)
Data$DAY_OF_WEEK = ordered(Data$DAY_OF_WEEK, 
                           levels = c("Monday", "Tuesday","Wednesday", 
                                      "Thursday", "Friday", "Saturday", 
                                      "Sunday"))
Data$ACCIDENT_TYPE = as_factor(Data$ACCIDENT_TYPE)
Data$LIGHT_CONDITION = as_factor(Data$LIGHT_CONDITION)
Data$ROAD_GEOMETRY = as_factor(Data$ROAD_GEOMETRY)
Data$SURFACE_COND = as_factor(Data$SURFACE_COND)
Data$ATMOSPH_COND = as_factor(Data$ATMOSPH_COND)
Data$ROAD_SURFACE_TYPE = as_factor(Data$ROAD_SURFACE_TYPE)
Data$OWNER_STATE = as_factor(Data$OWNER_STATE)
summary(Data)

DataFatal = subset(Data, fatal == TRUE)


############################# Driver Data Analysis #############################
#Age Distribution
Data %>%
  ggplot(aes(x = fatal, y = AGE, colour = fatal)) + 
    geom_boxplot(notch = TRUE) + 
    labs(x = "Fatality of Crash", y = "Age", colour = "Fatal",
         title = "Age Distribution of Fatal and Non-Fatal Crashes") +
    scale_y_continuous(breaks = seq(20, 90, 10)) +
    scale_color_discrete(labels = c("NonFatal", "Fatal")) +
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.line = element_line(linewidth = 1.2, colour = "grey80"),
          text = element_text(family = "Calibri"))
#ggsave("Age Distribution of Fatal and Non-Fatal Crashes.jpg")

#Fatality by Sex
SexTable = table(Data$SEX, Data$fatal)
SexTable = cbind(SexTable, SexTable[,2]/(SexTable[,1] + SexTable[,2]))
colnames(SexTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
SexTable

#Fatality by Age Group
AgeGroupTable = table(Data$Age.Group, Data$fatal)
AgeGroupTable = cbind(AgeGroupTable,
                      AgeGroupTable[,2] / (AgeGroupTable[,1] + AgeGroupTable[,2]))
colnames(AgeGroupTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
round(AgeGroupTable, 6)

#Fatality by Seatbelt/Helmet
SeatbeltTable = table(Data$HELMET_BELT_WORN, Data$fatal)
SeatbeltTable = cbind(SeatbeltTable, 
                      SeatbeltTable[,2] / (SeatbeltTable[,1] + SeatbeltTable[,2]))
colnames(SeatbeltTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
SeatbeltTable


################################# Vehicle Data ################################# 
#Vehicle Manufacturing Year
Data %>%
  ggplot(aes(x = VEHICLE_YEAR_MANUF)) + 
    geom_histogram(aes(y = ..density..), binwidth = 1, 
                   color = "#36B598", fill = "white") + 
    labs(x = "Manufacturing Year", y = "Density", 
         title = "Histogram of Vehicle Manufacturing Year") +
    scale_x_continuous(breaks = seq(1980, 2030, 5)) + 
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.line = element_line(linewidth = 1.2, colour = "grey80"),
          text = element_text(family = "Calibri")) + 
    facet_grid(fatal ~ .)
#ggsave("Histogram of Vehicle Manufacturing Year.jpg")

#Vehicle Body Style
BodyStyleTable = table(Data$VEHICLE_BODY_STYLE, Data$fatal)
BodyStyleTable = cbind(BodyStyleTable, 
                       BodyStyleTable[,2] / (BodyStyleTable[,1] + BodyStyleTable[,2]))
colnames(BodyStyleTable) = c("Non-Fatal", "Fatal", 
                             "Proportion of Fatal Crashes")
BodyStyleTable

#Vehicle Make
MakeTable = table(Data$VEHICLE_MAKE, Data$fatal)
MakeTable = cbind(MakeTable, 
                  MakeTable[,2] / (MakeTable[,1] + MakeTable[,2]))
colnames(MakeTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
MakeTable

#Vehicle Type
TypeTable = table(Data$VEHICLE_TYPE, Data$fatal)
TypeTable = cbind(TypeTable, 
                  TypeTable[,2] / (TypeTable[,1] + TypeTable[,2]))
colnames(TypeTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
TypeTable

#Fuel Type
FuelTable = table(Data$FUEL_TYPE, Data$fatal)
FuelTable = cbind(FuelTable, 
                  FuelTable[,2] / (FuelTable[,1] + FuelTable[,2]))
colnames(FuelTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
FuelTable

#Owner State
StateTable = table(Data$OWNER_STATE, Data$fatal)
StateTable = cbind(StateTable, 
                   StateTable[,2] / (StateTable[,1] + StateTable[,2]))
colnames(StateTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
StateTable


################################# Accident Data ################################ 
#Histogram of Speed Zones
Data %>%
  ggplot(aes(x = SPEED_ZONE)) + 
    geom_histogram(aes(y = ..density..), binwidth = 10, 
                   color = "#36B598", fill = "white") + 
    labs(x = "Speed Zone", y = "Density",
         title = "Histogram of Speed Zones") +
    scale_x_continuous(breaks = seq(40, 120, 10)) + 
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.line = element_line(linewidth = 1.2, colour = "grey80"),
          text = element_text(family = "Calibri")) + 
    facet_grid(fatal ~ .)
#ggsave("Histogram of Speed Zones.jpg")

#Accident Frequency by Time and Day
Data %>%
  mutate(ACCIDENTTIME = hms(ACCIDENTTIME),
         ACCIDENTHR = hour(ACCIDENTTIME),
         DAY_OF_WEEK) %>%
  group_by(DAY_OF_WEEK, ACCIDENTHR) %>%
  count() %>%
  ggplot(aes(x = ACCIDENTHR, y = n, 
                    colour = DAY_OF_WEEK, group = DAY_OF_WEEK)) +
  geom_line() + 
  geom_point()+
  scale_colour_manual(values=c(Monday = "#000D53", Tuesday = "#6571B6", 
                               Wednesday = "#E0B5FF", Thursday = "#059D61",
                               Friday = "#46E18F", Saturday = "#E12B7B", 
                               Sunday = "#E0681A")) + 
  labs(y = "Frequency", x = "Hour (24hr time)", colour = "Day of Week", 
       title = "Frequency of Accidents by Time of Day, Each Day of the Week") +
  scale_x_continuous(breaks = seq(0, 24, 1)) +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.line = element_line(linewidth = 1, colour = "grey80"),
        text = element_text(family = "Calibri"))
#ggsave("Frequency of Accidents by Time of Day, Each Day of the Week.jpg")

#Fatal Accidents by Time and Day
Data %>% 
  filter(fatal == TRUE) %>%
  mutate(ACCIDENTTIME = hms(ACCIDENTTIME),
         ACCIDENTHR = hour(ACCIDENTTIME),
         DAY_OF_WEEK) %>%
  group_by(DAY_OF_WEEK, ACCIDENTHR) %>%
  count() %>%
  ggplot(aes(x = ACCIDENTHR, y = n, 
             colour = DAY_OF_WEEK, group = DAY_OF_WEEK)) +
  geom_line() + 
  geom_point()+
  scale_colour_manual(values=c(Monday = "#000D53", Tuesday = "#6571B6", 
                               Wednesday = "#E0B5FF", Thursday = "#059D61",
                               Friday = "#46E18F", Saturday = "#E12B7B", 
                               Sunday = "#E0681A")) + 
  labs(y = "Frequency", x = "Hour (24hr time)", colour = "Day of Week", 
       title = "Frequency of Fatal Accidents by Time of Day, Each Day of the Week") +
  scale_x_continuous(breaks = seq(0, 24, 1)) +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.line = element_line(linewidth = 1, colour = "grey80"),
        text = element_text(family = "Calibri"))
#ggsave("Frequency of Fatal Accidents by Time of Day, Each Day of the Week.jpg")

#Histogram of Months
Data %>%
  ggplot(aes(x = MONTH)) + 
    geom_histogram(aes(y = ..density..), binwidth = 1, 
                   color = "#36B598", fill = "white") + 
    labs(x = "Month", y = "Density", 
         title = "Monthly Variation of Fatal and Non-Fatal Accidents") +
    scale_x_continuous(breaks = seq(1, 12, 1)) + 
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.line = element_line(linewidth = 1.2, colour = "grey80"),
          text = element_text(family = "Calibri")) + 
    facet_grid(fatal ~ .)
#ggsave("Monthly Variation of Fatal and Non-Fatal Accidents.jpg")

#
AccidentTyTable = table(Data$ACCIDENT_TYPE, Data$fatal)
AccidentTyTable = cbind(AccidentTyTable, AccidentTyTable[,2] / (AccidentTyTable[,1] + AccidentTyTable[,2]))
colnames(AccidentTyTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
AccidentTyTable




#Mosaic Plot of Road Geometries
Data %>%
  ggplot() +
    geom_mosaic(aes(x = product(fatal, ROAD_GEOMETRY), fill = fatal), alpha = 0.5) +
    labs(x = "Road Geometry", y = "Fatal", fill = "Fatal", 
         title = "Fatalities for Road Geometries") +
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.line = element_line(linewidth = 1.2, colour = "grey80"),
          text = element_text(family = "Calibri"))
#ggsave("Fatalities for Road Geometries.jpg")

RGeomTable = table(Data$ROAD_GEOMETRY, Data$fatal)
RGeomTable = cbind(RGeomTable, RGeomTable[,2] / (RGeomTable[,1] + RGeomTable[,2]))
colnames(RGeomTable) = c("Non-Fatal", "Fatal", "Proportion of Fatal Crashes")
RGeomTable

#Speed Zone and Road Geometries
Data$SPEED_ZONE = as_factor(Data$SPEED_ZONE)
DataFatal$SPEED_ZONE = as_factor(DataFatal$SPEED_ZONE)

prop1 = count(DataFatal, SURFACE_COND, SPEED_ZONE, .drop = FALSE)$n / 
  count(Data, SURFACE_COND, SPEED_ZONE, .drop = FALSE)$n

ggplot(cbind(count(Data, SURFACE_COND, SPEED_ZONE), prop1), 
       aes(x = SURFACE_COND, y = SPEED_ZONE)) + 
  geom_tile(aes(fill = prop1)) +
  scale_fill_gradient(low = "#36B598", high = "white") + 
  labs(x = "Surface Conditions", y = "Speed Zone", fill = "Fatality Rate", 
       title = "Proportion of Fatal Crashes for Speed Zone and Surface Conditions") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.line = element_line(linewidth = 1.2, colour = "grey80"),
        text = element_text(family = "Calibri"))
#ggsave("Proportion of Fatal Crashes for Speed Zone and Surface Conditions.jpg")

#Speed Zone and Atmosphere Conditions
prop2 = count(DataFatal, ATMOSPH_COND, SPEED_ZONE, .drop = FALSE)$n / 
  count(Data, ATMOSPH_COND, SPEED_ZONE, .drop = FALSE)$n

ggplot(cbind(count(Data, ATMOSPH_COND, SPEED_ZONE), prop2), 
       aes(x = ATMOSPH_COND, y = SPEED_ZONE)) + 
  geom_tile(aes(fill = prop2)) +
  scale_fill_gradient(low = "#36B598", high = "white") + 
  labs(x = "Atmosphere Conditions", y = "Speed Zone", fill = "Fatality Rate",
       title = "Proportion of Fatal Crashes for Speed Zone and Atmosphere Conditions") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.line = element_line(linewidth = 1.2, colour = "grey80"),
        text = element_text(family = "Calibri"))
#ggsave("Proportion of Fatal Crashes for Speed Zone and Atmosphere Conditions.jpg")

#Speed Zone and Road Surface Type
prop3 = count(DataFatal, ROAD_SURFACE_TYPE, SPEED_ZONE, .drop = FALSE)$n / 
  count(Data, ROAD_SURFACE_TYPE, SPEED_ZONE, .drop = FALSE)$n

ggplot(cbind(count(Data, ROAD_SURFACE_TYPE, SPEED_ZONE), prop3), 
       aes(x = ROAD_SURFACE_TYPE, y = SPEED_ZONE)) + 
  geom_tile(aes(fill = prop3)) +
  scale_fill_gradient(low = "#36B598", high = "white") + 
  labs(x = "Road Surface Type", y = "Speed Zone", fill = "Fatality Rate",
       title = "Proportion of Fatal Crashes for Speed Zone and Road Surface") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.line = element_line(linewidth = 1.2, colour = "grey80"),
        text = element_text(family = "Calibri"))
#ggsave("Proportion of Fatal Crashes for Speed Zone and Road Surface.jpg")

#Speed Zone and Light Conditions
prop4 = count(DataFatal, LIGHT_CONDITION, SPEED_ZONE, .drop = FALSE)$n / 
  count(Data, LIGHT_CONDITION, SPEED_ZONE, .drop = FALSE)$n

ggplot(cbind(count(Data, LIGHT_CONDITION, SPEED_ZONE), prop4), 
       aes(x = LIGHT_CONDITION, y = SPEED_ZONE)) + 
  geom_tile(aes(fill = prop4)) +
  scale_fill_gradient(low = "#36B598", high = "white") + 
  labs(x = "Light Conditions", y = "Speed Zone", fill = "Fatality Rate",
       title = "Proportion of Fatal Crashes for Speed Zone and Light Conditions") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.line = element_line(linewidth = 1.2, colour = "grey80"),
        text = element_text(family = "Calibri"))
#ggsave("Proportion of Fatal Crashes for Speed Zone and Light Conditions.jpg")


################################################################################
########################## ACTL3142 - Main Assignment ##########################
######################## Fatal Road Crashes in Victoria ########################
###################### Part I - Exploratory Data Analysis ######################
################################################################################



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

