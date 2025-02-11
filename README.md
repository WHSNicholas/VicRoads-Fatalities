# VicRoads Fatalities
## Project Background
_VicRoads_ is a government agency which manages the road network, motor registration, and ensuring safe and efficient transport systems. In addition to road management, VicRoads is also responsible for promoting road safety. Under _Lyra Technologies_, our goal was twofold:
1. to **gather insights** about which driver and vehicle profiles would be more likely to be involved in fatal accidents, and
2. to create a **predictive model** for whether a certain accident, given the location, vehicle, and driver data would be fatal, meaning at least one person died as a result of the accident.
This project would allow clients to understand which drivers are high-risk and to implement data-driven policies and/or to create a marketing campaign to inform and educate road users on safety related issues. It would also allow for the prediction of the fatality of accidents which could provide insights to cracks within first-responders and the emergency health systems in Victoria, allowing the industry to better respond to road-related incidents. 

## Executive Summary
The xGBoost model had the best model performance and yielded an **AUC of 0.9510**. The model was trained on _____ datapoints with _____ features on the accident, vehicle, person and geographic data provided from VicRoads. We performed data analysis and wrangling including **Principal Components Analysis** (PCA) for dimension reduction, **Synthetic Minority Oversampling Techniques** (SMOTE) for dealing with a 1.5% minority class imbalance, and **Bayesian Search** for hyperparameter tuning. We trained a Feedforward Neural Network (FNN) and an xGBoost model and found the xGBoost model outperformed the other models. 

The top 10 most important variables in determining whether an accident was fatal are:
1. Degree of urbanisation
2. Police attendence 
3. Vehicle towed away
4. Vehicle level of damage
5. Vehicle make
6. Accident type
7. Vehicle type
8. Speed zone
9. Caught fire
10. DCA code

## Data Exploration and Insights
### Data Sources and Wrangling
The data was attained from VicRoads as a publically available dataset [here](https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data). It contains data from 2012 to 2023 including _accident_ (basic accident details, time, severity, location), _person_ (person based details, age, gender etc), _vehicle_ (vehicle based data, vehicle type, make etc), _accident_event_ (sequence of events e.g. left road, rollover, caught fire), _road_surface_cond_ (whether road was wet, dry, icy etc), _atmospheric_cond_ (rain, winds etc), _sub_dca_ (detailed codes describing accident), _accident_node_ (master location table), and _Node Table_ with Lat/Long references.

The main difficulty was with deciding how to properly join the data from each individual .csv file so that modelling could be done. We decided to use the accident_id and vehicle_id as our identifiers in the dataset - that is, each row corresponds to each car that was involved in each accident and the driver's details were kept. This meant that any event-level details (such as what parts the car collided with, and the sequence of events) were maintained using binary encoding (e.g. 1 if the car hit an object, and 0 otherwise). 

The next major issue was the handling of missing and NA values in the dataset. Some columns had an abundance of NA's which we decided to keep as a separate 'Unknown' category, as neither imputation nor removal seemed valid. However for many columns both categorical and numerical, we opted to utilise xGBoost for regression and classification to impute the missing values. We selected the features that were highly relevant to each column to be imputed and utilised Bayesian Search for the hyperparameters. The choice of whether to retain the NA's as a separate category depended on our intuition about whether or not certain variables were dependent on the data available.

### Key Insights
The general methodology for gathering insights required that we realise we could not infer the distribution of certain variables by their availability in our dataset because we could not know whether the data was a complete record of all accidents that occured within this time frame, as there may have been missing records, or unavailable data not included in the data. For example, we would hesitate to claim that because there was 40% females in the data available, that therefore males are more likely to be involved in accidents. Instead, we aimed to understand whether males or females were more likely to be involved in fatal accidents - that is, for example, __% of men who had been involved in an accident were involved in a fatal accident, whereas that number was __% for females. 

From the data visualisation found in the Graphs section, we make the following observations with the numbers (1) giving reference to the graphs
- There was a seasonal trend in 2020-2022 which corresponded to a drop in accidents during lockdown periods (2)
- The Dec-Jan period has less accidents than the rest of the year (3)
- There are holiday specific averages that deviate from the standard
- More accidents happen at peak hours (4)
- Weekends do not follow the same pattern as weekdays (5)
- Although less accidents happen in the late nights (12am-4am), they are more fatal (4 & 9)
- Severity in the late nights is the same across weekends and weekdays (10)
- Collision with fixed objects are the most fatal accident type (11)
- Weekends are more fatal than weekdays (12)
- Accidents that happen in the dark without street lights are the most fatal (13)
- Multi-car (2+) accidents are less fatal than single car accidents (14)
- Accidents that occur not at intersections are most fatal (16)
- Accidents at higher speeds are more fatal (17)
- Accidents involving large vehicles (prime movers) are more fatal (23)
- Accidents involving vehicles with large seating capacities (40-50) are more likely to be fatal (27)
- Overtaking and going the wrong way are the most fatal types of accidents (31)
- Accidents where the vehicle caught fire are significantly more fatal (32)
- Accidents where the top/roof was the initial impact are most fatal whereas rear end collisions are the least fatal (34)
- Accidents with no traffic control are most fatal (36)
- Males are more likely to be involved in fatal accidents than women (40)
- If seatbelt or helmets are not worn, the accident is more likely to be fatal (42)
- If the driver was ejected, the accident is more likely to be fatal (46)

![Alt text]('Graphs/48. Fatal Accidents in Victoria (VicGrid94).png')


## Machine Learning Methods



## Results and Recommendations
