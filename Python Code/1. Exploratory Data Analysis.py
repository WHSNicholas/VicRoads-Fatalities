# -------------------------------------------------------------------------------------------------------------------- #
#                                           VicRoads Motor Fatalities Model                                            #
#                                              Exploratory Data Analysis                                               #
# -------------------------------------------------------------------------------------------------------------------- #
"""
Title: VicRoads Motor Fatalities Model
Script: Exploratory Data Analysis

Authors: Nicholas Wong
Creation Date: 11th December 2024
Modification Date: 11th December 2024

Purpose: This script explores the relationship between road fatalities and driver/accident profiles. We use data
         publicly available from VicRoad to perform data analysis as well as preparing the data for the development of a
         predictive model using machine learning techniques.

Dependencies: pandas

Instructions: Ensure that the working directory is set to VicRoads-Fatalities

Data Sources: VicRoad Data obtained from https://discover.data.vic.gov.au/dataset/victoria-road-crash-data
- Accident Data
- Vehicle Data
- Accident Event Data
- Atmospheric Condition Data
- Sub DCA Data
- Person Data
- Node Data
- Road Surface Condition Data
- Accident Location Data

Fonts: "CMU Serif.ttf"

Table of Contents:
1. Data Integration
  1.1. Preamble
  1.2. Importing Data
2. Data Cleaning
  2.1. Structuring Data
  2.2. NA Cleaning
3. Data Exploration
4. Data Transformation
  4.1. Transformation
  4.2. Dimension Reduction
"""

# 1. Data Integration --------------------------------------------------------------------------------------------------
# 1.1. Preamble --------------------------------------------------------------------------------------------------------
# Required Packages
import pandas as pd

# 1.2. Importing Data --------------------------------------------------------------------------------------------------
# Importing Data
accident = pd.read_csv("Data/accident.csv")
vehicle = pd.read_csv("Data/vehicle.csv")
accident_event = pd.read_csv("Data/accident_event.csv")
atmospheric_cond = pd.read_csv("Data/atmospheric_cond.csv")
sub_dca = pd.read_csv("Data/sub_dca.csv")
person = pd.read_csv("Data/person.csv")
node = pd.read_csv("Data/node.csv")
road_surface_cond = pd.read_csv("Data/road_surface_cond.csv")
accident_location = pd.read_csv("Data/accident_location.csv")




# 2. Data Cleaning -----------------------------------------------------------------------------------------------------
# 2.1. Renaming Columns ------------------------------------------------------------------------------------------------
accident.columns = accident.columns.str.lower().str.replace(' ', '_')
accident.rename(columns={"accident_no": "accident_id"}, inplace=True)


# 2.2. Dropping Columns ------------------------------------------------------------------------------------------------


# 2.3. Data Formatting ------------------------------------------------------------------------------------------------




# 2.4. Structuring Data ------------------------------------------------------------------------------------------------





# 2.5. NA Cleaning -----------------------------------------------------------------------------------------------------




# 3. Data Exploration --------------------------------------------------------------------------------------------------



# 4. Data Transformation -----------------------------------------------------------------------------------------------
# 4.1. Transformation --------------------------------------------------------------------------------------------------


# 4.2. Dimension Reduction ---------------------------------------------------------------------------------------------





# -------------------------------------------------------------------------------------------------------------------- #
#                                           VicRoads Motor Fatalities Model                                            #
#                                              Exploratory Data Analysis                                               #
# -------------------------------------------------------------------------------------------------------------------- #