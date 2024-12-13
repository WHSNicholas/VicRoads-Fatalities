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

dataframes = [
    accident,
    vehicle,
    accident_event,
    atmospheric_cond,
    sub_dca,
    person,
    node,
    road_surface_cond,
    accident_location
]


# 2. Data Cleaning -----------------------------------------------------------------------------------------------------
# 2.1. Renaming Columns ------------------------------------------------------------------------------------------------
def snake_case(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')

def rename(df):
    df.rename(
    columns={
        'accident_no': 'accident_id',
        'no_of_vehicles': 'vehicles',
        'no_persons_killed': 'persons_killed',
        'no_persons_inj_2': 'persons_inj_a',
        'no_persons_inj_3': 'persons_inj_b',
        'no_persons': 'persons',
        'vehicle_year_manuf': 'vehicle_year',
        'road_surface_type': 'road_surface',
        'reg_state': 'registration_state',
        'construction_type': 'construction',
        'fuel_type': 'fuel',
        'no_of_wheels': 'wheels',
        'no_of_cylinders': 'cylinders',
        'total_no_occupants': 'occupants',
        'towed_away_flag': 'towed_away',
        'event_seq_no': 'event_seq',
        'ejected_code': 'ejected',
        'node_id': 'node',
        'road_route_1': 'road_route',
        'road_type_int': 'road_type_intersection',
        'road_name_int': 'road_name_intersection'
    },
    inplace=True
)

for df in dataframes:
    snake_case(df)
    rename(df)



# 2.2. Dropping Columns ------------------------------------------------------------------------------------------------
accident.drop(
    labels=['accident_type_desc', 'day_week_desc', 'dca_desc', 'no_persons_not_inj', 'road_geometry_desc'],
    axis = 1,
    inplace=True
)

vehicle.drop(
    labels=['road_surface_type_desc', 'vehicle_power', 'vehicle_type_desc', 'vehicle_colour_1', 'vehicle_colour_2',
            'traffic_control_desc'],
    axis = 1,
    inplace=True
)

accident_event.drop(
    labels=['event_type_desc', 'vehicle_1_coll_pt_desc', 'vehicle_2_coll_pt_desc', 'person_id', 'object_type_desc'],
    axis = 1,
    inplace=True
)

atmospheric_cond.drop(
    labels=['atmosph_cond_seq', 'atmosph_cond_desc'],
    axis = 1,
    inplace=True
)

sub_dca.drop(
    labels=['sub_dca_code_desc', 'sub_dca_seq'],
    axis = 1,
    inplace=True
)

person.drop(
    labels=['inj_level_desc', 'road_user_type_desc'],
    axis = 1,
    inplace=True
)

road_surface_cond.drop(
    labels=['surface_cond_desc', 'surface_cond_seq'],
    axis = 1,
    inplace=True
)




# 2.3. Data Formatting ------------------------------------------------------------------------------------------------
# Creating the modelling dataframe
vicroad_df = pd.DataFrame(
    {
        'id': vehicle.accident_id + '-' + vehicle.vehicle_id,
        'accident_id': vehicle.accident_id,
        'vehicle_id': vehicle.vehicle_id
    }
)

# One-hot Encoding
event_dummies = pd.get_dummies(
    accident_event[['accident_id', 'event_type', 'vehicle_1_coll_pt', 'vehicle_2_coll_pt']],
    columns=['event_type', 'vehicle_1_coll_pt', 'vehicle_2_coll_pt'],
    prefix=['event_type', 'vehicle_1_coll_pt', 'vehicle_2_coll_pt']
).groupby('accident_id').max().reset_index()

atmospheric_dummies = pd.get_dummies(
    atmospheric_cond,
    columns=['atmosph_cond'],
    prefix=['atmosph_cond']
).groupby('accident_id').max().reset_index()

sub_dca_dummies = pd.get_dummies(
    sub_dca,
    columns=['sub_dca_code'],
    prefix=['sub_dca_code']
).groupby('accident_id').max().reset_index()

road_surf_cond_dummies = pd.get_dummies(
    road_surface_cond,
    columns=['surface_cond'],
    prefix=['surface_cond']
).groupby('accident_id').max().reset_index()

# Merge with accident
vicroad_df = pd.merge(
    vicroad_df,
    accident,
    on='accident_id',
    how='left'
)

# Merge with vehicle
vicroad_df = pd.merge(
    vicroad_df,
    vehicle,
    on=['accident_id', 'vehicle_id'],
    how='left'
)

accident_event.vehicle_2_coll_pt.unique()

# Merge with accident_event
vicroad_df = pd.merge(
    vicroad_df,
    event_dummies,
    on='accident_id',
    how='left'
)

# Merge with atmospheric_cond
vicroad_df = pd.merge(
    vicroad_df,
    atmospheric_dummies,
    on='accident_id',
    how='left'
)


# Merge with sub_dca
vicroad_df = pd.merge(
    vicroad_df,
    sub_dca_dummies,
    on='accident_id',
    how='left'
)

# Merge with person
person_driver = person[person.seating_position == 'D']

vicroad_df = pd.merge(
    vicroad_df,
    person_driver[~person_driver.duplicated(subset=['accident_id', 'vehicle_id'], keep=False)],
    on=['accident_id', 'vehicle_id'],
    how='left'
)


# Merge with node
node_unique = node.drop('accident_id', axis = 1).groupby('node', as_index=False).agg(lambda x: x.iloc[0])

vicroad_df = pd.merge(
    vicroad_df,
    node_unique,
    on=['node'],
    how='left'
)


# Merge with road_surface_cond
vicroad_df = pd.merge(
    vicroad_df,
    road_surf_cond_dummies,
    on=['accident_id'],
    how='left'
)

# Merge with road_surface_cond
vicroad_df = pd.merge(
    vicroad_df,
    accident_location.drop(['node'], axis=1),
    on=['accident_id'],
    how='left'
)



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