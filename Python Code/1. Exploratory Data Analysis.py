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

# ----------------------------------------------------------------------------------------------------------------------
# 1. Data Integration
# ----------------------------------------------------------------------------------------------------------------------

# 1.1. Preamble ----------------------------------------------------------------------------------------------
# Required Packages
import pandas as pd

# 1.2. Importing CSV Data ------------------------------------------------------------------------------------
accident            = pd.read_csv("Data/accident.csv")
vehicle             = pd.read_csv("Data/vehicle.csv")
accident_event      = pd.read_csv("Data/accident_event.csv")
atmospheric_cond    = pd.read_csv("Data/atmospheric_cond.csv")
sub_dca             = pd.read_csv("Data/sub_dca.csv")
person              = pd.read_csv("Data/person.csv")
node                = pd.read_csv("Data/node.csv")
road_surface_cond   = pd.read_csv("Data/road_surface_cond.csv")
accident_location   = pd.read_csv("Data/accident_location.csv")

dataframes = {
    'accident': accident,
    'vehicle': vehicle,
    'accident_event': accident_event,
    'atmospheric_cond': atmospheric_cond,
    'sub_dca': sub_dca,
    'person': person,
    'node': node,
    'road_surface_cond': road_surface_cond,
    'accident_location': accident_location
}


# ----------------------------------------------------------------------------------------------------------------------
# 2. Data Cleaning
# ----------------------------------------------------------------------------------------------------------------------

# 2.1. Cleaning Columns --------------------------------------------------------------------------------------
# Renaming column names
def snake_case(dataframe):
    """
    Convert all column names to snake_case.
    """
    dataframe.columns = (
        dataframe.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

def rename_columns(dataframe):
    """
    Rename some commonly known columns if they exist.
    """
    rename_dict = {
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
    }

    for old_col, new_col in rename_dict.items():
        if old_col in dataframe.columns:
            dataframe.rename(columns={old_col: new_col}, inplace=True)

for df in dataframes.values():
    snake_case(df)
    rename_columns(df)

del [accident, vehicle, accident_event, atmospheric_cond, sub_dca, person, node, road_surface_cond, accident_location]
del df

# Dropping columns
dataframes['accident'].drop(
    labels=['accident_type_desc', 'day_week_desc', 'dca_desc', 'no_persons_not_inj', 'road_geometry_desc'],
    axis=1,
    inplace=True
)

dataframes['vehicle'].drop(
    labels=['road_surface_type_desc', 'vehicle_power', 'vehicle_type_desc', 'vehicle_colour_1', 'vehicle_colour_2',
            'traffic_control_desc'],
    axis=1,
    inplace=True
)

dataframes['accident_event'].drop(
    labels=['event_type_desc', 'vehicle_1_coll_pt_desc', 'vehicle_2_coll_pt_desc', 'person_id', 'object_type_desc'],
    axis=1,
    inplace=True
)

dataframes['atmospheric_cond'].drop(
    labels=['atmosph_cond_seq', 'atmosph_cond_desc'],
    axis=1,
    inplace=True
)

dataframes['sub_dca'].drop(
    labels=['sub_dca_code_desc', 'sub_dca_seq'],
    axis=1,
    inplace=True
)

dataframes['person'].drop(
    labels=['inj_level_desc', 'road_user_type_desc'],
    axis=1,
    inplace=True
)

dataframes['road_surface_cond'].drop(
    labels=['surface_cond_desc', 'surface_cond_seq'],
    axis=1,
    inplace=True
)


# 2.2. Data Formatting ---------------------------------------------------------------------------------------
# Creating merged dataframe
vicroad_df = pd.DataFrame(
    {
        'id': dataframes['vehicle']['accident_id'].astype(str) + '-' + dataframes['vehicle']['vehicle_id'].astype(str),
        'accident_id': dataframes['vehicle']['accident_id'],
        'vehicle_id': dataframes['vehicle']['vehicle_id']
    }
)

# One-Hot Encoding
event_dummies = (
    pd.get_dummies(
        dataframes['accident_event'][['accident_id', 'event_type', 'vehicle_1_coll_pt', 'vehicle_2_coll_pt']],
        columns=['event_type', 'vehicle_1_coll_pt', 'vehicle_2_coll_pt'],
        prefix=['event_type', 'vehicle_1_coll_pt', 'vehicle_2_coll_pt']
    )
    .groupby('accident_id').max().reset_index()
)

atmospheric_dummies = (
    pd.get_dummies(dataframes['atmospheric_cond'], columns=['atmosph_cond'], prefix=['atmosph_cond'])
    .groupby('accident_id').max().reset_index()
)

sub_dca_dummies = (
    pd.get_dummies(dataframes['sub_dca'], columns=['sub_dca_code'], prefix=['sub_dca_code'])
    .groupby('accident_id').max().reset_index()
)

road_surf_cond_dummies = (
    pd.get_dummies(dataframes['road_surface_cond'], columns=['surface_cond'], prefix=['surface_cond'])
    .groupby('accident_id').max().reset_index()
)

person_driver = dataframes['person'][dataframes['person']['seating_position'] == 'D']
person_driver = person_driver[~person_driver.duplicated(subset=['accident_id', 'vehicle_id'], keep=False)]

node_unique = (
    dataframes['node']
    .drop('accident_id', axis=1, errors='ignore')
    .groupby('node', as_index=False)
    .agg(lambda x: x.iloc[0])
)

# 2.4. Structuring Data --------------------------------------------------------------------------------------
vicroad_df = (
    vicroad_df
    .merge(dataframes['accident'], on='accident_id', how='left')
    .merge(dataframes['vehicle'], on=['accident_id','vehicle_id'], how='left')
    .merge(event_dummies, on='accident_id', how='left')
    .merge(atmospheric_dummies, on='accident_id', how='left')
    .merge(sub_dca_dummies, on='accident_id', how='left')
    .merge(person_driver, on=['accident_id','vehicle_id'], how='left')
    .merge(node_unique, on='node', how='left')
    .merge(road_surf_cond_dummies, on='accident_id', how='left')
    .merge(dataframes['accident_location'].drop('node', axis=1, errors='ignore'), on='accident_id', how='left')
)

del [event_dummies, atmospheric_dummies, sub_dca_dummies, road_surf_cond_dummies, person_driver, node_unique]


# 2.5. NA Cleaning -------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# 3. Data Exploration
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 4. Data Transformation
# ----------------------------------------------------------------------------------------------------------------------

# 4.1. Transformation ----------------------------------------------------------------------------------------


# 4.2. Dimension Reduction -----------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------- #
#                                           VicRoads Motor Fatalities Model                                            #
#                                              Exploratory Data Analysis                                               #
# -------------------------------------------------------------------------------------------------------------------- #