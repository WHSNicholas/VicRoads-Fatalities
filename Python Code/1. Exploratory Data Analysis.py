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
import numpy as np

# Settings
pd.set_option('display.max_rows', None)

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

        # Accident Data
        'accident_type_desc': 'accident_type',
        'day_week_desc': 'day',
        'no_of_vehicles': 'vehicles',
        'no_persons_killed': 'persons_killed',
        'no_persons_inj_2': 'persons_inj_a',
        'no_persons_inj_3': 'persons_inj_b',
        'no_persons': 'persons',
        'road_geometry_desc': 'road_geometry',

        # Vehicle Data
        'vehicle_year_manuf': 'vehicle_year',
        'road_surface_type_desc': 'road_surface',
        'reg_state': 'registration_state',
        'vehicle_type_desc': 'vehicle_type',
        'construction_type': 'construction',
        'fuel_type': 'fuel',
        'no_of_wheels': 'wheels',
        'no_of_cylinders': 'cylinders',
        'total_no_occupants': 'occupants',
        'towed_away_flag': 'towed_away',
        'traffic_control_desc': 'traffic_control',

        # Accident Event Data
        'event_seq_no': 'event_seq',
        'event_type_desc': 'event_type',
        'vehicle_1_coll_pt_desc': 'veh_1_coll',
        'vehicle_2_coll_pt_desc': 'veh_2_coll',
        'object_type_desc': 'object_type',

        # Atmospheric Condition Data
        'atmosph_cond_desc': 'atmosph_cond',

        # Person Data
        'inj_level_desc': 'inj_level',
        'road_user_type_desc': 'road_user',
        'licence_state': 'license_state',
        'ejected_code': 'ejected',

        # Node Data
        'node_id': 'node',

        # Road Surface Condition Data
        'surface_cond_desc': 'surface_cond',

        # Accident Location
        'road_route_1': 'road_route',
        'road_type_int': 'road_type_intersection',
        'road_name_int': 'road_name_intersection'
    }

    for old_col, new_col in rename_dict.items():
        if old_col in dataframe.columns:
            dataframe.rename(columns={old_col: new_col}, inplace=True)


# Dropping columns
dataframes['accident'].drop(
    labels=['ACCIDENT_TYPE', 'DAY_OF_WEEK', 'DCA_DESC', 'NO_PERSONS_NOT_INJ', 'ROAD_GEOMETRY'],
    axis=1,
    inplace=True
)

dataframes['vehicle'].drop(
    labels=['ROAD_SURFACE_TYPE', 'VEHICLE_POWER', 'VEHICLE_TYPE', 'VEHICLE_COLOUR_1', 'VEHICLE_COLOUR_2',
            'TRAFFIC_CONTROL'],
    axis=1,
    inplace=True
)

dataframes['accident_event'].drop(
    labels=['EVENT_TYPE', 'VEHICLE_1_COLL_PT', 'VEHICLE_2_COLL_PT', 'PERSON_ID'],
    axis=1,
    inplace=True
)

dataframes['atmospheric_cond'].drop(
    labels=['ATMOSPH_COND_SEQ', 'ATMOSPH_COND'],
    axis=1,
    inplace=True
)

dataframes['sub_dca'].drop(
    labels=['SUB_DCA_CODE_DESC', 'SUB_DCA_SEQ'],
    axis=1,
    inplace=True
)

dataframes['person'].drop(
    labels=['INJ_LEVEL', 'ROAD_USER_TYPE'],
    axis=1,
    inplace=True
)

dataframes['road_surface_cond'].drop(
    labels=['SURFACE_COND', 'SURFACE_COND_SEQ'],
    axis=1,
    inplace=True
)

for df in dataframes.values():
    snake_case(df)
    rename_columns(df)

del [accident, vehicle, accident_event, atmospheric_cond, sub_dca, person, node, road_surface_cond, accident_location]
del df


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
        dataframes['accident_event'][['accident_id', 'event_type', 'veh_1_coll', 'veh_2_coll']],
        columns=['event_type', 'veh_1_coll', 'veh_2_coll'],
        prefix=['event_type', 'veh_1_coll', 'veh_2_coll']
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

# 2.3. Structuring Data --------------------------------------------------------------------------------------
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

vicroad_df.columns = vicroad_df.columns.str.replace(' ', '_').str.lower()

vicroad_df.drop(
    labels=[
        'accident_id', 'vehicle_id', 'construction', 'vehicle_weight', 'carry_capacity', 'cubic_capacity',
        'event_type_not_applicable', 'event_type_not_known', 'veh_1_coll_not_known_or_not_applicable',
        'veh_2_coll_not_known_or_not_applicable', 'atmosph_cond_not_known', 'sub_dca_code_nrq', 'person_id',
        'seating_position', 'surface_cond_unk.'
    ],
    axis=1,
    inplace=True)

del [event_dummies, atmospheric_dummies, sub_dca_dummies, road_surf_cond_dummies, person_driver, node_unique]


# 2.4. NA Cleaning -------------------------------------------------------------------------------------------
# Converting Strings to Category
category = [
    'accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'rma', 'vehicle_dca_code',
    'initial_direction', 'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make', 'vehicle_model',
    'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type', 'caught_fire',
    'initial_impact', 'lamps', 'traffic_control', 'sex', 'age_group', 'inj_level', 'helmet_belt_worn', 'road_user',
    'license_state', 'node_type', 'lga_name', 'lga_name_all', 'deg_urban_name', 'postcode_crash',
    'road_route', 'road_name', 'road_type', 'road_name_intersection', 'road_type_intersection', 'direction_location'
]

vicroad_df['postcode_crash'] = vicroad_df['postcode_crash'].astype('Int64')
vicroad_df['road_route'] = vicroad_df['road_route'].astype('Int64')
vicroad_df[category] = vicroad_df[category].astype('category')

# Converting Columns to Boolean
boolean = [
    col for col in vicroad_df.columns
    if (
            col.startswith('event_type') or
            col.startswith('veh_1_coll') or
            col.startswith('veh_2_coll') or
            col.startswith('atmosph_cond') or
            col.startswith('sub_dca_code') or
            col.startswith('taken_hospital') or
            col.startswith('surface_cond')
    )
]

vicroad_df[boolean] = vicroad_df[boolean].apply(
    lambda col: col.map(lambda x: True if x in ['y', 'yes', 'Y', True] else False)
)
vicroad_df['police_attend'] = vicroad_df['police_attend'].map(lambda x: True if x == 2 else False)
vicroad_df['towed_away'] = vicroad_df['towed_away'].map(lambda x: True if x == 2 else False)
vicroad_df['ejected'] = vicroad_df['ejected'].map(lambda x: True if x == 2 else False)

# Converting Columns to String
vicroad_df['id'] = vicroad_df['id'].astype('string')

# Converting Columns to Datetime
vicroad_df['accident_date'] = pd.to_datetime(vicroad_df['accident_date'])
vicroad_df['accident_time'] = pd.to_datetime(vicroad_df['accident_time'], format='%H:%M:%S', errors='coerce').dt.time

# Converting Columns to Int
integer = ['vehicle_year', 'wheels', 'cylinders', 'seating_capacity', 'tare_weight', 'occupants']
vicroad_df[integer] = vicroad_df[integer].astype('Int64')

vicroad_df.dtypes
del [boolean, category, integer]




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