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
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    root_mean_squared_error,
    r2_score
)
from scipy.stats import uniform, randint
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt
import seaborn as sns


# Settings
pd.set_option('display.max_rows', None)
SEED = 1

# 1.2. Importing CSV Data ------------------------------------------------------------------------------------
accident            = pd.read_csv("Data/VicRoad Data/accident.csv")
vehicle             = pd.read_csv("Data/VicRoad Data/vehicle.csv")
accident_event      = pd.read_csv("Data/VicRoad Data/accident_event.csv")
atmospheric_cond    = pd.read_csv("Data/VicRoad Data/atmospheric_cond.csv")
sub_dca             = pd.read_csv("Data/VicRoad Data/sub_dca.csv")
person              = pd.read_csv("Data/VicRoad Data/person.csv")
node                = pd.read_csv("Data/VicRoad Data/node.csv")
road_surface_cond   = pd.read_csv("Data/VicRoad Data/road_surface_cond.csv")
accident_location   = pd.read_csv("Data/VicRoad Data/accident_location.csv")

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

# Converting Strings to Category
category = [
    'accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'severity', 'rma', 'vehicle_dca_code',
    'initial_direction', 'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make', 'vehicle_model',
    'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type', 'initial_impact',
    'level_of_damage', 'traffic_control', 'sex', 'age_group', 'inj_level', 'helmet_belt_worn', 'road_user',
    'license_state', 'node_type', 'lga_name', 'lga_name_all', 'deg_urban_name', 'postcode_crash', 'road_route',
    'road_name', 'road_type', 'road_name_intersection', 'road_type_intersection', 'direction_location'
]

vicroad_df['postcode_crash'] = vicroad_df['postcode_crash'].astype('Int64')
vicroad_df['road_route'] = vicroad_df['road_route'].astype('Int64')
vicroad_df[category] = vicroad_df[category].astype('category')

vicroad_df['light_condition'] = vicroad_df['light_condition'].cat.rename_categories(
    {
        1: 'Day',
        2: 'Dusk/dawn',
        3: 'Dark (street lights on)',
        4: 'Dark (street lights off)',
        5: 'Dark (no street lights)',
        6: 'Dark (street lights unknown)',
        9: 'Unknown'
    }
)

vicroad_df['severity'] = vicroad_df['severity'].cat.rename_categories(
    {
        1: 'Fatal',
        2: 'Serious injury',
        3: 'Other injury',
        4: 'No injury'
    }
)

vicroad_df['vehicle_dca_code'] = vicroad_df['vehicle_dca_code'].cat.rename_categories(
    {
        1: 'Vehicle 1',
        2: 'Vehicle 2',
        3: 'Not known which vehicle was number 1',
        8: 'Not involved in initial event'
    }
)


vicroad_df['registration_state'] = vicroad_df['registration_state'].astype(str)
vicroad_df['registration_state'] = vicroad_df['registration_state'].replace({'B': pd.NA, 'O': pd.NA, 'Z': pd.NA})
vicroad_df['registration_state'] = vicroad_df['registration_state'].astype('category')
vicroad_df['registration_state'] = vicroad_df['registration_state'].cat.rename_categories(
    {
        'A': 'ACT',
        'B': 'NaN',
        'D': 'NT',
        'N': 'NSW',
        'O': 'NaN',
        'Q': 'QLD',
        'S': 'SA',
        'T': 'TAS',
        'V': 'VIC',
        'W': 'WA'
    }
)

vicroad_df['fuel'] = vicroad_df['fuel'].cat.rename_categories(
    {
        'D': 'Diesel',
        'E': 'Electric',
        'G': 'Gas',
        'M': 'Multi',
        'P': 'Petrol',
        'R': 'Rotary',
        'Z': 'Unknown'
    }
)

vicroad_df['license_state'] = vicroad_df['license_state'].astype(str)
vicroad_df['license_state'] = vicroad_df['license_state'].replace({'B': pd.NA, 'O': pd.NA, 'Z': pd.NA})
vicroad_df['license_state'] = vicroad_df['license_state'].astype('category')
vicroad_df['license_state'] = vicroad_df['license_state'].cat.rename_categories(
    {
        'A': 'ACT',
        'B': 'NaN',
        'D': 'NT',
        'N': 'NSW',
        'O': 'NaN',
        'Q': 'QLD',
        'S': 'SA',
        'T': 'TAS',
        'V': 'VIC',
        'W': 'WA'
    }
)

vicroad_df['driver_intent'] = vicroad_df['driver_intent'].cat.rename_categories(
    {
        1: 'going straight ahead',
        2: 'turning right',
        3: 'turning left',
        4: 'leaving a driveway',
        5: 'U-turning',
        6: 'Changing lanees',
        7: 'Overtaking',
        8: 'Merging',
        9: 'Reversing',
        10: 'Parking or unparking',
        11: 'Parked legally',
        12: 'Parking illegally',
        13: 'Stationary accident',
        14: 'Stationary broken down',
        15: 'Other stationary',
        16: 'Avoiding animals',
        17: 'Slow/stopping',
        18: 'Out of control',
        19: 'Wrong way',
        99: 'Not known'
    }
)

vicroad_df['vehicle_movement'] = vicroad_df['vehicle_movement'].cat.rename_categories(
    {
        1: 'going straight ahead',
        2: 'turning right',
        3: 'turning left',
        4: 'leaving a driveway',
        5: 'U-turning',
        6: 'Changing lanes',
        7: 'Overtaking',
        8: 'Merging',
        9: 'Reversing',
        10: 'Parking or unparking',
        11: 'Parked legally',
        12: 'Parking illegally',
        13: 'Stationary accident',
        14: 'Stationary broken down',
        15: 'Other stationary',
        16: 'Avoiding animals',
        17: 'Slow/stopping',
        18: 'Out of control',
        19: 'Wrong way',
        99: 'Not known'
    }
)

vicroad_df['trailer_type'] = vicroad_df['trailer_type'].cat.rename_categories(
    {
        'A': 'Caravan',
        'B': 'Trailer (general)',
        'C': 'Trailer (boat)',
        'D': 'Horse',
        'E': 'Machinery',
        'F': 'Farm/agricultural equipment',
        'G': 'Not known what is being towed',
        'H': 'Not applicable',
        'I': 'Trailer (exempt)',
        'J': 'Semi trailer',
        'K': 'Pig trailer',
        'L': 'Dog trailer'
    }
)

vicroad_df['initial_impact'] = vicroad_df['initial_impact'].cat.rename_categories(
    {
        '0': 'Towed unit',
        '1': 'Right front corner',
        '2': 'Right side forwards',
        '3': 'Right side rearwards',
        '4': 'Right rear corner',
        '5': 'Left front corner',
        '6': 'Left side forwards',
        '7': 'Left side rearwards',
        '8': 'Left rear corner',
        '9': 'Not known/not applicable',
        'F': 'Front',
        'N': 'None',
        'R': 'Rear',
        'S': 'Sidecar',
        'T': 'Top/roof',
        'U': 'Undercarriage'
    }
)
vicroad_df.loc[vicroad_df['initial_impact'].isin([' ']), 'initial_impact'] = pd.NA

vicroad_df['level_of_damage'] = vicroad_df['level_of_damage'].cat.rename_categories(
    {
        1: 'Minor',
        2: 'Moderate (driveable vehicle)',
        3: 'Moderate (unit towed away)',
        4: 'Major (unit towed away)',
        5: 'Extensive (unrepairable)',
        6: 'Nil damage',
        9: 'Not known'

    }
)

vicroad_df['helmet_belt_worn'] = vicroad_df['helmet_belt_worn'].cat.rename_categories(
    {
        1: 'Seatbelt worn',
        2: 'Seatbelt not worn',
        3: 'Child restraint worn',
        4: 'Child restraint not worn',
        5: 'Seatbelt/restraint not fitted',
        6: 'Crash helmet worn',
        7: 'Crash helmet not worn',
        8: 'Not appropriate',
        9: 'Not known'

    }
)

vicroad_df['node_type'] = vicroad_df['node_type'].cat.rename_categories(
    {
        'I': 'Intersection',
        'N': 'Non intersection',
        'O': 'Off road',
        'U': 'Unknown'
    }
)

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
vicroad_df['caught_fire'] = vicroad_df['caught_fire'].map(lambda x: True if x == 1 else False)
vicroad_df['lamps'] = vicroad_df['lamps'].map(lambda x: True if x == 1 else False)

# Converting Columns to String
vicroad_df['id'] = vicroad_df['id'].astype('string')

# Converting Columns to Datetime
vicroad_df['accident_date'] = pd.to_datetime(vicroad_df['accident_date'])
vicroad_df['accident_time'] = pd.to_datetime(vicroad_df['accident_time'], format='%H:%M:%S', errors='coerce').dt.time

# Converting Columns to Int
integer = ['vehicle_year', 'wheels', 'cylinders', 'seating_capacity', 'tare_weight', 'occupants']
vicroad_df[integer] = vicroad_df[integer].astype('Int64')

del [boolean, category, integer]

# Splitting the Data
vicroad_x_train, vicroad_x_test, vicroad_y_train, vicroad_y_test = train_test_split(
    vicroad_df.drop('persons_killed', axis = 1),
    vicroad_df['persons_killed'],
    test_size=0.3,
    random_state=SEED
)

vicroad_x_test, vicroad_x_val, vicroad_y_test, vicroad_y_val = train_test_split(
    vicroad_x_test,
    vicroad_y_test,
    test_size=0.5,
    random_state=SEED
)


# 2.4. Data Cleaning -------------------------------------------------------------------------------------------
def filter_rows(df):
    df_filtered = df[~df['vehicle_type'].isin([
        'Bicycle', 'Horse (ridden or drawn)', 'Not Known', 'Other Vehicle', 'Tram', 'Motor Scooter', 'Quad Bike',
        'Train', 'Not Applicable'
    ])].copy()

    df_filtered = df_filtered[~df_filtered['lga_name_all'].isna()].copy()
    df_filtered = df_filtered[~df_filtered['caught_fire'].isna()].copy()
    df_filtered = df_filtered[~df_filtered['lamps'].isna()].copy()

    return df_filtered

def clean_outliers(df):
    """
    Cleans outliers and invalid data in the given DataFrame.

    This function performs the following data cleaning tasks:
    1. Replaces invalid speed zone values (777, 888, 999) with NaN.
    2. Corrects unrealistic vehicle years by subtracting 1000 if the year exceeds 2100.
    3. Handles missing vehicle year values:
       - For bicycles, horses, and electric devices with a year of 0, the median vehicle year is used.
       - For other vehicles with a year of 0, replaces with NaN.

    :param df: pandas.DataFrame
        The input DataFrame containing the data to be cleaned. Must include the columns:
        - 'speed_zone': Speed zone data, potentially containing invalid values.
        - 'vehicle_year': Year of the vehicle, potentially containing unrealistic values.
        - 'vehicle_type': Type of the vehicle, used to infer missing years.
    :return: pandas.DataFrame
        The cleaned DataFrame with outliers and invalid data corrected.
    """
    # Speed Zone
    unknown_speeds = [777, 888, 999]
    df.loc[df['speed_zone'].isin(unknown_speeds), 'speed_zone'] = pd.NA

    # Vehicle Year
    df.loc[df['vehicle_year'] > 2100, 'vehicle_year'] -= 1000

    df.loc[
        (df['vehicle_year'] == 0) &
        (df['vehicle_type'] == 'Electric Device'),
        'vehicle_year'
    ] = df['vehicle_year'].median()

    df.loc[df['vehicle_year'] == 0, 'vehicle_year'] = pd.NA

    # Registration State
    df.loc[df['registration_state'].isin(['nan', 'NaN']), 'registration_state'] = pd.NA
    df['registration_state'] = df['registration_state'].cat.remove_unused_categories()

    # Fuel
    df.loc[df['fuel'].isin(['S', 'O']), 'fuel'] = 'Unknown'
    df['fuel'] = df['fuel'].cat.remove_unused_categories()

    # License State
    df.loc[df['license_state'].isin(['nan', 'NaN']), 'license_state'] = pd.NA
    df['license_state'] = df['license_state'].cat.remove_unused_categories()

    # Node Type
    df.loc[df['node_type'].isin([' ']), 'node_type'] = pd.NA
    df['node_type'] = df['node_type'].cat.remove_unused_categories()

    # Road Type
    df['road_type'] = df['road_type'].cat.add_categories('OTHER')
    counts = df['road_type'].value_counts()
    df.loc[df['road_type'].isin(counts[counts<10].index), 'road_type'] = 'OTHER'
    df['road_type'] = df['road_type'].cat.remove_unused_categories()

    # Road Type
    df['road_type_intersection'] = df['road_type_intersection'].cat.add_categories('OTHER')
    counts = df['road_type_intersection'].value_counts()
    df.loc[df['road_type_intersection'].isin(counts[counts < 10].index), 'road_type_intersection'] = 'OTHER'
    df['road_type_intersection'] = df['road_type_intersection'].cat.remove_unused_categories()

    return df


def impute_nans(
        df: pd.DataFrame,
        column: str,
        features: list,
        task: str = 'regression',
        params: dict = None,
        test_size: float = 0.2
):
    """
    Imputes missing values in a specified column using an XGBoost model with provided hyperparameters.

    This function:
    1. Splits the data into training and imputation sets.
    2. Performs a train/validation split on the training set.
    3. Initializes an XGBoost model with the provided parameters.
    4. Trains the model on the training set.
    5. Evaluates and prints training and validation metrics.
    6. Imputes missing values in the target column using the trained model.

    :param df: pandas.DataFrame
        The input DataFrame containing the data.
    :param column: str
        The name of the column with missing values to be imputed.
    :param features: list of str
        The list of feature column names to use for training the model. Should include the target column.
    :param task: str, optional
        Type of task - 'classification' or 'regression'. Default is 'regression'.
    :param params: dict, optional
        Dictionary of XGBoost hyperparameters. If None, default XGBoost parameters are used.
    :param test_size: float, optional
        Proportion of the dataset to include in the validation split. Default is 0.2.

    :return: numpy.ndarray
        An array of predicted values to replace the missing values in the specified column.
    """

    # Validate task
    if task not in ['classification', 'regression']:
        raise ValueError("Task must be either 'classification' or 'regression'.")

    # Data Processing
    impute = df[df[column].isna()][features]
    train = df[~df[column].isna()][features]

    x_train = train.drop(columns=[column])
    y_train = train[column]

    if task == 'classification':
        unique_classes = y_train.nunique()
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

    x_impute = impute.drop(columns=[column])

    # Splitting Data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=SEED)

    # Building Model
    if task == 'regression':
        xgb_model = xgb.XGBRegressor(
            random_state=SEED,
            objective='reg:squarederror' if params is None else params.get('objective', 'reg:squarederror'),
            **({} if params is None else params)
        )
    else:
        if unique_classes == 2:
            default_objective = 'binary:logistic'
            default_eval_metric = 'auc'
        else:
            default_objective = 'multi:softprob'
            default_eval_metric = 'mlogloss'

        xgb_model = xgb.XGBClassifier(
            random_state=SEED,
            objective=default_objective if params is None else params.get('objective', default_objective),
            eval_metric=default_eval_metric if params is None else params.get('eval_metric', default_eval_metric),
            **({} if params is None else params)
        )

    # Training Model
    xgb_model.fit(x_train, y_train)

    # Make predictions on the training set
    y_train_pred = xgb_model.predict(x_train)

    # Evaluate training performance
    if task == 'regression':
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        print(f"Training RMSE imputation of {column}: {train_rmse:.4f}")
    else:
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"Training Accuracy imputation of {column}: {train_accuracy:.4f}")

    # Make predictions on the validation set
    y_val_pred = xgb_model.predict(x_val)

    # Evaluate validation performance
    if task == 'regression':
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        print(f"Validation RMSE imputation of {column}: {val_rmse:.4f}")
    else:
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy imputation of {column}: {val_accuracy:.4f}")

    # Retrain the model on the entire non-missing dataset
    if task == 'classification':
        xgb_model.fit(train.drop(columns=[column]), le.fit_transform(train[column]))
    else:
        xgb_model.fit(train.drop(columns=[column]), train[column])

    imputed_values = xgb_model.predict(x_impute)

    if task == 'classification':
        imputed_values = le.inverse_transform(imputed_values)

    return imputed_values


def clean_nans(df):
    # RMA
    df['rma'] = df['rma'].cat.add_categories('Other')
    df['rma'] = df['rma'].fillna('Other')

    # Speed Zone
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
    ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 200,
        'max_depth': 12,
        'learning_rate': 0.02781409973319061,
        'subsample': 1,
        'colsample_bytree': 0.5296547151341503,
        'gamma': 1.2092971141171853
    }

    df.loc[df['speed_zone'].isna(), 'speed_zone'] = impute_nans(
        df=df,
        column='speed_zone',
        features=features,
        params=params,
        task='regression'
    ).round(-1)

    # Vehicle Year
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone', 'vehicle_year'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
               ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05907766229088168,
        'subsample': 1,
        'colsample_bytree': 0.5,
        'gamma': 0.8301170555761738
    }

    df.loc[df['vehicle_year'].isna(), 'vehicle_year'] = impute_nans(
        df=df,
        column='vehicle_year',
        features=features,
        params=params,
        task='regression'
    ).round()

    # Vehicle DCA Code
    df['vehicle_dca_code'] = df['vehicle_dca_code'].cat.add_categories('Not known')
    df['vehicle_dca_code'] = df['vehicle_dca_code'].fillna('Not known')

    # Initial Direction
    df['initial_direction'] = df['initial_direction'].fillna('NK')

    # Road Surface
    df['road_surface'] = df['road_surface'].fillna('Not known')

    # Registration State
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone', 'vehicle_year', 'registration_state'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
    ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05059645868702228,
        'subsample': 0.7213114396573612,
        'colsample_bytree': 1,
        'gamma': 1.651407961531168
    }

    df.loc[df['registration_state'].isna(), 'registration_state'] = impute_nans(
        df=df,
        column='registration_state',
        features=features,
        params=params,
        task='classification'
    )

    # Vehicle Body Style
    df['vehicle_body_style'] = df['vehicle_body_style'].cat.add_categories('Other')
    df['vehicle_body_style'] = df['vehicle_body_style'].fillna('Other')

    # Vehicle Make
    df['vehicle_make'] = df['vehicle_make'].fillna('OTHR')

    # Vehicle Model
    df['vehicle_model'] = df['vehicle_model'].cat.add_categories('OTHR')
    df['vehicle_model'] = df['vehicle_model'].fillna('OTHR')

    # Fuel
    df['fuel'] = df['fuel'].fillna('Unknown')

    # Wheels
    df.loc[df['wheels'].isna() & (df['vehicle_type'].isin([
        'Station Wagon',
        'Car',
        'Utility',
        'Mini Bus(9-13 seats)',
        'Panel Van',
        'Taxi',
        'Plant machinery and Agricultural equipment',
        'Parked trailers',
        'Electric Device'
    ])), 'wheels'] = 4

    df.loc[df['wheels'].isna() & (df['vehicle_type'].isin(['Motor Cycle', 'Moped'])), 'wheels'] = 2

    df.loc[df['wheels'].isna() & (df['vehicle_type'].isin([
        'Bus/Coach', 'Heavy Vehicle (Rigid) > 4.5 Tonnes',
        'Light Commercial Vehicle (Rigid) <= 4.5 Tonnes GVM',
        'Prime Mover - Single Trailer',
        'Prime Mover B-Double',
        'Prime Mover B-Triple',
        'Prime Mover Only',
        'Rigid Truck(Weight Unknown)',
        'Prime Mover (No of Trailers Unknown)'
    ])), 'wheels'] = 6

    # Cylinders
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone', 'vehicle_year', 'registration_state', 'cylinders'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
    ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 234,
        'max_depth': 4,
        'learning_rate': 0.04453091431697788,
        'subsample': 0.9302149446401943,
        'colsample_bytree': 1,
        'gamma': 0
    }

    df.loc[df['cylinders'].isna(), 'cylinders'] = impute_nans(
        df=df,
        column='cylinders',
        features=features,
        params=params,
        task='regression'
    ).round()

    # Seating Capacity
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone', 'vehicle_year', 'registration_state', 'cylinders', 'seating_capacity'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
               ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 147,
        'max_depth': 10,
        'learning_rate': 0.017608973085763972,
        'subsample': 0.8606478747104616,
        'colsample_bytree': 0.820209399199052,
        'gamma': 1.9536223473997503
    }

    df.loc[df['seating_capacity'].isna(), 'seating_capacity'] = impute_nans(
        df=df,
        column='seating_capacity',
        features=features,
        params=params,
        task='regression'
    ).round()

    # Tare Weight
    df.loc[df['tare_weight'].isna(), 'tare_weight'] = df['tare_weight'].median()

    # Occupants
    df.loc[df['occupants'].isna(), 'occupants'] = df['occupants'].median()

    # Driver Intent
    df.loc[df['driver_intent'].isna(), 'driver_intent'] = 'Not known'

    # Vehicle Movement
    df.loc[df['vehicle_movement'].isna(), 'vehicle_movement'] = 'Not known'

    # Trailer Type
    df.loc[df['trailer_type'].isna(), 'trailer_type'] = 'Not applicable'

    # Initial Impact
    df.loc[df['initial_impact'].isna(), 'initial_impact'] = 'Not known/not applicable'

    # Traffic Control
    df.loc[df['traffic_control'].isna(), 'traffic_control'] = 'Unknown'

    # Sex
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone', 'vehicle_year', 'registration_state', 'cylinders', 'seating_capacity'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
    ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.06010753860728294,
        'subsample': 1,
        'colsample_bytree': 0.5,
        'gamma': 1.6023674383805464
    }

    df.loc[df['sex'].isna(), 'sex'] = impute_nans(
        df=df,
        column='sex',
        features=features,
        params=params,
        task='classification'
    )

    # Age Group
    df.loc[df['age_group'].isna(), 'age_group'] = 'Unknown'

    # Injury Level
    df.loc[df['inj_level'].isna(), 'inj_level'] = 'Not injured'

    # Helmet Belt Worn
    df.loc[df['helmet_belt_worn'].isna(), 'helmet_belt_worn'] = 'Not known'

    # Road User
    df.loc[df['road_user'].isna(), 'road_user'] = 'Drivers'

    # License State
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone', 'vehicle_year', 'registration_state'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
    ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 168,
        'max_depth': 13,
        'learning_rate': 0.011261664948176812,
        'subsample': 0.726487008270275,
        'colsample_bytree': 0.9388425659603135,
        'gamma': 1.797560042213643
    }

    df.loc[df['license_state'].isna(), 'license_state'] = impute_nans(
        df=df,
        column='license_state',
        features=features,
        params=params,
        task='classification'
    )

    # Node Type
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone', 'vehicle_year', 'registration_state', 'cylinders', 'seating_capacity',
                'node_type'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
    ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 300,
        'max_depth': 20,
        'learning_rate': 0.1112167311086204,
        'subsample': 1,
        'colsample_bytree': 1,
        'gamma': 0
    }

    df.loc[df['node_type'].isna(), 'node_type'] = impute_nans(
        df=df,
        column='node_type',
        features=features,
        params=params,
        task='classification'
    )

    # Deg Urban Name
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'road_type', 'speed_zone', 'vehicle_year', 'registration_state', 'cylinders', 'seating_capacity',
                'node_type', 'deg_urban_name'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
    ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 164,
        'max_depth': 15,
        'learning_rate': 0.01,
        'subsample': 1,
        'colsample_bytree': 0.5,
        'gamma': 0
    }

    df.loc[df['deg_urban_name'].isna(), 'deg_urban_name'] = impute_nans(
        df=df,
        column='deg_urban_name',
        features=features,
        params=params,
        task='classification'
    )

    # Road Type
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'speed_zone', 'vehicle_year', 'registration_state', 'cylinders', 'seating_capacity', 'node_type',
                'deg_urban_name', 'road_type'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
    ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.11925942543170741,
        'subsample': 1,
        'colsample_bytree': 0.5,
        'gamma': 0
    }

    df.loc[df['road_type'].isna(), 'road_type'] = impute_nans(
        df=df,
        column='road_type',
        features=features,
        params=params,
        task='classification'
    )

    # Road Type Intersection
    features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
                'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
                'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
                'speed_zone', 'vehicle_year', 'registration_state', 'cylinders', 'seating_capacity', 'node_type',
                'deg_urban_name', 'road_type', 'road_type_intersection'] + [
                   col for col in df.columns
                   if (
                col.startswith('event_type') or
                col.startswith('veh_1_coll') or
                col.startswith('veh_2_coll') or
                col.startswith('atmosph_cond') or
                col.startswith('sub_dca_code') or
                col.startswith('surface_cond')
        )
               ]

    params = {
        'n_jobs': -1,
        'verbosity': 2,
        'enable_categorical': True,
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.05388619528876108,
        'subsample': 1,
        'colsample_bytree': 1,
        'gamma': 0
    }

    df.loc[df['road_type_intersection'].isna(), 'road_type_intersection'] = impute_nans(
        df=df,
        column='road_type_intersection',
        features=features,
        params=params,
        task='classification'
    )


    return df

# Training Data
vicroad_x_train = filter_rows(vicroad_x_train)
vicroad_y_train = vicroad_y_train.loc[vicroad_x_train.index]

vicroad_x_train = clean_outliers(vicroad_x_train)
vicroad_y_train = vicroad_y_train.loc[vicroad_x_train.index]

vicroad_x_train = clean_nans(vicroad_x_train)
vicroad_y_train = vicroad_y_train.loc[vicroad_x_train.index]

# Validation Data
vicroad_x_val = filter_rows(vicroad_x_val)
vicroad_y_val = vicroad_y_val.loc[vicroad_x_val.index]

vicroad_x_val = clean_outliers(vicroad_x_val)
vicroad_y_val = vicroad_y_val.loc[vicroad_x_val.index]

vicroad_x_val = clean_nans(vicroad_x_val)
vicroad_y_val = vicroad_y_val.loc[vicroad_x_val.index]

# Test Data
vicroad_x_test = filter_rows(vicroad_x_test)
vicroad_y_test = vicroad_y_test.loc[vicroad_x_test.index]

vicroad_x_test = clean_outliers(vicroad_x_test)
vicroad_y_test = vicroad_y_test.loc[vicroad_x_test.index]

vicroad_x_test = clean_nans(vicroad_x_test)
vicroad_y_test = vicroad_y_test.loc[vicroad_x_test.index]

# Exporting Cleaned Data
vicroad_x_train.to_csv('Data/Cleaned Data/vicroad_x_train.csv', index=False)
vicroad_x_val.to_csv('Data/Cleaned Data/vicroad_x_val.csv', index=False)
vicroad_x_test.to_csv('Data/Cleaned Data/vicroad_x_test.csv', index=False)

vicroad_y_train.to_csv('Data/Cleaned Data/vicroad_y_train.csv', index=False)
vicroad_y_val.to_csv('Data/Cleaned Data/vicroad_y_val.csv', index=False)
vicroad_y_test.to_csv('Data/Cleaned Data/vicroad_y_test.csv', index=False)

vicroad_x_train.fuel.unique()

# ----------------------------------------------------------------------------------------------------------------------
# 3. Data Exploration
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 4. Data Transformation
# ----------------------------------------------------------------------------------------------------------------------

# 4.1. Transformation ----------------------------------------------------------------------------------------


# 4.2. Dimension Reduction -----------------------------------------------------------------------------------
# TEST
# -------------------------------------------------------------------------------------------------------------------- #
#                                           VicRoads Motor Fatalities Model                                            #
#                                              Exploratory Data Analysis                                               #
# -------------------------------------------------------------------------------------------------------------------- #