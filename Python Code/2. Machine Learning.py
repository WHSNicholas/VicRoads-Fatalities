# -------------------------------------------------------------------------------------------------------------------- #
#                                           VicRoads Motor Fatalities Model                                            #
#                                                  Machine Learning                                                    #
# -------------------------------------------------------------------------------------------------------------------- #
"""
Title: VicRoads Motor Fatalities Model
Script: Machine Learning

Authors: Nicholas Wong
Creation Date: 27th January 2025
Modification Date: 27th January 2025

Purpose: This script builds a predictive model for the fatality of any given accident using various machine learning
         techniques. We use PCA for dimension reduction, SMOTE for imbalanced data, and build XGBoost and Neural Network
         models for classification.

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
  1.2. Importing CSV Data
  1.3. Data Preparation
  1.4. Data Encoding
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.feature_selection import RFE
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from geopy.distance import geodesic
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.version_utils import callbacks

# Settings
pd.set_option('display.max_rows', None)
SEED = 1

# 1.2. Importing CSV Data ------------------------------------------------------------------------------------
dataframes = {}

dataframes['vicroad_x_train'] = pd.read_csv("Data/Cleaned Data/vicroad_x_train.csv")
dataframes['vicroad_x_val'] = pd.read_csv('Data/Cleaned Data/vicroad_x_val.csv')
dataframes['vicroad_x_test'] = pd.read_csv('Data/Cleaned Data/vicroad_x_test.csv')
dataframes['vicroad_y_train'] = pd.read_csv('Data/Cleaned Data/vicroad_y_train.csv')
dataframes['vicroad_y_val'] = pd.read_csv('Data/Cleaned Data/vicroad_y_val.csv')
dataframes['vicroad_y_test'] = pd.read_csv('Data/Cleaned Data/vicroad_y_test.csv')
dataframes['vicroad_df'] = pd.read_csv('Data/Cleaned Data/vicroad_df.csv')


# 1.3. Data Preparation ----------------------------------------------------------------------------------------
x_train = dataframes['vicroad_x_train'].drop(
    [
        'id', 'node', 'persons_inj_a', 'persons_inj_b', 'severity', 'vehicle_dca_code', 'inj_level', 'taken_hospital',
        'lga_name_all', 'road_name', 'road_name_intersection', 'vehicle_model', 'postcode_crash', 'road_route'
    ],
    axis = 1)

x_val = dataframes['vicroad_x_val'].drop(
    [
        'id', 'node', 'persons_inj_a', 'persons_inj_b', 'severity', 'vehicle_dca_code', 'inj_level', 'taken_hospital',
        'lga_name_all', 'road_name', 'road_name_intersection', 'vehicle_model', 'postcode_crash', 'road_route'
    ],
    axis = 1)

x_test = dataframes['vicroad_x_test'].drop(
    [
        'id', 'node', 'persons_inj_a', 'persons_inj_b', 'severity', 'vehicle_dca_code', 'inj_level', 'taken_hospital',
        'lga_name_all', 'road_name', 'road_name_intersection', 'vehicle_model', 'postcode_crash', 'road_route'
    ],
    axis = 1)

# Creating Target Variable
y_train = dataframes['vicroad_y_train']
y_train['fatal'] = np.where(y_train['persons_killed'] > 0, 1, 0)
y_train = y_train.drop('persons_killed', axis=1)

y_val = dataframes['vicroad_y_val']
y_val['fatal'] = np.where(y_val['persons_killed'] > 0, 1, 0)
y_val = y_val.drop('persons_killed', axis=1)

y_test = dataframes['vicroad_y_test']
y_test['fatal'] = np.where(y_test['persons_killed'] > 0, 1, 0)
y_test = y_test.drop('persons_killed', axis=1)

# Data Types
x_train['accident_date'] = pd.to_datetime(x_train['accident_date'])
x_val['accident_date'] = pd.to_datetime(x_val['accident_date'])
x_test['accident_date'] = pd.to_datetime(x_test['accident_date'])

# 1.4. Data Encoding -------------------------------------------------------------------------------------------
x_train = pd.get_dummies(
    x_train,
    columns=['accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'rma', 'initial_direction',
             'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make',
             'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type',
             'initial_impact', 'level_of_damage', 'traffic_control', 'sex', 'age_group', 'helmet_belt_worn',
             'road_user', 'license_state', 'node_type', 'lga_name', 'deg_urban_name',
             'road_type', 'road_type_intersection', 'direction_location'],
    prefix=['accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'rma', 'initial_direction',
             'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make',
             'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type',
             'initial_impact', 'level_of_damage', 'traffic_control', 'sex', 'age_group', 'helmet_belt_worn',
             'road_user', 'license_state', 'node_type', 'lga_name', 'deg_urban_name',
             'road_type', 'road_type_intersection', 'direction_location'],
    drop_first=True
)

x_val = pd.get_dummies(
    x_val,
    columns=['accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'rma', 'initial_direction',
             'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make',
             'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type',
             'initial_impact', 'level_of_damage', 'traffic_control', 'sex', 'age_group', 'helmet_belt_worn',
             'road_user', 'license_state', 'node_type', 'lga_name', 'deg_urban_name',
             'road_type', 'road_type_intersection', 'direction_location'],
    prefix=['accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'rma', 'initial_direction',
             'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make',
             'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type',
             'initial_impact', 'level_of_damage', 'traffic_control', 'sex', 'age_group', 'helmet_belt_worn',
             'road_user', 'license_state', 'node_type', 'lga_name', 'deg_urban_name',
             'road_type', 'road_type_intersection', 'direction_location'],
    drop_first=True
)

x_test = pd.get_dummies(
    x_test,
    columns=['accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'rma', 'initial_direction',
             'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make',
             'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type',
             'initial_impact', 'level_of_damage', 'traffic_control', 'sex', 'age_group', 'helmet_belt_worn',
             'road_user', 'license_state', 'node_type', 'lga_name', 'deg_urban_name',
             'road_type', 'road_type_intersection', 'direction_location'],
    prefix=['accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'rma', 'initial_direction',
             'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make',
             'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type',
             'initial_impact', 'level_of_damage', 'traffic_control', 'sex', 'age_group', 'helmet_belt_worn',
             'road_user', 'license_state', 'node_type', 'lga_name', 'deg_urban_name',
             'road_type', 'road_type_intersection', 'direction_location'],
    drop_first=True
)

x_train.columns = x_train.columns.str.replace(' ', '_').str.lower()
x_val.columns = x_val.columns.str.replace(' ', '_').str.lower()
x_test.columns = x_test.columns.str.replace(' ', '_').str.lower()

# Ensure Datasets have the same columns
common_columns = x_train.columns.union(x_val.columns).union(x_test.columns)

x_train = x_train.reindex(columns=common_columns, fill_value=0)
x_val = x_val.reindex(columns=common_columns, fill_value=0)
x_test = x_test.reindex(columns=common_columns, fill_value=0)

del common_columns


# ----------------------------------------------------------------------------------------------------------------------
# 2. Data Transformation
# ----------------------------------------------------------------------------------------------------------------------

# 2.1. Feature Engineering -------------------------------------------------------------------------------------
def feature_engineering(df):
    # Accident Date
    df['year'] = df['accident_date'].dt.year
    df['month'] = df['accident_date'].dt.month
    df['time (days)'] = (df['accident_date'] - df['accident_date'].min()).dt.days

    # Holidays
    years = df['accident_date'].dt.year.unique()
    aus_holidays = pd.to_datetime(list(holidays.Australia(years=years).keys()))
    vic_holidays = pd.to_datetime(list(holidays.Australia(years=years, prov='VIC').keys()))
    holiday = set(aus_holidays).union(set(vic_holidays))
    df['is_holiday'] = df['accident_date'].isin(holiday).astype(int)

    df = df.drop(columns=['accident_date'], axis=1)

    # Accident Time
    df['accident_time'] = pd.to_datetime(df['accident_time'], format='%H:%M:%S').dt.time
    df['accident_time'] = df['accident_time'].apply(
        lambda t: t.hour + t.minute / 60.0 + t.second / 3600.0 if pd.notnull(t) else np.nan
    )

    # Vehicle
    df['vehicle_age'] = df['year'] - df['vehicle_year']
    df['engine_efficiency'] = df['cylinders'] / df['vehicle_age'].replace(0, 1)

    # Geospatial
    melbourne_cbd = (-37.8136, 144.9631)

    def calculate_distance(row):
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            return geodesic((row['latitude'], row['longitude']), melbourne_cbd).kilometers
        else:
            return np.nan

    df['distance_to_cbd_km'] = df.apply(calculate_distance, axis=1)

    # Cluster Label using K-Means
    geospatial_data = df[['latitude', 'longitude']].dropna()
    kmeans = KMeans(n_clusters=5, random_state=SEED)
    df.loc[geospatial_data.index, 'geospatial_cluster'] = kmeans.fit_predict(geospatial_data)

    return df

x_train = feature_engineering(x_train)
x_val = feature_engineering(x_val)
x_test = feature_engineering(x_test)

# cor = x_train.corr()


# 2.2. Data Standardisation ------------------------------------------------------------------------------------
scaler = StandardScaler()

# Separate date and time columns
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)

# Standardize validation and test data using the same scaler
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns, index=x_val.index)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

# 2.3. Dimension Reduction -------------------------------------------------------------------------------------
category = [
    'accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'severity', 'rma', 'vehicle_dca_code',
    'initial_direction', 'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make', 'vehicle_model',
    'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type', 'initial_impact',
    'level_of_damage', 'traffic_control', 'sex', 'age_group', 'inj_level', 'helmet_belt_worn', 'road_user',
    'license_state', 'node_type', 'lga_name', 'lga_name_all', 'deg_urban_name', 'road_route',
    'road_name', 'road_type', 'road_name_intersection', 'road_type_intersection', 'direction_location', 'atmosph_cond',
    'event_type', 'sub_dca_code', 'surface_cond', 'veh_1_coll', 'veh_2_coll'
]

pca_dict = {}  # To store PCA objects for reuse
pca_components = []
columns_to_drop = []

# Perform PCA on training data
for var in category:
    relevant_columns = [col for col in x_train.columns if col.startswith(f'{var}_')]

    if relevant_columns:
        subset = x_train[relevant_columns]
        columns_to_drop.extend(relevant_columns)

        # Fit PCA
        pca = PCA(n_components=0.75)
        pca_result = pca.fit_transform(subset)

        # Save PCA object for validation and test data
        pca_dict[var] = pca

        # Create DataFrame for PCA results
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'{var}_pca_{i + 1}' for i in range(pca_result.shape[1])],
            index=x_train.index
        )
        pca_components.append(pca_df)

pca_components = pd.concat(pca_components, axis=1)
x_train = pd.concat([x_train, pca_components], axis=1)
x_train = x_train.drop(columns=columns_to_drop)

# Perform PCA on validation data
pca_components = []
columns_to_drop = []

for var, pca in pca_dict.items():
    relevant_columns = [col for col in x_val.columns if col.startswith(f'{var}_')]

    if relevant_columns:
        subset = x_val[relevant_columns]
        columns_to_drop.extend(relevant_columns)

        # Transform using the same PCA fitted on training data
        pca_result = pca.transform(subset)

        # Create DataFrame for PCA results
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'{var}_pca_{i + 1}' for i in range(pca_result.shape[1])],
            index=x_val.index
        )
        pca_components.append(pca_df)

pca_components = pd.concat(pca_components, axis=1)
x_val = pd.concat([x_val, pca_components], axis=1)
x_val = x_val.drop(columns=columns_to_drop)

# Perform PCA on test data
pca_components = []
columns_to_drop = []

for var, pca in pca_dict.items():
    relevant_columns = [col for col in x_test.columns if col.startswith(f'{var}_')]

    if relevant_columns:
        subset = x_test[relevant_columns]
        columns_to_drop.extend(relevant_columns)

        # Transform using the same PCA fitted on training data
        pca_result = pca.transform(subset)

        # Create DataFrame for PCA results
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'{var}_pca_{i + 1}' for i in range(pca_result.shape[1])],
            index=x_test.index
        )
        pca_components.append(pca_df)

pca_components = pd.concat(pca_components, axis=1)
x_test = pd.concat([x_test, pca_components], axis=1)
x_test = x_test.drop(columns=columns_to_drop)

# Clean up variables
del [var, pca_df, pca_result, pca_components, columns_to_drop, subset, relevant_columns]


# 2.4. Feature Importance --------------------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test, threshold=0.5, model_name="Model"):
    """
    Evaluates a trained classification model and prints key metrics.

    Parameters:
    - model: Trained classifier (e.g., XGBoost, Neural Network)
    - x_test: Test feature set
    - y_test: True labels for test set
    - threshold: Decision threshold for binary classification
    - model_name: Name of the model (for plotting)

    Returns:
    - Dictionary of evaluation metrics
    """

    # Get predicted probabilities (only for class 1)
    y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_test)

    # Apply decision threshold
    y_pred = (y_pred_proba > threshold).astype(int)

    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Print Metrics
    print(f"\n📊 {model_name} Evaluation Results:")
    print(f"Accuracy: {accuracy:.8f}")
    print(f"Precision: {precision:.8f}")
    print(f"Recall: {recall:.8f}")
    print(f"F1 Score: {f1:.8f}")
    print(f"AUC-ROC: {auc:.8f}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fatal", "Fatal"],
                yticklabels=["Non-Fatal", "Fatal"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, label=f'{model_name}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend()
    plt.show()

    # Best Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]

    print("Best threshold for F1:", best_threshold)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

# Initialize the model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=SEED,
    n_jobs=-1,
    verbosity=2,
    eval_metric='auc',
    colsample_bytree=0.5,
    gamma=0.0,
    learning_rate=0.07292699809069496,
    max_depth=15,
    n_estimators=280,
    subsample=1.0,
    #scale_pos_weight=64,
)

search_spaces = {
    'n_estimators': Integer(50, 500),  # Number of boosting rounds
    'max_depth': Integer(3, 40),  # Depth of each tree
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),  # Step size shrinkage
    'subsample': Real(0.5, 1.0),  # Fraction of samples for training
    'colsample_bytree': Real(0.5, 1.0),  # Fraction of features per tree
    'gamma': Real(0, 5),  # Minimum loss reduction
}

bayes_search = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=search_spaces,
    n_iter=50,
    scoring='roc_auc',
    cv=8,
    random_state=SEED,
    n_jobs=-1,
    verbose=2
)

# bayes_search.fit(x_train, y_train)
# print("Best Parameters found by BayesSearchCV:")
# print(bayes_search.best_params_)

# Retrieve best model
# xgb_model = bayes_search.best_estimator_

# Retrain the best model
xgb_model.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_val, y_val)],
    verbose=False
)

pred = xgb_model.predict(x_val)

# Retrieve training history
evals_result = xgb_model.evals_result()
train_auc = evals_result['validation_0']['auc'][-1]
val_auc= evals_result['validation_1']['auc'][-1]

print("Training AUC:", train_auc)
print("Validation AUC:", val_auc)

xgb_results_init = evaluate_model(xgb_model, x_val, y_val, threshold=0.04207764, model_name="Initial XGBoost")

# Get feature importance
importance_xgb = xgb_model.feature_importances_

# Create a DataFrame for visualization
feat_importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': importance_xgb
}).sort_values(by='importance', ascending=False)

# Plot for XGBoost
plt.figure(figsize=(6, 20))
sns.barplot(x='importance', y='feature', data=feat_importance.head(100))
plt.title('Top 100 Feature Importance (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 2.5. Imbalanced Data -----------------------------------------------------------------------------------------
sm = SMOTE(
    random_state=SEED,
    sampling_strategy='auto',
    k_neighbors=3
)

adasyn = ADASYN(
    sampling_strategy='auto',
    random_state=SEED,
    n_neighbors=2000
)

border_sm= BorderlineSMOTE(
    random_state=SEED,
    sampling_strategy='auto',
    k_neighbors=2000,
    m_neighbors=4000,
    kind="borderline-2"
)

sm_tomek = SMOTETomek(random_state=SEED)

#x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)
x_train_sm, y_train_sm = adasyn.fit_resample(x_train, y_train)
#x_train_sm, y_train_sm = border_sm.fit_resample(x_train, y_train)
#x_train_sm, y_train_sm = sm_tomek.fit_resample(x_train, y_train)

# Retrain the best model
xgb_model.fit(
    x_train_sm, y_train_sm,
    eval_set=[(x_train_sm, y_train_sm), (x_val, y_val)],
    verbose=False
)

pred = xgb_model.predict(x_val)

# Retrieve training history
evals_result = xgb_model.evals_result()
train_auc = evals_result['validation_0']['auc'][-1]
val_auc= evals_result['validation_1']['auc'][-1]

print("Training AUC:", train_auc)
print("Validation AUC:", val_auc)

xgb_results_smote = evaluate_model(xgb_model, x_val, y_val, threshold=0.06953415, model_name="SMOTE XGBoost")

# ----------------------------------------------------------------------------------------------------------------------
# 3. Machine Learning
# ----------------------------------------------------------------------------------------------------------------------
# 3.1. XGBoost (L1 Regularisation) -----------------------------------------------------------------------------
# Initialize the model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=SEED,
    n_jobs=-1,
    verbosity=2,
    eval_metric='auc',
    colsample_bytree=0.5,
    gamma=0.0,
    learning_rate=0.07292699809069496,
    max_depth=15,
    n_estimators=280,
    subsample=1.0,
    alpha=5.325826766633403e-05
)

# Defining Search Space
search_spaces = {
    'alpha': Real(0, 1.0)
}

bayes_search = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=search_spaces,
    n_iter=20,
    scoring='roc_auc',
    cv=3,
    random_state=SEED,
    n_jobs=-1,
    verbose=2
)

# bayes_search.fit(x_train_sm, y_train_sm)
#
# print("Best Parameters found by BayesSearchCV:")
# print(bayes_search.best_params_)
#
# # Retrieve best model
# xgb_model = bayes_search.best_estimator_

# Retrain the best model
xgb_model.fit(
    x_train_sm, y_train_sm,
    eval_set=[(x_train_sm, y_train_sm), (x_val, y_val)],
    verbose=False
)

# Evaluation Metrics
evals_result = xgb_model.evals_result()
train_auc = evals_result['validation_0']['auc'][-1]
val_auc= evals_result['validation_1']['auc'][-1]

print("Training AUC:", train_auc)
print("Validation AUC:", val_auc)
xgb_results = evaluate_model(xgb_model, x_val, y_val, threshold=0.09121445, model_name="XGBoost")

# Feature Importance
feature_importance = xgb_model.feature_importances_
fi_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importance})
fi_df = fi_df.sort_values(by='Importance', ascending=False)


# Refitting with Selected Features
top_features = fi_df.head(525)['Feature'].tolist()
x_train_sel = x_train_sm[top_features]
x_val_sel = x_val[top_features]
x_test_sel = x_test[top_features]

xgb_model.fit(
    x_train_sel, y_train_sm,
    eval_set=[(x_train_sel, y_train_sm), (x_val_sel, y_val)],
    verbose=False
)

# Evaluation Metrics
evals_result = xgb_model.evals_result()
train_auc = evals_result['validation_0']['auc'][-1]
val_auc= evals_result['validation_1']['auc'][-1]

print("Training AUC:", train_auc)
print("Validation AUC:", val_auc)
xgb_results_fi = evaluate_model(xgb_model, x_val_sel, y_val, threshold=0.059913896, model_name="Feature Selection XGBoost")

# Feature Importance
feature_importance = xgb_model.feature_importances_
fi_df_2 = pd.DataFrame({'Feature': x_train_sel.columns, 'Importance': feature_importance})
fi_df_2 = fi_df_2.sort_values(by='Importance', ascending=False)


# 3.2. XGBoost (Recursive Feature Elimination) -----------------------------------------------------------------
# # Initialize RFE with desired number of features
# selector = RFE(
#     estimator=xgb_model,
#     n_features_to_select=600,
#     step=10,
#     verbose=2
# )
#
# # Fit RFE
# selector = selector.fit(x_train_sm, y_train_sm)
#
# selected_features = x_train_sm.columns[selector.support_].tolist()
# print("Selected Features:", selected_features)
#
# x_train_rfe = x_train_sm[selected_features]
# x_val_rfe = x_val[selected_features]
# x_test_rfe = x_test[selected_features]
#
# xgb_model.fit(
#     x_train_rfe, y_train_sm,
#     eval_set=[(x_train_rfe, y_train_sm), (x_val_rfe, y_val)],
#     verbose=False
# )
#
# # Evaluation Metrics
# evals_result = xgb_model.evals_result()
# train_auc = evals_result['validation_0']['auc'][-1]
# val_auc= evals_result['validation_1']['auc'][-1]
#
# print("Training AUC:", train_auc)
# print("Validation AUC:", val_auc)


# 3.3. Neural Network ------------------------------------------------------------------------------------------
neural_net = Sequential([
    Input(shape=(x_train_sm.shape[1],)),
    Dense(64, activation='tanh'),
    Dropout(0.4),
    Dense(16, activation='tanh'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

neural_net.compile(
#    optimizer=Adam(learning_rate=0.00001),
    optimizer=SGD(learning_rate=0.00005, momentum=0.9, nesterov=True),
    loss='binary_crossentropy',
    metrics=['AUC']
)

history = neural_net.fit(
    x_train_sm, y_train_sm,
    epochs=100,
    batch_size=10,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on Test Set
val_loss, val_auc = neural_net.evaluate(x_val, y_val, verbose=2)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation AUC: {val_auc:.4f}")

nn_results_ = evaluate_model(neural_net, x_val, y_val, threshold=0.9337299, model_name="Neural Network")


# 3.4. Model Evaluation ----------------------------------------------------------------------------------------
xgb_model.fit(
    pd.concat([x_train_sm, x_val]), pd.concat([y_train_sm, y_val]),
    eval_set=[(pd.concat([x_train_sm, x_val]), pd.concat([y_train_sm, y_val])), (x_test, y_test)],
    verbose=2
)

evals_result = xgb_model.evals_result()
train_auc = evals_result['validation_0']['auc'][-1]
test_auc = evals_result['validation_1']['auc'][-1]

print(f"XGBoost Training AUC: {train_auc:.4f}")
print(f"XGBoost Test AUC: {test_auc:.4f}")
xgb_results_final = evaluate_model(xgb_model, x_test, y_test, threshold=0.056607462, model_name="XGBoost")

# Neural Network
neural_net = Sequential([
    Input(shape=(x_train_sm.shape[1],)),
    Dense(64, activation='tanh'),
    Dropout(0.4),
    Dense(16, activation='tanh'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

neural_net.compile(
    optimizer=SGD(learning_rate=0.00005, momentum=0.9, nesterov=True),
    loss='binary_crossentropy',
    metrics=['AUC']
)

history = neural_net.fit(
    pd.concat([x_train_sm, x_val]), pd.concat([y_train_sm, y_val]),
    epochs=38,
    batch_size=10,
    verbose=2
)

test_loss, test_auc = neural_net.evaluate(x_test, y_test, verbose=2)

print(f"Neural Network Test Loss: {test_loss:.4f}")
print(f"Neural Network Test AUC: {test_auc:.4f}")
nn_results_ = evaluate_model(neural_net, x_test, y_test, threshold=0.93481755, model_name="Neural Network")



# -------------------------------------------------------------------------------------------------------------------- #
#                                           VicRoads Motor Fatalities Model                                            #
#                                                  Machine Learning                                                    #
# -------------------------------------------------------------------------------------------------------------------- #