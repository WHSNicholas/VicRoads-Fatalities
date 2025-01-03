from sklearn.preprocessing import LabelEncoder

df = vicroad_x_train
column = 'amg_y'
features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
            'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
            'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
            'road_type', 'speed_zone', 'vehicle_year', 'registration_state', 'cylinders', 'seating_capacity', 'amg_x', 'amg_y'] + [
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
test_size=0.2
SEED=1
task='regression'


# Data Processing
impute = df[df[column].isna()][features]
train = df[~df[column].isna()][features]

x_train = train.drop(columns=[column])
y_train = train[column]

x_impute = impute.drop(columns=[column])

# Splitting Data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=SEED+1)

# Define the XGBoost Model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=SEED,
    n_jobs=-1,
    verbosity=2,
    enable_categorical=True
)


search_spaces = {
    'n_estimators': Integer(50, 300),  # Number of boosting rounds
    'max_depth': Integer(3, 20),  # Depth of each tree
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),  # Step size shrinkage
    'subsample': Real(0.5, 1.0),  # Fraction of samples for training
    'colsample_bytree': Real(0.5, 1.0),  # Fraction of features per tree
    'gamma': Real(0, 5),  # Minimum loss reduction
}

bayes_search = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=search_spaces,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=SEED,
    n_jobs=-1,
    verbose=2
)

# Hyperparameter Tuning
bayes_search.fit(x_train, y_train)

print("Best Parameters found by BayesSearchCV:")
print(bayes_search.best_params_)

# Retrieve best model
xgb_model = bayes_search.best_estimator_

# Retrain the best model
xgb_model.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_val, y_val)],
    verbose=False
)

# Retrieve training history
evals_result = xgb_model.evals_result()
train_rmse = evals_result['validation_0']['rmse'][-1]
val_rmse = evals_result['validation_1']['rmse'][-1]

print("Training RMSE:", train_rmse)
print("Validation RMSE:", val_rmse)









le = LabelEncoder()

df = vicroad_x_train
column = 'road_type_intersection'
features = ['accident_type', 'day', 'dca_code', 'light_condition', 'vehicles', 'persons_inj_a', 'persons_inj_b',
            'persons', 'police_attend', 'road_geometry', 'severity', 'level_of_damage', 'traffic_control', 'sex',
            'age_group', 'inj_level', 'helmet_belt_worn', 'road_user', 'license_state', 'taken_hospital', 'ejected',
            'speed_zone', 'vehicle_year', 'registration_state', 'cylinders', 'seating_capacity', 'node_type', 'deg_urban_name', 'road_type', 'road_type_intersection'] + [
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

test_size=0.2
SEED=1
task='classification'


# Data Processing
impute = df[df[column].isna()][features]
train = df[~df[column].isna()][features]

x_train = train.drop(columns=[column])
y_train = train[column]

x_impute = impute.drop(columns=[column])

# Splitting Data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=SEED)

y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# Define the XGBoost Model
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    random_state=SEED,
    n_jobs=-1,
    verbosity=2,
    enable_categorical=True
)

# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     random_state=SEED,
#     n_jobs=-1,
#     verbosity=2,
#     enable_categorical=True
# )


search_spaces = {
    'n_estimators': Integer(50, 300),  # Number of boosting rounds
    'max_depth': Integer(3, 20),  # Depth of each tree
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),  # Step size shrinkage
    'subsample': Real(0.5, 1.0),  # Fraction of samples for training
    'colsample_bytree': Real(0.5, 1.0),  # Fraction of features per tree
    'gamma': Real(0, 5),  # Minimum loss reduction
}

bayes_search = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=search_spaces,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=SEED,
    n_jobs=-1,
    verbose=2
)

bayes_search.fit(x_train, y_train_encoded)

print("Best Parameters found by BayesSearchCV:")
print(bayes_search.best_params_)

# Retrieve best model
xgb_model = bayes_search.best_estimator_

# Retrain the best model
xgb_model.fit(
    x_train, y_train_encoded,
    eval_set=[(x_train, y_train_encoded), (x_val, y_val_encoded)],
    verbose=False
)

pred = xgb_model.predict(x_val)
y_val
# Retrieve training history
evals_result = xgb_model.evals_result()
train_logloss = evals_result['validation_0']['mlogloss'][-1]
val_logloss = evals_result['validation_1']['mlogloss'][-1]

print("Training LogLoss:", train_logloss)
print("Validation LogLoss:", val_logloss)


# Inspect the distribution of the target variable
print("Class Distribution in the Entire Dataset:")
print(vicroad_x_train.registration_state.value_counts())

x = vicroad_x_train.road_type.value_counts()




