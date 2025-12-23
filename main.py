import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve, roc_auc_score

data = pd.read_csv('nba_final_clean.csv')

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
data.columns = data.columns.str.lower()


def feetToMeter(dt):
    return dt * 0.3048

def heightToCM(x):
    if not isinstance(x, str):
        return None
    try:
        feet, inches = x.split('-')
        return float(feet) * 30.48 + float(inches) * 2.54
    except:
        return None

print(data.columns)
print(data.shape)
print(data.isnull().sum()/len(data))

# convert feet and inch to cm
data["shooter_height_cm"] = data["shooter_height"].apply(heightToCM)
data["defender_height_cm"] = data["defender_height"].apply(heightToCM)

# ---------------------------------------------------------
# HANDLING MISSING VALUES: WINGSPAN IMPUTATION
# ---------------------------------------------------------
# we have 127k records missing value for shot clock 0.04 so we can drop them
data.dropna(subset=["shot_clock"], inplace=True)

# Fill missing height values with the average height
data["shooter_height_cm"].fillna(data["shooter_height_cm"].mean(), inplace=True)
data["defender_height_cm"].fillna(data["defender_height_cm"].mean(), inplace=True)

# convert wingspan data inch to cm
data["defender_wingspan_cm"] =  data["defender_wingspan"].apply(lambda x : x*2.54)
data["shooter_wingspan_cm"] =  data["shooter_wingspan"].apply(lambda x : x*2.54)

# 1. Calculate the average ratio of Wingspan to Height from existing data
# (NBA players typically have a wingspan greater than their height)
avg_wingspan_ratio = data['defender_wingspan_cm'].mean() / data['defender_height_cm'].mean()

# 2. Fill missing wingspan values using the calculated ratio
# Logic: Estimated Wingspan = Player Height * Average Ratio
data["shooter_wingspan_cm"].fillna(data["shooter_height_cm"]*avg_wingspan_ratio, inplace=True)
data["defender_wingspan_cm"].fillna(data["defender_height_cm"]*avg_wingspan_ratio, inplace=True)


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

# 1. Distance Conversion: Feet to Meter
data["shot_dist_m"] = data["shot_dist"].apply(feetToMeter)

# 2. Target Variable: Convert 'made' to 1, 'missed' to 0 for classification
data["result"] = data["shot_result"].apply(lambda x : 1 if x == "made" else 0)

# 3. Game Pressure: Check if the score difference is 10 points or less (Critical Moment)
data["is_critical"] = data["final_margin"].apply(lambda x : 1 if abs(x) <= 10 else 0)

# 4. Shot Clock Pressure (Slope)
# We created a custom function. If time < 5 seconds, the pressure increases exponentially.
# This represents the "panic" factor in the last seconds.
data["shot_clock_isSlope"] = data["shot_clock"].apply(lambda x : 0 if x > 5.0 else (1/25) * (x-5) * (x-5))

# 5. Home Advantage: 1 if playing at Home, 0 if Away
data["is_home"] = data["location"].apply(lambda x : 1 if x == "H" else 0)

# 6. Physical Mismatch: Ratio of Shooter's wingspan to Defender's wingspan
# If ratio > 1, the shooter has a physical advantage.
data["ratio_wingspan_def_of"] = data.apply(lambda row: row["shooter_wingspan_cm"]/row["defender_wingspan_cm"] , axis=1)

# 7. Shot Angle Calculation
# Calculating the angle of the shot relative to the basket using X, Y coordinates
data["shot_angle"] = np.arctan2(data["loc_x"], data["loc_y"]) * (180 / np.pi)
data["shot_angle"] = data["shot_angle"].fillna(0) # Fill NaN angles with 0 (Center)

def get_defensive_pressure(dist):
    if dist < 2: return "Very Tight"
    elif dist < 4: return "Tight"
    elif dist < 6: return "Open"
    else: return "Wide Open"

data["pressure_level"] = data["close_def_dist"].apply(get_defensive_pressure)
# 9. Catch and Shoot vs. Isolation
# Logic: If dribbles = 0, it's a "Catch and Shoot" (usually higher percentage).
# If dribbles > 0, the player is creating their own shot (Isolation).
data["is_catch_and_shoot"] = data["dribbles"].apply(lambda x: 1 if x == 0 else 0)
# Alternative: High Dribble Count (Player creates distinct fatigue/difficulty)
data["is_high_dribble"] = data["dribbles"].apply(lambda x: 1 if x > 3 else 0)

# 10. Shot Difficulty Index (Composite Feature)
# Logic: We combine Shot Distance and Defender Proximity.
# Higher Shot Distance (+) increases difficulty.
# Higher Defender Distance (-) decreases difficulty.
# Formula: (Shot Distance) - (Defender Distance * 1.5)
# Note: We give more weight (1.5) to the defender because a hand in the face is very impactful.
data["shot_difficulty"] = data["shot_dist"] - (data["close_def_dist"] * 1.5)


# Checking the ratio of missing values after operations
print("Missing Value Ratio:\n", data.isnull().sum()/len(data))

# ---------------------------------------------------------
# ENCODING CATEGORICAL FEATURES
# ---------------------------------------------------------
# Machine Learning models work with numbers, not text.
# We fill missing zone data with "Unknown" to treat it as a separate category.
data['shot_zone_basic'] = data['shot_zone_basic'].fillna("Unknown")
data['shot_zone_area'] = data['shot_zone_area'].fillna("Unknown")

# Using LabelEncoder to convert text categories (e.g., "Left Corner") into numbers (e.g., 1, 2)
le = LabelEncoder()
data['shot_zone_basic_code'] = le.fit_transform(data['shot_zone_basic'])
data['shot_zone_area_code'] = le.fit_transform(data['shot_zone_area'])


# ---------------------------------------------------------
# DROPPING UNNECESSARY COLUMNS
# ---------------------------------------------------------
# We drop columns for three reasons:
# 1. IDs (game_id, player_id): They contain no generalized pattern for prediction.
# 2. Raw Features: We drop 'shooter_height', 'shot_clock' etc. because we created better versions (e.g., 'shooter_height_cm', 'shot_clock_isSlope').
# 3. Data Leakage: We MUST drop 'shot_result', 'fgm', 'pts' because they hold the answer we want to predict.

drop_cols = ["shot_zone_basic", "shot_zone_area", "game_id", "matchup", "location",
             "w", "final_margin", "shot_number", "game_clock", "shot_clock",
             "closest_defender_player_id", "fgm", "pts", "player_name", "player_id",
             "shooter_height", "shooter_weight", "shooter_wingspan", "defender_height",
             "defender_weight", "defender_wingspan", "shot_number_api", "shot_type",
             "api_points", "api_points", "api_calc_dist", "closest_defender", "shot_result"]

data.drop(columns=drop_cols, inplace=True)

# Final check of the clean dataframe
print("Final Data Shape:", data.shape)
print(data.head())





# ---------------------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------------------
X = data.drop(columns=['result'])
y = data['result']

# We use stratify=y to maintain the same ratio of made/missed shots in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------------------------------------------
# TARGET ENCODING (FOR HIGH CARDINALITY FEATURE)
# ---------------------------------------------------------
# Problem: 'action_type' has many categories (Jump Shot, Dunk, Layup, etc.).
# One-Hot Encoding would create too many columns (High Dimensionality).
# Solution: We replace the category name with its "Success Rate" (Mean of Target).

# 1. Calculate success rates ONLY on Training Data (To prevent Data Leakage)
# If we used the whole dataset, the model would cheat by knowing the future.
action_rates = y_train.groupby(X_train['action_type']).mean()

# 2. Map these rates to Train and Test sets
X_train['action_type_encoded'] = X_train['action_type'].map(action_rates)
X_test['action_type_encoded'] = X_test['action_type'].map(action_rates)

# 3. Handle Unknown Categories
# If a specific move appears in the Test set but was never seen in Train set,
# we fill it with the global average success rate.
global_mean = y_train.mean()
X_train['action_type_encoded'] = X_train['action_type_encoded'].fillna(global_mean)
X_test['action_type_encoded'] = X_test['action_type_encoded'].fillna(global_mean)

# 4. Drop the original string column
X_train = X_train.drop(columns=['action_type'])
X_test = X_test.drop(columns=['action_type'])

print("Encoding Completed. New feature 'action_type_encoded' created based on shot success probability.")

# ---------------------------------------------------------
# 1. HYPERPARAMETER TUNING (Randomized Search) RANDOM FOREST
# ---------------------------------------------------------
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# class_weight='balanced' is critical because missed shots might be more frequent than made shots.
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=param_grid,
                               n_iter=20,
                               cv=3,
                               verbose=1,  # 2 çok kalabalık yapar, 1 yeterli
                               random_state=42,
                               n_jobs=-1)

print("Searching for best parameters...")
rf_random.fit(X_train, y_train)

best_rf_model = rf_random.best_estimator_

print(f"Best Score (CV): {rf_random.best_score_:.4f}")
print(f"Best Params: {rf_random.best_params_}")

# ---------------------------------------------------------
# 2. FEATURE IMPORTANCE
# ---------------------------------------------------------
# Which factors are most important for a successful shot?
importances = pd.Series(best_rf_model.feature_importances_, index=X_train.columns)

plt.figure(figsize=(10, 6))
importances.nlargest(10).plot(kind='barh', color='#1f77b4')
plt.title("Top 10 Features Influencing Shot Success")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.show()

# ---------------------------------------------------------
# 3. THRESHOLD OPTIMIZATION
# ---------------------------------------------------------
# Default threshold is 0.50. But in sports analytics, we might want to be safer.
y_proba = best_rf_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.30, 0.75, 0.05)
best_f1_thresh = 0.5
max_f1_score = 0

print(f"\n{'Threshold':<10} {'Accuracy':<10} {'Recall (Made)':<15} {'F1-Score':<10}")
print("-" * 50)

for thresh in thresholds:
    preds = (y_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    recall_1 = classification_report(y_test, preds, output_dict=True)['1']['recall']

    print(f"{thresh:.2f}       {acc:.4f}     {recall_1:.4f}             {f1:.4f}")

    if f1 > max_f1_score:
        max_f1_score = f1
        best_f1_thresh = thresh

print("-" * 50)
print(f"Algorithm Suggestion (Best F1): {best_f1_thresh:.2f}")

# ---------------------------------------------------------
# 4. FINAL DECISION
# ---------------------------------------------------------

SELECTED_THRESHOLD = best_f1_thresh

print(f"\n>>> FINAL CHOSEN THRESHOLD: {SELECTED_THRESHOLD} <<<")

# Final Predictions based on Selected Threshold
final_preds = (y_proba >= SELECTED_THRESHOLD).astype(int)

# ---------------------------------------------------------
# 5. FINAL REPORT & VISUALIZATION
# ---------------------------------------------------------
print("\n--- FINAL CLASSIFICATION REPORT ---")
print(classification_report(y_test, final_preds))

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Missed (0)', 'Made (1)'],
            yticklabels=['Missed (0)', 'Made (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold: {SELECTED_THRESHOLD})')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')


# Find the point on ROC curve closest to our selected threshold
diffs = np.abs(thresholds - SELECTED_THRESHOLD)

fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_proba)
idx = (np.abs(thresholds_roc - SELECTED_THRESHOLD)).argmin()
plt.scatter(fpr_roc[idx], tpr_roc[idx], c='green', s=100, zorder=10, label=f'Chosen Threshold ({SELECTED_THRESHOLD})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("--- STARTING XGBOOST HYPERPARAMETER TUNING ---")

# ---------------------------------------------------------
# 1. HYPERPARAMETER GRID
# ---------------------------------------------------------
# XGBoost has more parameters than Random Forest.
# scale_pos_weight: Helps if "Made Shots" are fewer than "Missed Shots" (Imbalanced Data).
# learning_rate: Lower is usually better but requires more n_estimators.
param_grid_xgb = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'scale_pos_weight': [1.2, 1.5],
    'gamma': [0, 0.1, 0.2]
}

xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid_xgb,
    n_iter=20,
    scoring='f1',  # We optimize for F1 Score to balance Precision/Recall
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(X_train, y_train)

print(f"\nBest CV Score: {xgb_random.best_score_:.4f}")
print("Best Parameters:", xgb_random.best_params_)

best_xgb = xgb_random.best_estimator_

# ---------------------------------------------------------
# 2. FEATURE IMPORTANCE (XGBOOST STYLE)
# ---------------------------------------------------------
# XGBoost calculates importance differently (Gain).
importances = pd.Series(best_xgb.feature_importances_, index=X_train.columns)

plt.figure(figsize=(10, 6))
importances.nlargest(10).plot(kind='barh', color='#800080')  # Purple for XGBoost
plt.title("Top 10 Features (XGBoost)")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.show()

# ---------------------------------------------------------
# 3. THRESHOLD TUNING
# ---------------------------------------------------------
# XGBoost probabilities might be different distribution than RF.
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.30, 0.75, 0.05)
best_f1_thresh = 0.5
max_f1_score = 0

print(f"\n{'Threshold':<10} {'Accuracy':<10} {'Recall (Made)':<15} {'F1-Score':<10}")
print("-" * 55)

for thresh in thresholds:
    preds = (y_proba_xgb >= thresh).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Safe Recall Calculation
    report = classification_report(y_test, preds, output_dict=True)
    recall_1 = report['1']['recall'] if '1' in report else 0

    print(f"{thresh:.2f}       {acc:.4f}     {recall_1:.4f}             {f1:.4f}")

    if f1 > max_f1_score:
        max_f1_score = f1
        best_f1_thresh = thresh

print("-" * 55)
print(f"Algorithm Suggestion: {best_f1_thresh:.2f}")

# ---------------------------------------------------------
# 4. FINAL SELECTION & REPORT
# ---------------------------------------------------------
SELECTED_THRESH_XGB = best_f1_thresh


print(f"\n>>> FINAL CHOSEN THRESHOLD (XGBoost): {SELECTED_THRESH_XGB} <<<")

final_preds_xgb = (y_proba_xgb >= SELECTED_THRESH_XGB).astype(int)

print("\n--- FINAL XGBOOST REPORT ---")
print(classification_report(y_test, final_preds_xgb))

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds_xgb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Missed', 'Made'],
            yticklabels=['Missed', 'Made'])
plt.title(f'XGBoost Confusion Matrix (Thresh: {SELECTED_THRESH_XGB})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_xgb)
auc_score = roc_auc_score(y_test, y_proba_xgb)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='purple', lw=3, label=f'XGBoost ROC (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Mark the selected threshold
idx = (np.abs(thresholds_roc - SELECTED_THRESH_XGB)).argmin()
plt.scatter(fpr[idx], tpr[idx], c='red', s=100, zorder=10, label=f'Chosen ({SELECTED_THRESH_XGB})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()





from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
import matplotlib.pyplot as plt

print("--- PREPARING KNN MODEL ---")

# 1. FEATURE SCALING (MANDATORY FOR KNN)
# KNN uses Euclidean distance. If we don't scale, features like "Shot Distance" (0-90)
# will dominate features like "Shot Clock" (0-24).
scaler = StandardScaler()

# Fit on training set, transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data Scaled. Model training started...")

# 2. MODEL INITIALIZATION
# n_neighbors (k): Selected 21 to smooth out the noise in the large dataset.
# weights='distance': Closer neighbors have more influence on the vote.
knn = KNeighborsClassifier(
    n_neighbors=21,
    n_jobs=-1,       # Use all CPU cores
    weights='distance'
)

# 3. TRAINING (LAZY LEARNING)
# KNN doesn't learn a model, it memorizes the training data.
start_time = time.time()
knn.fit(X_train_scaled, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds.")

# 4. PREDICTION
# This step is computationally expensive as it calculates distance to all training points.
print("Predicting (This might take a while)...")
y_pred_knn = knn.predict(X_test_scaled)

# 5. RESULTS
print(f"\nKNN Accuracy: %{accuracy_score(y_test, y_pred_knn)*100:.2f}")
print("\n--- DETAILED CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred_knn))

# ---------------------------------------------------------
# OPTIMIZATION: FINDING THE BEST 'K' VALUE (ELBOW METHOD)
# ---------------------------------------------------------
error_rate = []
k_values = range(1, 30, 2)  # Checking odd numbers from 1 to 29

print("Optimizing 'k' value... (This loop takes time)")

# We use a smaller subset of data for this loop to save time (Optional but recommended)
# X_train_sub, _, y_train_sub, _ = train_test_split(X_train_scaled, y_train, train_size=0.1, random_state=42)

for i in k_values:
    # Use X_train_sub if you want speed, otherwise X_train_scaled
    knn_i = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn_i.fit(X_train_scaled, y_train)
    pred_i = knn_i.predict(X_test_scaled)

    # Calculate error (1 - Accuracy)
    error_rate.append(np.mean(pred_i != y_test))
    print(f"k={i} processed.")

# PLOTTING THE ERROR RATE
plt.figure(figsize=(10, 6))
plt.plot(k_values, error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value (Elbow Method)')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()