import os
import sys
import pandas as pd
import yaml
import wandb
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Ensure project root is in the Python path.
sys.path.append(os.path.abspath("."))
from src.data.data_loading import load_data
from src.data.data_cleaning import (
    remove_duplicates,
    handle_missing_values,
    encode_features
)

#Load configuration file.
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

#Initialize Weights & Biases.
wandb.init(
    project=config["wandb"]["project"],
    job_type="feature_selection",
    config=config
)

#Data loading and preprocessing.
df = load_data(config["data"]["raw_path"])

#Remove duplicates row
df = remove_duplicates(df)

#Handle missing values.
df = handle_missing_values(
    df,
    strategy=config["data"]["imputation_strategy"],
    threshold=config["data"]["missing_threshold"]
)

#Encode categorical features
df = encode_features(df)

#Split features and target.
target = config["data"]["target_col"]
X = df.drop(columns=[target])
y = df[target]

#Feature importance method.

#1. Correlation.
corr = X.corrwith(y).abs()

#2. Mutual information
mi = mutual_info_classif(X, y, random_state=config["data"]["random_state"])
mi = pd.Series(mi, index=X.columns)

#3.Random forest feature importance.
rf = RandomForestClassifier(random_state=config["data"]["random_state"])
rf.fit(X, y)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

#Normalization helper function.

#Normalize a pandas Series to range [0, 1].
def normalize(s):
    #Avoids division by 0 when all values are equal.
    return (s - s.min()) / (s.max() - s.min()) if (s.max() - s.min()) != 0 else s

#Combine feature importance scores.
feature_scores = pd.DataFrame({
    "correlation": normalize(corr),
    "mutual_info": normalize(mi),
    "rf_importance": normalize(rf_importance)
})

#Final score: average of all methods.
feature_scores["final_score"] = feature_scores.mean(axis=1)
#Sort features by importance.
feature_scores = feature_scores.sort_values("final_score", ascending=False).reset_index()

#Multicollinearity analysis (VIF).

#Calculate Variance Inflation Factor (VIF) for each feature.
def calculate_vif(X_df):
   
    #Ensure all features are numeric.
    X_numeric = X_df.astype(float) 
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_numeric.columns
    
    #Compute VIF for each feature.
    vif_data["VIF"] = [
        variance_inflation_factor(X_numeric.values, i)
        for i in range(X_numeric.shape[1])
    ]
    #VIF>10 may indicate strong multicolinearity.
    return vif_data

#Calculate VIF.
vif_df = calculate_vif(X)

#Log results to wandb.
wandb.log({
    "feature_ranking": wandb.Table(dataframe=feature_scores),
    "vif_analysis": wandb.Table(dataframe=vif_df)
})

#Console output.
print("\n=== FEATURE RANKING ===")
print(feature_scores.head(10))

print("\n=== VIF ANALYSIS (Consider removing features with VIF > 10) ===")
print(vif_df.sort_values("VIF", ascending=False))

#Finish wandb run.
wandb.finish()