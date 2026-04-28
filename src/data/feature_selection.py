import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr


def select_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict):

    target = config["data"]["target_col"]
    top_k = config["feature_selection"].get("top_k", 7)

    print("Starting feature selection using Pearson + MI + RF")

    #Split.
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    #Encoding.
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(test_df.drop(columns=[target]))
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    #1. Pearson Correlation.
    corr_scores = []

    for col in X_train.columns:
        try:
            corr, _ = pearsonr(X_train[col], y_train)
            corr_scores.append(abs(corr))
        except:
            corr_scores.append(0)

    corr_df = pd.DataFrame({
        "feature": X_train.columns,
        "corr_score": corr_scores
    })

    #2. Mutual Information.
    mi = mutual_info_classif(X_train, y_train, random_state=42)

    mi_df = pd.DataFrame({
        "feature": X_train.columns,
        "mi_score": mi
    })

    #3. Random Forest.
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    rf_df = pd.DataFrame({
        "feature": X_train.columns,
        "rf_score": rf.feature_importances_
    })

    #Merge scores.
    feat_df = corr_df.merge(mi_df, on="feature").merge(rf_df, on="feature")

    #Normalize.
    for col in ["corr_score", "mi_score", "rf_score"]:
        feat_df[col] = (feat_df[col] - feat_df[col].min()) / (
            feat_df[col].max() - feat_df[col].min() + 1e-9
        )

    #Final score.
    feat_df["final_score"] = (
        0.3 * feat_df["corr_score"] +
        0.3 * feat_df["mi_score"] +
        0.4 * feat_df["rf_score"]
    )

    feat_df = feat_df.sort_values(by="final_score", ascending=False)

    print("\nFeature ranking:")
    print(feat_df)

    #Select top features.
    selected_features = feat_df["feature"].head(top_k).tolist()

    print(f"\nSelected features: {selected_features}")

    #Apply selection.
    train_selected = train_df[selected_features + [target]].copy()
    test_selected = test_df[selected_features + [target]].copy()

    return train_selected, test_selected, selected_features, feat_df