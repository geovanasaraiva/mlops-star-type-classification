import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict):

    target = config["data"]["target_col"]

    print("Starting feature selection with Random Forest.")

    #Split features and target.
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    # Ensure all features are numeric (required by sklearn).
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(test_df.drop(columns=[target]))

    #Align train and test columns.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    #Train Random Forest model.
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    #Compute feature importance.
    feat_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nFeature importance:")
    print(feat_importance)
    print()

    #Select top features.
    selected_features = feat_importance["feature"].head(7).tolist()

    print(f"Selected features: {selected_features}")

    #Keep only selected features.
    train_selected = train_df[selected_features + [target]].copy()
    test_selected = test_df[selected_features + [target]].copy()

    return train_selected, test_selected, selected_features, feat_importance