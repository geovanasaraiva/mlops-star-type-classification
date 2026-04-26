from sklearn.ensemble import RandomForestClassifier
import pandas as pd


SELECTED_FEATURES = [
    "R",
    "A_M",
    "L",
    "Temperature",
    "Spectral_Class_M",
    "Color_red",
    "Spectral_Class_O",
]


def calculate_feature_importance(
    train_df: pd.DataFrame,
    target_col: str,
    random_state: int = 42
) -> pd.DataFrame:
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)

    feat_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return feat_importance


def select_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    selected_features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = selected_features or SELECTED_FEATURES
    selected_columns = features + [target_col]

    missing_train_cols = [col for col in selected_columns if col not in train_df.columns]
    missing_test_cols = [col for col in selected_columns if col not in test_df.columns]

    if missing_train_cols or missing_test_cols:
        raise ValueError(
            "Selected columns are missing. "
            f"Train: {missing_train_cols}. Test: {missing_test_cols}."
        )

    return train_df[selected_columns], test_df[selected_columns]
