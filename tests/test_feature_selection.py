import pandas as pd
import pytest

from src.data.feature_selection import calculate_feature_importance, select_features


def test_calculate_feature_importance():
    train_df = pd.DataFrame({
        "R": [1, 2, 3, 4],
        "A_M": [4, 3, 2, 1],
        "Type": [0, 0, 1, 1]
    })

    feat_importance = calculate_feature_importance(train_df, target_col="Type")

    assert list(feat_importance.columns) == ["feature", "importance"]
    assert set(feat_importance["feature"]) == {"R", "A_M"}
    assert feat_importance["importance"].sum() > 0


def test_select_features_keeps_selected_columns_and_target():
    train_df = pd.DataFrame({
        "R": [1, 2],
        "A_M": [3, 4],
        "unused": [5, 6],
        "Type": [0, 1]
    })
    test_df = pd.DataFrame({
        "R": [7, 8],
        "A_M": [9, 10],
        "unused": [11, 12],
        "Type": [1, 0]
    })

    train_selected, test_selected = select_features(
        train_df,
        test_df,
        target_col="Type",
        selected_features=["R", "A_M"]
    )

    assert list(train_selected.columns) == ["R", "A_M", "Type"]
    assert list(test_selected.columns) == ["R", "A_M", "Type"]


def test_select_features_raises_for_missing_columns():
    train_df = pd.DataFrame({
        "R": [1, 2],
        "Type": [0, 1]
    })
    test_df = pd.DataFrame({
        "Type": [1, 0]
    })

    with pytest.raises(ValueError, match="Selected columns are missing"):
        select_features(
            train_df,
            test_df,
            target_col="Type",
            selected_features=["R"]
        )
