import pandas as pd
from src.data.split_data import split_train_test

#This tests whether the split preserves the total number of samples.
def test_split_shapes():
    #Ensures that no lines are lost or duplicated during the split between training and testing.
    df = pd.DataFrame({
        "f1": range(10),
        "Type": [0, 1] * 5
    })
    train_df, test_df = split_train_test(df, target_col="Type")

    #The sum of the sizes must equal the original dataset.
    assert len(train_df) + len(test_df) == len(df)

#Tests whether stratification is working correctly.
def test_stratification():
    #Checks if the proportion of the target class is similar between training and testing.
    df = pd.DataFrame({
        "f1": range(100),
        "Type": [0]*50 + [1]*50
    })
    train_df, test_df = split_train_test(df, target_col="Type")

    #Calculates class proportion.
    train_ratio = train_df["Type"].mean()
    test_ratio = test_df["Type"].mean()

    #Check if the difference between proportions is small.
    assert abs(train_ratio - test_ratio) < 0.1

#Tests whether the target column remains in the final dataset.
def test_target_not_in_features():
    #Ensures that the split does not remove the target variable when recombining X and y.
    df = pd.DataFrame({
        "f1": range(10),
        "Type": [0, 1] * 5
    })
    train_df, test_df = split_train_test(df, target_col="Type")

    #The target must exist in both datasets.
    assert "Type" in train_df.columns
    assert "Type" in test_df.columns