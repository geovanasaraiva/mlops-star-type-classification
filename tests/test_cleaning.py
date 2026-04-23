import pandas as pd
from src.data.data_cleaning import remove_duplicates, handle_missing_values

#Tests the duplicate removal function.
def test_remove_duplicates():
    #Creates a DataFrame with duplicate rows and verifies that they are correctly removed.
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": [3, 3, 4]
    })

    df_clean = remove_duplicates(df)

    #It is expected that only 2 unique lines will remain.
    assert len(df_clean) == 2

#Test the handling of missing values.
def test_handle_missing_values():
    #Verify that the function removes or pads null values, ensuring that the final DataFrame does not contain NaNs.
    df = pd.DataFrame({
        "num": [1, None, 3],
        "cat": ["a", None, "b"]
    })

    df_clean = handle_missing_values(df)

    #After treatment, there should be no zero values.
    assert df_clean.isnull().sum().sum() == 0

#Tests for removing columns with many missing values.
def test_drop_columns_by_threshold():
    #If the proportion of null values ​​exceeds the threshold, the column should be removed.
    df = pd.DataFrame({
        "a": [1, None, None], #66% missing.
        "b": [1, 2, 3]        #0% missing.
    })

    df_clean = handle_missing_values(df, threshold=0.5)

    #Column 'a' should be removed as it exceeds the threshold.
    assert "a" not in df_clean.columns