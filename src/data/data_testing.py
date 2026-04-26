import pandas as pd

#Checks if the dataset contains any missing values.
def test_no_missing_values(df: pd.DataFrame) -> None:
    #Total sum of missing values ​​in the dataset.
    total_missing = df.isnull().sum().sum() 
    #If any value is missing, the test fails.
    assert total_missing == 0, \
        f"TEST FAILED: {total_missing} missing values remain."
    print("PASS: no missing values.")

#Checks if the target variable contains only valid values.
def test_target_values(
    df: pd.DataFrame,
    target_col: str,
    allowed_values
) -> None:

    #Identifies values ​​outside the allowed set.
    invalid_mask = ~df[target_col].isin(allowed_values)
    invalid_count = invalid_mask.sum()

    #Failure if invalid values ​​exist.
    assert invalid_count == 0, \
        f"TEST FAILED: {invalid_count} unexpected values in '{target_col}': {df.loc[invalid_mask, target_col].unique()}"

    print(f"PASS: '{target_col}' contains only valid values.")

#Validates whether variables follow expected ranges based on the problem domain.
def test_range_checks(df: pd.DataFrame) -> None:
    #Type must be between 0 and 5.
    assert df['Type'].between(0, 5).all(), \
        "Error: 'Type' out of range [0, 5]"
    #Temperature should be positive.
    assert (df['Temperature'] > 0).all(), \
        "Error: Negative or null temperature detected"

    print("PASS:values ​​within expected ranges.")

#Ensures that all expected columns are present in the dataset.
def test_column_names(df: pd.DataFrame, expected_columns: list) -> None:
    current_columns = df.columns.tolist()
    # Checks for the presence of each expected column.
    for col in expected_columns:
        assert col in current_columns, \
            f"TEST FAILED: Column '{col}' not found!"

    print("PASS: All expected columns are present.")

#Checks if numeric columns have the correct type.
def test_data_types(df: pd.DataFrame) -> None:
    numeric_cols = ['Temperature', 'L', 'R', 'A_M']
    #Checks if each column is numeric.
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), \
            f"TEST FAILED: {col} it's not numeric!"

    print("PASS: Validated data types.")