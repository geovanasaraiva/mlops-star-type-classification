import pandas as pd
from src.data.validate_data import (
    test_no_missing_values,
    test_target_values,
    test_range_checks
)

#Tests the function for checking for missing values.
def test_no_missing():
    #Create a DataFrame without null values ​​and validate that the function does not raise an error.
    df = pd.DataFrame({
        "a": [1, 2, 3]
    })
    #Must pass without errors
    test_no_missing_values(df)

#Tests the validation of the target variable.
def test_target():
    #Checks if all values ​​are within the allowed set
    df = pd.DataFrame({
        "Type": [0, 1, 2]
    })
    #All values ​​are valid
    test_target_values(df, "Type", [0,1,2])

#Tests the range validity of the variables.
def test_range():
    #Checks if the values ​​are within the expected limits defined in the validation function.
    df = pd.DataFrame({
        "Type": [0, 1, 2],
        "Temperature": [100, 200, 300]
    })
    test_range_checks(df)