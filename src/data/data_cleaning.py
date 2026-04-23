#Import librarys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


#REMOVE DUPLICATES
    #Eliminates duplicate records to ensure project reproducibility and stability.
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame: #Remove duplicate records to avoid MLP bias.
    before = len(df) #Store the initial size to ensure traceability.
    df = df.drop_duplicates() #Remove exact duplicates
    removed = before - len(df) #Calculates the total removed for dataset quality audit.
    print(f"Removed {removed} duplicate rows ({removed/before:.1%})")
    return df


# TREATING OF MISSING VALUES
    #Remove columns with excessive missing values ​​(>50%) 
    #Imputes numeric and categorical values ​​separately   
def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    threshold: float = 0.5
) -> pd.DataFrame:
    #Identify columns that exceed the missing values ​​limit.
    missing_fraction = df.isnull().mean() 
    cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()

    #Remove the columns
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns (>{threshold:.0%} missing): {cols_to_drop}")

    #Separate features by type
    numeric_cols = df.select_dtypes(include=[np.number]).columns #Separate features by type
    cat_cols = df.select_dtypes(include=["object"]).columns

    #Numerical Imputation
    imputer = SimpleImputer(strategy=strategy) 
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    #Categorical Treatment (placeholder to maintain line traceability)
    df[cat_cols] = df[cat_cols].fillna("missing") 

    return df



#OUTLIERS TREATMENTS
    #Treat outliers by IQR to avoid explosive gradients in the MLP.
    #The 'cap' (Winsorization) mode is used to maintain data volume.
def handle_outliers_iqr(df: pd.DataFrame, columns: list, mode="cap") -> pd.DataFrame:

    df = df.copy() #Creates a copy of the dataset 

    #Ensuring data security and pipeline stability.
    for col in columns: #Loop through each defined column
        if col not in df.columns: #Checks if the requested column actually exists in the current DataFrame.
            continue

        #Applies only to numerical data.
        if not np.issubdtype(df[col].dtype, np.number): 
            continue

        #Define the statistical distribution (25% and 75%)
        Q1 = df[col].quantile(0.25) 
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        #Define Tukey's boundaries (1.5 * IQR)
        lower = Q1 - 1.5 * IQR 
        upper = Q3 + 1.5 * IQR

        #Remove extreme samples
        if mode == "remove": 
            df = df[(df[col] >= lower) & (df[col] <= upper)]

        #Winsorization. Limits extremes to stabilize convergence.
        elif mode == "cap": 
            df[col] = np.where(df[col] < lower, lower, df[col])
            df[col] = np.where(df[col] > upper, upper, df[col])

    print(f"IQR applied ({mode}) on columns: {columns}")
    return df


#ENCODING
    #Ensures that all inputs are numeric for performing calculations.
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() #Creates a copy of the dataset.

    #String normalization
    df['Color'] = df['Color'].str.lower().str.strip() 

    #Consolidates similar or synonymous categories.
    df['Color'] = df['Color'].replace({ 
        'yellow-white': 'white-yellow',
        'yellowish white': 'white-yellow',
        'blue white': 'blue-white',
        'whitish': 'white'
    })

    #Convert categorical variables into binary representations (One-Hot Encoding).
    df = pd.get_dummies( 
        df,
        columns=['Color', 'Spectral_Class'],
        drop_first=True
    )

    return df


#FINAL PIPELINE
def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:

    #Duplicate removal
    df = remove_duplicates(df)

    #Treatment of missings: Implements the discard threshold (>50%)
    df = handle_missing_values(
        df,
        strategy="median" #Median strategy for variables with skewed distribution.
    )

    #Outlier Treatment (IQR) specifically applied to column 'A_M'
    iqr_cols = ["A_M"]   
    df = handle_outliers_iqr(df, iqr_cols, mode="cap")

    #Feature Encoding. Converts categorical variables into numeric format for mathematical calculations.
    df = encode_features(df)

    return df