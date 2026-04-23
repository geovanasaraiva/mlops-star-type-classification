import pandas as pd
from sklearn.model_selection import train_test_split

#It performs the division of the dataset into training and test sets.
def split_train_test( #It separates features and target, and applies a stratified split.
    df: pd.DataFrame, #Complete dataframe
    target_col: str, #Target variable name
    test_size: float = 0.2, #Test set proportion
    random_state: int = 42 #Seed of reproducibility
):
    
    #Separates independent variables (X) and target variable (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    #Divide the dataset into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    #Recombine features and target
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, test_df