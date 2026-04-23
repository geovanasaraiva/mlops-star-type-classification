import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

#Prepare PyTorch DataLoaders from DataFrames.
def prepare_dataloaders(train_df, test_df, target_col: str, batch_size: int):

    #Separation of features and target.
    X_train = train_df.drop(columns=[target_col]).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.int64)

    X_test = test_df.drop(columns=[target_col]).values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.int64)

    #Normalization.
    #Adjust the scaler only on the training data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #Apply the same transformation to the test
    X_test = scaler.transform(X_test)

    #Conversion to PyTorch tensors.
    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )

    test_ds = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test)
    )

    #Creating the DataLoaders.
    #shuffle=True in training, improving generalization.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    #shuffle=False in the test for consistent evaluation.
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler

"""
Parameters:
- train_df: Training DataFrame
- test_df: Test DataFrame
- target_col: Target variable name
- batch_size: Training batch size

Returns:
- train_loader: Training DataLoader
- test_loader: Test DataLoader
- scaler: Object used for normalization (important for production)

"""