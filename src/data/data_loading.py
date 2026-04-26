import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) #Loading raw data.
    return df

"""
Performs the ingestion of the raw dataset into the pipeline.
Ensures initial traceability in Weights & Biases.
"""