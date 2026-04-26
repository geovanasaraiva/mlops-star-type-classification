import os
import sys
import yaml
import wandb
import pandas as pd

#ENSURE IMPORTS.
sys.path.append(os.path.abspath("."))

#IMPORTS.
from src.utils.utils import set_seed

from src.data.data_loading import load_data
from src.data.data_cleaning import (
    remove_duplicates,
    handle_missing_values,
    encode_features
)
from src.data.split_data import split_train_test
from src.data.validate_data import compare_distributions
from src.data.feature_selection import (
    calculate_feature_importance,
    select_features
)

from src.model.data_loader import prepare_dataloaders
from src.model.train import train_model

#LOAD CONFIG YAML.
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

#SEED.
set_seed(config["data"]["random_state"])

#LOAD DATA.
df_raw = load_data(config["data"]["raw_path"])
print(f"Raw data: {df_raw.shape}")

#CLEANING.
df_clean = remove_duplicates(df_raw)

df_clean = handle_missing_values(
    df_clean,
    strategy=config["data"]["imputation_strategy"],
    threshold=config["data"]["missing_threshold"]
)

df_clean = encode_features(df_clean)

print(f"Clean data: {df_clean.shape}")

#SPLIT.
train_df, test_df = split_train_test(
    df_clean,
    target_col=config["data"]["target_col"],
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"]
)

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

#FEATURE SELECTION.
feat_importance = calculate_feature_importance(
    train_df,
    target_col=config["data"]["target_col"],
    random_state=config["data"]["random_state"]
)

print("\nFeature importance:")
print(feat_importance)

train_df, test_df = select_features(
    train_df,
    test_df,
    target_col=config["data"]["target_col"]
)

print(f"Selected train shape: {train_df.shape}")
print(f"Selected test shape: {test_df.shape}")

#DISTRIBUTION CHECK.
feature_cols = [
    c for c in train_df.columns
    if c != config["data"]["target_col"]
]

comp_results = compare_distributions(
    train_df,
    test_df,
    feature_cols
)

comp_df = pd.DataFrame(comp_results).T.reset_index()
comp_df.columns = ["feature", "test", "statistic", "p_value"]

print("\nDistribution comparison:")
print(comp_df.head())

#W&B.
wandb.init(
    project="mlops-star-type",
    job_type="pipeline",
    config=config
)

wandb.log({
    "distribution_comparison": wandb.Table(dataframe=comp_df)
})

wandb.summary["train_size"] = len(train_df)
wandb.summary["test_size"] = len(test_df)

#DATALOADERS.
train_loader, test_loader, scaler = prepare_dataloaders(
    train_df,
    test_df,
    target_col=config["data"]["target_col"],
    batch_size=config["model"]["batch_size"]
)

input_dim = train_loader.dataset.tensors[0].shape[1]
print(f"Input dimension: {input_dim}")

#TRAIN.
model = train_model(
    config,
    train_loader,
    test_loader,
    input_dim
)

#END.
wandb.finish()

print("Pipeline executado com sucesso 🚀")