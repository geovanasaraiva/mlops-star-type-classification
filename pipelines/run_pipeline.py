import os
import sys
import yaml
import wandb
import pandas as pd

sys.path.append(os.path.abspath("."))

from src.utils.utils import set_seed
from src.data.data_cleaning import clean_pipeline
from src.data.split_data import split_train_test
from src.data.validate_data import compare_distributions
from src.model.data_loader import prepare_dataloaders
from src.model.train import train_model
from src.data.feature_selection import select_features

#Load configuration.
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

set_seed(config["data"]["random_state"])

#Initialize Weights and Biases run.
run = wandb.init(
    project=config["wandb"]["project"],
    job_type="pipeline",
    config=config,
    name="mlp_training"
)

#Load raw dataset.
df = pd.read_csv("dataset/raw/Stars.csv")

target = config["data"]["target_col"]

#Apply data cleaning pipeline.
df = clean_pipeline(df)

#Split dataset into training and test sets.
train_df, test_df = split_train_test(
    df,
    target_col=target,
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"]
)

#Save train and test datasets as W&B artifacts.
for name, data in [("train_data", train_df), ("test_data", test_df)]:
    artifact = wandb.Artifact(name, type="dataset")
    path = f"temp_{name}.csv"
    data.to_csv(path, index=False)
    artifact.add_file(path)
    run.log_artifact(artifact)

#Perform feature selection using Random Forest.
train_df, test_df, selected_features, feat_importance = select_features(
    train_df,
    test_df,
    config
)

run.log({
    "feature_importance": wandb.Table(dataframe=feat_importance)
})

#Validate distribution between train and test sets.
features = [f for f in selected_features if f != target]

comp_results = compare_distributions(train_df, test_df, features)

comp_df = pd.DataFrame(comp_results).T.reset_index()
comp_df.columns = ["feature", "test", "statistic", "p_value"]

run.log({
    "distribution_comparison": wandb.Table(dataframe=comp_df)
})

#Prepare data loaders for model training.
train_loader, test_loader, scaler = prepare_dataloaders(
    train_df,
    test_df,
    target_col=target,
    batch_size=config["model"]["batch_size"]
)

input_dim = len(selected_features)

#Train final model.
model = train_model(
    config,
    train_loader,
    test_loader,
    input_dim
)

run.finish()

print("Pipeline executed successfully.")