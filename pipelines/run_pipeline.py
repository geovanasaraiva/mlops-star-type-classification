import os
import sys
import yaml
import wandb
import pandas as pd
import torch

#Ensures project root is in the Python path for module resolution.
sys.path.append(os.path.abspath("."))

#Import project modules for reproducible pipeline execution.
from src.utils.utils import set_seed
from src.data.data_cleaning import clean_pipeline
from src.data.split_data import split_train_test
from src.data.feature_selection import select_features
from src.model.data_loader import prepare_dataloaders
from src.model.train import train_model

#Load configuration file to control pipeline parameters.
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

#Set random seed for reproducibility across all steps.
set_seed(config["data"]["random_state"])

#Define project name and target variable.
project = config["wandb"]["project"]
target = config["data"]["target_col"]

#Define path to raw dataset.
raw_path = "dataset/raw/Stars.csv"

#1. Clean.
#========================
#Initialize W&B run for data cleaning stage.
run1 = wandb.init(project=project, job_type="clean")

#Load raw data and apply cleaning pipeline.
df = pd.read_csv(raw_path)
df = clean_pipeline(df)

#Save cleaned dataset locally.
clean_path = "clean.csv"
df.to_csv(clean_path, index=False)

#Create artifact for cleaned data to enable lineage tracking.
clean_art = wandb.Artifact("clean_data", type="dataset")
clean_art.add_file(clean_path)

#Log artifact to W&B and finalize run.
run1.log_artifact(clean_art)
run1.finish()

#2. Split.
#Initialize W&B run for data splitting stage.
run2 = wandb.init(project=project, job_type="split")

#Load latest cleaned dataset artifact.
clean_art = run2.use_artifact("clean_data:latest")
clean_dir = clean_art.download()

#Read cleaned dataset.
clean_df = pd.read_csv(os.path.join(clean_dir, "clean.csv"))

#Split dataset into train and test sets with stratification.
train_df, test_df = split_train_test(
    clean_df,
    target_col=target,
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"]
)

#Save split datasets locally.
train_path = "train.csv"
test_path = "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

#Create artifacts for train and test datasets.
train_art = wandb.Artifact("train_data", type="dataset")
train_art.add_file(train_path)

test_art = wandb.Artifact("test_data", type="dataset")
test_art.add_file(test_path)

#Log artifacts and finalize run.
run2.log_artifact(train_art)
run2.log_artifact(test_art)
run2.finish()

#3. Feature selection.
run_fs = wandb.init(project=project, job_type="feature_selection")

#Check if feature selection is enabled.
use_fs = config.get("feature_selection", {}).get("use", True)

#Load datasets.
train_art = run_fs.use_artifact("train_data:latest")
test_art = run_fs.use_artifact("test_data:latest")

train_dir = train_art.download()
test_dir = test_art.download()

train_df = pd.read_csv(os.path.join(train_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(test_dir, "test.csv"))

if use_fs:
    print("Feature selection ENABLED.")

    train_sel, test_sel, selected_features, feat_importance = select_features(
        train_df,
        test_df,
        config
    )

else:
    print("Feature selection DISABLED. Using all features.")

    train_sel = train_df.copy()
    test_sel = test_df.copy()
    selected_features = list(train_df.drop(columns=[target]).columns)
    feat_importance = pd.DataFrame()

#Save outputs.
train_sel_path = "train_selected.csv"
test_sel_path = "test_selected.csv"
feat_imp_path = "feature_importance.csv"

train_sel.to_csv(train_sel_path, index=False)
test_sel.to_csv(test_sel_path, index=False)

if not feat_importance.empty:
    feat_importance.to_csv(feat_imp_path, index=False)

#Create artifacts.
train_sel_art = wandb.Artifact("train_selected", type="dataset")
train_sel_art.add_file(train_sel_path)

test_sel_art = wandb.Artifact("test_selected", type="dataset")
test_sel_art.add_file(test_sel_path)

#Metadata.
train_sel_art.metadata = {
    "selected_features": selected_features,
    "feature_selection_used": use_fs,
    "top_k": config.get("feature_selection", {}).get("top_k", None)
}

run_fs.log_artifact(train_sel_art)
run_fs.log_artifact(test_sel_art)

#Log feature importance (only if FS was used).
if use_fs and not feat_importance.empty:
    feat_imp_art = wandb.Artifact("feature_importance", type="analysis")
    feat_imp_art.add_file(feat_imp_path)
    run_fs.log_artifact(feat_imp_art)

    #Log table to W&B.
    run_fs.log({
        "feature_importance_table": wandb.Table(dataframe=feat_importance)
    })

run_fs.finish()

#4. Train.
#Initialize W&B run for model training stage.
run3 = wandb.init(project=project, job_type="train")

#Load selected train and test datasets.
train_art = run3.use_artifact("train_selected:latest")
test_art = run3.use_artifact("test_selected:latest")

train_dir = train_art.download()
test_dir = test_art.download()

#Read datasets for model training.
train_df = pd.read_csv(os.path.join(train_dir, "train_selected.csv"))
test_df = pd.read_csv(os.path.join(test_dir, "test_selected.csv"))

#Prepare PyTorch dataloaders with normalization.
train_loader, test_loader, scaler = prepare_dataloaders(
    train_df,
    test_df,
    target_col=target,
    batch_size=config["model"]["batch_size"]
)

#Define input dimension based on feature count.
input_dim = train_df.shape[1] - 1

#Train model using configured hyperparameters.
model = train_model(
    config,
    train_loader,
    test_loader,
    input_dim=input_dim
)

#Save trained model weights.
model_path = "model.pth"
torch.save(model.state_dict(), model_path)

#Create artifact for trained model.
model_art = wandb.Artifact("trained_model", type="model")
model_art.add_file(model_path)

#Log model artifact and finalize run.
run3.log_artifact(model_art)
run3.finish()

#Final message indicating successful pipeline execution.
print("Complete pipeline.")