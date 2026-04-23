import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.model import MLP

#Train an MLP model using PyTorch.
def train_model(config, train_loader, test_loader, input_dim):

    #Device definition (CPU or GPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Model initialization.
    model = MLP(
        input_size=input_dim,
        hidden_sizes=config["model"]["hidden_sizes"],
        dropout=config["model"]["dropout"],
        output_size=6
    ).to(device)

    #Loss function (classification).
    criterion = nn.CrossEntropyLoss()

    #Optmizer.
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"]
    )

    #Monitoring with WANDB.
    wandb.watch(model, log="all", log_freq=10)

    #Early stopping.
    best_val_loss = float("inf")
    patience = config["model"]["early_stopping_patience"]
    patience_counter = 0

    #Training loop.
    for epoch in range(config["model"]["epochs"]):

        #Training mode.
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            #Move data to CPU/GPU.
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #Zero out accumulated gradients.
            optimizer.zero_grad()

            #Forward pass.
            output = model(X_batch)
            #Loss calculation.
            loss = criterion(output, y_batch)

            #Backpropagation.
            loss.backward()
            optimizer.step()

            #Accumulate loss weighted by batch size.
            train_loss += loss.item() * X_batch.size(0)

        #Average training loss.
        train_loss /= len(train_loader.dataset)

        #Avaliation mode.
        model.eval()
        val_loss = 0.0
        correct = 0

        #Disable gradient calculation.
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(X_batch)
                loss = criterion(output, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                #Class prediction.
                pred = torch.argmax(output, dim=1)
                correct += (pred == y_batch).sum().item()

        #Average validation loss.
        val_loss /= len(test_loader.dataset)

        #Accuracy calculation.
        val_acc = correct / len(test_loader.dataset)

        #Wandb log.
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        #Early stopping and saving the best model.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            #Saves weight of the best model.
            torch.save(model.state_dict(), "best_model.pt")

            #Log model as an wandb artifact.
            model_artifact = wandb.Artifact(
                "trained_model",
                type="model"
            )
            model_artifact.add_file("best_model.pt")
            wandb.log_artifact(model_artifact)

        else:
            #Stop training if there is no improvement.
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    #Load the best model before return.
    model.load_state_dict(torch.load("best_model.pt"))
    return model

"""
Supports:
- GPU (if available)
- Early stopping
- Monitoring with Weights & Biases (wandb)
- Saving the best model

Parameters:
- config: dictionary with model hyperparameters
- train_loader: Training DataLoader
- test_loader: Validation/test DataLoader
- input_dim: number of input features

Returns:
- model trained with the best weights (lowest val_loss)
"""