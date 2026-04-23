# ⭐ Star Type Classification MLopss Project
This project develops a complete machine learning cycle, from cleaning the dataset for preprocessing to training an MLP (multilayer perceptron) neural network with artifact versioning and reprodutibility.
# Repository structure
- config: config.yaml file with hyperparemeters.
- data: Directories for raw and processed data.
- src: Python scripts containing logic for cleaning, splitting, model definition, and training.
- tests: Python scripts containing logic for tests and validation.
- notebooks: Jupyter notebook containing code for study and visualization.
- pipeline: Requirements and run file.
# Installation and requirements.
- Creation of the virtual environment.
- Importing libraries.
- Wheigts and Biases autentication.
# The MLops pipeline.
- Ingestion: Loading of raw data and artifact registration.
- Cleaning: Removal of duplicates, handling of missing values ​​and outliers (IQR), followed by artifact registration.
- Split: Stratified split between training and testing with statistical validation (KS and Chi² tests).
- Training: Execution of the training loop with Early Stopping, monitoring for validation loss.
# Monitoring and Results.
All results can be acessed on W&B.
- Logged metrics: Accuracy, loss, confusion matrix, and system metrics.
- Linhagem: O gráfico de linhagem (Lineage Graph) no W&B permite rastrear qual versão do dado gerou qual modelo.
