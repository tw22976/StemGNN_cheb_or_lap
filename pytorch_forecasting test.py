import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytorch_lightning
from pytorch_forecasting.models.baseline import Baseline
from pytorch_forecasting.models.n_beats import NBeats

from pytorch_forecasting import TimeSeriesDataSet, DataLoader



# Load the dataset
data = pd.read_csv('random7.csv', index_col=0)

# Define normalization function
def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic

# Split the data into train, validation, and test sets
train_size = int(0.7 * len(data))
val_size = int(0.2 * len(data))
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Normalize the data
normalize_method = 'min_max'
train_data, norm_statistic = normalized(train_data, normalize_method)
val_data, _ = normalized(val_data, normalize_method, norm_statistic)
test_data, _ = normalized(test_data, normalize_method, norm_statistic)

# Create the TimeSeriesDataSet
max_prediction_length = 10
max_encoder_length = 50
training_cutoff = train_data.index[-max_prediction_length]
context_length = max_encoder_length

training = TimeSeriesDataSet(
    train_data,
    time_idx='time',
    target='target',
    group_ids=['stock'],
    min_encoder_length=context_length,
    max_encoder_length=context_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=['target'],
)

# Initialize the NBeats model
trainer = pl.Trainer(gpus=1, max_epochs=100)
model = NBeats.from_dataset(training, learning_rate=1e-3, width=512, backcast_loss_ratio=0.1)

# Train the model
train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
trainer.fit(model, train_dataloader)

# Define function to calculate metrics
def calculate_metrics(true, pred, normalize_method):
    if normalize_method == 'min_max':
        true = true * (norm_statistic['max'] - norm_statistic['min']) + norm_statistic['min']
        pred = pred * (norm_statistic['max'] - norm_statistic['min']) + norm_statistic['min']
    elif normalize_method == 'z_score':
        true = true * norm_statistic['std'] + norm_statistic['mean']
        pred = pred * norm_statistic['std'] + norm_statistic['mean']
    mape = np.mean(np.abs((true - pred) / true)) * 100
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return mape, mae, rmse

# Calculate metrics for different data splits
def calculate_metrics_for_split(data, model, normalize_method):
    dataloader = training.to_dataloader(data)
    all_true = []
    all_pred = []
    for batch in dataloader:
        x, y = model.transform_batch(batch)
        with torch.no_grad():
            prediction = model.backcast(x)[0]
        all_true.append(y.cpu().numpy())
        all_pred.append(prediction.cpu().numpy())
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    mape, mae, rmse = calculate_metrics(all_true, all_pred, normalize_method)
    return mape, mae, rmse

# Calculate metrics for different data sets
train_mape, train_mae, train_rmse = calculate_metrics_for_split(train_data, model, normalize_method)
val_mape, val_mae, val_rmse = calculate_metrics_for_split(val_data, model, normalize_method)
test_mape, test_mae, test_rmse = calculate_metrics_for_split(test_data, model, normalize_method)

# Calculate Train Total Loss
train_total_loss = trainer.callback_metrics['train_loss']

# Store the results in a DataFrame
results = pd.DataFrame({
    'Data Split': ['Train', 'Validation', 'Test'],
    'MAPE': [train_mape, val_mape, test_mape],
    'MAE': [train_mae, val_mae, test_mae],
    'RMSE': [train_rmse, val_rmse, test_rmse],
    'Train Total Loss': [train_total_loss, np.nan, np.nan]
})

# Save results to CSV file
results.to_csv('random7_NBeats_outcome.csv', index=False)
