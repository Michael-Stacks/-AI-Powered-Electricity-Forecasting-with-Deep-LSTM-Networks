import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
import os

from torch import nn
from torch.utils.data import DataLoader
from utils.dataLoader import LoadingData
from NN_LSTM.LSTM import LSTM
from NN_LSTM.LSTM import LSTMTrainer, LSTMPredictor
from config import *


# Load datasets
Dataset = LoadingData()
train_dataloader, valid_dataloader, test_dataloader = Dataset.get_train_valid_test_dataloader(time_windows_length=TIME_WINDOWS_LENGTH,
                                                                                              overlap=OVERLAP,
                                                                                              prediction_length=PREDICTION_LENGTH,
                                                                                              train_valid_test_split=TRAIN_VALID_TEST_SPLIT,
                                                                                              num_workers=0,
                                                                                              batch_size=BATCH_SIZE,
                                                                                              shuffle=True)

input, output = next(iter(train_dataloader))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size=input.shape[2], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=output.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

trainer = LSTMTrainer(model=model,
                      train_loader=train_dataloader,
                      val_loader=valid_dataloader,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device)



start_time = timeit.default_timer()

trainer.train(num_epochs=EPOCHS)
trainer.plot_loss_curves()

end_time = timeit.default_timer()
print(f"Training took {end_time - start_time:.2f} seconds.")

# Load the saved model parameters

model.load_state_dict(torch.load('results/models/best_model.pt'))

# Make a prediction with the loaded model
predictor = LSTMPredictor(model=model, device=device)

input, output = next(iter(test_dataloader))
prediction = predictor.predict(input[0])

input, output = Dataset.get_scaler().inverse_transform(input[0]), Dataset.get_scaler().inverse_transform(output[0])
prediction = Dataset.get_scaler().inverse_transform(prediction)

os.makedirs('./results/prediction/', exist_ok=True)

plt.figure(figsize=(12, 8))
plt.plot(input[:, 0], label='input-production')
plt.plot(range(len(input[:, 0]), len(input[:, 0]) + len(output[:, 0])), output[:, 0], label='target-production')
plt.plot(range(len(input[:, 0]), len(input[:, 0]) + len(prediction[:, 0])), prediction[:, 0], label='prediction-production')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Production (kWh)')
plt.title('Production Prediction')
plt.tight_layout()
plt.savefig('./results/prediction/production.png', dpi=800)

plt.figure(figsize=(12, 8))
plt.plot(input[:, 1], label='input-consumption')
plt.plot(range(len(input[:, 1]), len(input[:, 1]) + len(output[:, 1])), output[:, 1], label='target-consumption')
plt.plot(range(len(input[:, 1]), len(input[:, 1]) + len(prediction[:, 1])), prediction[:, 1], label='prediction-consumption')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Consumption (kWh)')
plt.title('Consumption Prediction')
plt.tight_layout()
plt.savefig('./results/prediction/consumption.png', dpi=800)

print('Done!')
