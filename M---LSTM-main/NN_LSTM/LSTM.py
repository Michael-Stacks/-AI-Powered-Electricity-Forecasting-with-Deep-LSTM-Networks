import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from colorama import Fore

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

class LSTMPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, x):
        x = x.unsqueeze(0)  # Add a batch dimension
        x = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        return output.squeeze().cpu().numpy()

class LSTMTrainer:
        def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
            """
            Initializes the LSTMTrainer class.

            Args:
            - model: the LSTM model to be trained
            - train_loader: the data loader for the training set
            - val_loader: the data loader for the validation set
            - criterion: the loss function to be used
            - optimizer: the optimizer to be used for training
            - device: the device (CPU or GPU) to be used for training
            """
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.criterion = criterion
            self.optimizer = optimizer
            self.device = device
            self.best_val_loss = float('inf')
            self.results_dir = './results/models/'
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            self.train_losses = []
            self.val_losses = []

        def train(self, num_epochs):
            """
            Trains the LSTM model for a specified number of epochs.

            Args:
            - num_epochs: the number of epochs to train the model for
            """
            self.model.to(self.device)
            for epoch in range(num_epochs):
                train_loss = 0.0
                val_loss = 0.0
                self.model.train()
                with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(self.train_loader):.4f}, Val Loss: {val_loss/len(self.val_loader):.4f}",
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)) as pbar:
                    for i, (inputs, targets) in enumerate(self.train_loader):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        self.optimizer.step()
                        train_loss += loss.item()
                        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(self.train_loader):.4f}, Val Loss: {val_loss/len(self.val_loader):.4f}")
                        pbar.update(1)

                self.model.eval()
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(self.val_loader):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item()

                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(self.train_loader):.4f}, Val Loss: {val_loss/len(self.val_loader):.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = os.path.join(self.results_dir, f"best_model.pt")
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint at {checkpoint_path}")
                
                self.train_losses.append(train_loss/len(self.train_loader))
                self.val_losses.append(val_loss/len(self.val_loader))
        
        def plot_loss_curves(self):
            """
            Plots the training and validation loss curves.
            """
            sns.set_style("darkgrid")
            fig, ax = plt.subplots()
            sns.lineplot(x=range(1, len(self.train_losses)+1), y=self.train_losses, ax=ax, label="Training Loss")
            sns.lineplot(x=range(1, len(self.val_losses)+1), y=self.val_losses, ax=ax, label="Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Losses")
            os.makedirs('./results/loss_curves/', exist_ok=True)
            plt.savefig('./results/loss_curves/loss_curves.png', dpi=800)

