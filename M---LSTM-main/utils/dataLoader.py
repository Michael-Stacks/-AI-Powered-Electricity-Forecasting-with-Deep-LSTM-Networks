
import os
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from config import *
from tqdm import tqdm
from colorama import Fore

class LoadingData:
    __instance = None
    __data: pd.DataFrame = None
    __scaler: MinMaxScaler = None
    __columns: list[str] = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance._load_data()
            cls.__instance._handle_scaler()
        return cls.__instance

    def _load_data(self, data_dir=DATA_FOLDER):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
                data = pd.read_csv(file_path)
                data['DateTime'] = pd.to_datetime(data['DateTime'])
                data = data.set_index('DateTime')
                data = data.loc[:,['Production', 'Consumption']]
                self.__columns = data.columns
                self.__data = data

    def get_data(self) -> pd.DataFrame:
        return self.__data
    
    def get_scaler(self) -> MinMaxScaler:
        return self.__scaler
    
    def get_columns(self) -> list[str]:
        return self.__columns
    
    def _handle_scaler(self):
        self.__scaler: MinMaxScaler = MinMaxScaler()
        data_scaler: np.array = self.__scaler.fit_transform(self.__data)
        self.__data = pd.DataFrame(data_scaler, columns=self.__data.columns, index=self.__data.index)
        
    from tqdm import tqdm

    def _prepare_data(self, time_windows_length:int, overlap:int, prediction_length:int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        data = self.get_data()
        time_window = []
        for i in tqdm(range(0, len(data) - overlap, overlap), desc='Preparing data', bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            if i + time_windows_length + prediction_length <= len(data):
                X = data.iloc[i:i + time_windows_length]
                y = data.iloc[i + time_windows_length:i + time_windows_length + prediction_length]
                time_window.append((X, y))
                
        return time_window
    
    def get_train_valid_test_dataloader(self, time_windows_length:int, overlap:int, prediction_length:int,
                                        train_valid_test_split:list[float] = [0.70, 0.15, 0.15],
                                        num_workers=0, batch_size=32, shuffle=True,
                                        ) -> tuple[DataLoader, DataLoader, DataLoader]:
        data = self._prepare_data(time_windows_length=time_windows_length, overlap=overlap, prediction_length=prediction_length)
        n = len(data)
        train_size = int(train_valid_test_split[0] * n)
        valid_size = int(train_valid_test_split[1] * n)
        test_size = n - train_size - valid_size
        train_dataset = CustomDataset(data[:train_size])
        valid_dataset = CustomDataset(data[train_size:train_size + valid_size])
        test_dataset = CustomDataset(data[train_size + valid_size:])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader, valid_loader, test_loader
    
class CustomDataset(Dataset):
    def __init__(self, data: list[tuple[pd.DataFrame, pd.DataFrame]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X, y = self.data[idx]
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        return X, y
    
    