DATA_FOLDER='./data'
TIME_WINDOWS_LENGTH = 24*7 # 24 hours * 7 days
OVERLAP = 2 # 6 hours
PREDICTION_LENGTH = 24*7 # 24 hours * 3 days
TRAIN_VALID_TEST_SPLIT = [0.70, 0.15, 0.15] # 70% train, 15% valid, 15% test

# Model parameters
BATCH_SIZE = 32 # batch size for dataloader
LEARNING_RATE = 0.001# number of workers for dataloader
HIDDEN_SIZE = 128 # hidden size of LSTM
NUM_LAYERS = 2 # number of layers of LSTM
EPOCHS = 50