import multiprocessing
import datetime
import os
import torch


timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filename = f"saved_model_{timestamp}.pt"

BATCH_SIZE = 8
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

TRAINING_STEPS = 100

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_MFCC = 40
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 3
BI_LSTM = True
FRAME_AGGREGATION_MEAN = True
LEARNING_RATE = 0.0001
SEQ_LEN = 100
SPECAUG_TRAINING = False
SAVE_MODEL_FREQUENCY=10


USE_FULL_SEQUENCE_INFERENCE = False
SLIDING_WINDOW_STEP = 50
EVAL_THRESHOLD_STEP = 0.001