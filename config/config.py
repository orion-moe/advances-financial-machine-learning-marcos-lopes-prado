import os

# Paths para checkpoints e outros arquivos
CHECKPOINT_DATA_PATH = 'data_processing_checkpoint.pkl'
FOLD_CHECKPOINT_PATH = 'fold_checkpoint.pkl'
ENSEMBLE_CHECKPOINT_PATH = 'ensemble_models.pkl'
OPTUNA_STORAGE = 'sqlite:///optuna_study.db'
OPTUNA_STUDY_NAME = 'pytorch_model_tuning'

# Configuração do dispositivo para PyTorch
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuração de Logging
LOG_FILE = 'modeling_orion.log'

# Outros parâmetros globais
CH_HOST = '192.168.99.46'
CH_USER = 'default'
CH_PASSWORD = ''
CH_DATABASE = 'orion'

# Feature Engineering
FEATURES = ['log_return', 'vwap', 'ma_5', 'ma_10', 'ma_20', 'frac_diff', 'volatility', 'rsi']
