import os
import pickle
import logging
from config.config import CHECKPOINT_DATA_PATH, FOLD_CHECKPOINT_PATH, ENSEMBLE_CHECKPOINT_PATH

def save_data_processing_checkpoint(last_bar_time):
    with open(CHECKPOINT_DATA_PATH, 'wb') as f:
        pickle.dump({'last_bar_time': last_bar_time}, f)

def load_data_processing_checkpoint():
    if os.path.exists(CHECKPOINT_DATA_PATH):
        with open(CHECKPOINT_DATA_PATH, 'rb') as f:
            checkpoint = pickle.load(f)
            return checkpoint.get('last_bar_time')
    return None

def save_fold_checkpoint(fold_number):
    with open(FOLD_CHECKPOINT_PATH, 'wb') as f:
        pickle.dump({'completed_folds': fold_number}, f)

def load_fold_checkpoint():
    if os.path.exists(FOLD_CHECKPOINT_PATH):
        with open(FOLD_CHECKPOINT_PATH, 'rb') as f:
            checkpoint = pickle.load(f)
            return checkpoint.get('completed_folds', 0)
    return 0

def save_ensemble_models(models, path=ENSEMBLE_CHECKPOINT_PATH):
    with open(path, 'wb') as f:
        pickle.dump([model.state_dict() for model in models], f)

def load_ensemble_models(model_architecture, path=ENSEMBLE_CHECKPOINT_PATH):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model_states = pickle.load(f)
        models = []
        for state in model_states:
            model = model_architecture()
            model.load_state_dict(state)
            model.to('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            models.append(model)
        return models
    return []
