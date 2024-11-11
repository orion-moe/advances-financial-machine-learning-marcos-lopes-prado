import torch
from torch import nn
import numpy as np
import logging
from config.config import DEVICE
from models.neural_net import NeuralNet

def ensemble_predictions(models, X, batch_size=8192):
    try:
        predictions = []
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
        
        for model in models:
            model.eval()
            model_preds = []
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    X_batch = X_tensor[start_idx:end_idx]
                    outputs = model(X_batch)
                    probs = nn.functional.softmax(outputs, dim=1)
                    model_preds.append(probs.cpu().numpy())
            model_preds = np.vstack(model_preds)
            predictions.append(model_preds)
        
        avg_predictions = np.mean(predictions, axis=0)
        ensemble_preds = np.argmax(avg_predictions, axis=1) - 1  # Reverter a codificação para -1,0,1
        return ensemble_preds
    except Exception as e:
        logging.error(f"Erro durante o ensembling das previsões: {e}")
        raise

def train_and_collect_models(X_train, y_train, X_val, y_val, params, num_models=5):
    from training.train import prepare_dataloaders, train_model
    from models.neural_net import NeuralNet
    import torch.optim as optim

    models = []
    for i in range(num_models):
        model = NeuralNet(
            input_size=X_train.shape[1],
            hidden_sizes=params['hidden_sizes'],
            num_classes=3,
            dropout=params['dropout_rate']
        ).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer_model = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        train_loader, val_loader = prepare_dataloaders(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=8192
        )
        
        model, best_metrics = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer_model, 
            num_epochs=params['num_epochs'],
            fold=1,  # Ajustar conforme necessário
            checkpoint_dir='ensemble_checkpoints',
            load_checkpoint=False
        )
        models.append(model)
        logging.info(f"Modelo {i+1}/{num_models} treinado com sucesso.")
        print(f"Modelo {i+1}/{num_models} treinado com sucesso.")
    return models
