import optuna
from config.config import OPTUNA_STORAGE, OPTUNA_STUDY_NAME
from training.train import prepare_dataloaders, train_model
from models.neural_net import NeuralNet
from utils.checkpoint import save_checkpoint
from config.config import DEVICE

def get_optuna_study():
    return optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction='maximize',
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=1)
    )

def objective(trial, X_train, y_train, X_val, y_val):
    hidden_size1 = trial.suggest_int('hidden_size1', 128, 512)
    hidden_size2 = trial.suggest_int('hidden_size2', 64, 256)
    hidden_size3 = trial.suggest_int('hidden_size3', 32, 128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 8e-6)
    num_epochs = trial.suggest_int('num_epochs', 1, 2)

    model = NeuralNet(
        input_size=X_train.shape[1],
        hidden_sizes=[hidden_size1, hidden_size2, hidden_size3],
        num_classes=3,
        dropout=dropout_rate
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        optimizer, 
        num_epochs=num_epochs,
        fold=1,
        checkpoint_dir='optuna_checkpoints',
        load_checkpoint=False
    )

    return best_metrics['accuracy']
