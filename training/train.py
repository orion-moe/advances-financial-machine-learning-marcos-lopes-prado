import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import logging
from utils.checkpoint import save_checkpoint
from utils.metrics import precision_score, recall_score, f1_score
from config.config import DEVICE

def prepare_dataloaders(X_train, y_train, X_val, y_val, batch_size=8192):
    try:
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy().astype(np.float32)
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.to_numpy().astype(np.float32)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=4)
        
        return train_loader, val_loader
    except Exception as e:
        logging.error(f"Erro durante a preparação dos DataLoaders: {e}")
        raise

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, fold=1, checkpoint_dir='checkpoints', load_checkpoint=True):
    try:
        model.to(DEVICE)
        scaler_amp = torch.cuda.amp.GradScaler()  # Inicializar mixed precision
        best_accuracy = 0.0
        best_metrics = {}
        
        fold_dir = os.path.join(checkpoint_dir, f'fold_{fold}')
        checkpoint_path = os.path.join(fold_dir, 'checkpoint.pth')

        start_epoch = 1

        if load_checkpoint and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler_amp.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_accuracy = checkpoint.get('best_accuracy', 0.0)
                best_metrics = checkpoint.get('best_metrics', {})
                logging.info(f"Checkpoint carregado para o fold {fold}, a partir da epoch {start_epoch}")
                print(f"Checkpoint carregado para o fold {fold}, a partir da epoch {start_epoch}")
            except RuntimeError as e:
                logging.error(f"Erro ao carregar state_dict do checkpoint para o fold {fold}: {e}")
                print(f"Erro ao carregar state_dict do checkpoint para o fold {fold}: {e}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            model.train()
            running_loss = 0.0
            y_true = []
            y_pred = []
            progress_bar = tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch}/{num_epochs}", leave=False)
            for batch_idx, (X_batch, y_batch) in enumerate(progress_bar, 1):
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

                running_loss += loss.item()
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({'Loss': f'{running_loss / batch_idx:.4f}'})
                
                _, preds = torch.max(outputs, 1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
            
            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = np.mean(np.array(y_pred) == np.array(y_true))
            train_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            train_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            train_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            logging.info(f"Fold {fold} - Epoch {epoch}/{num_epochs} - Treino - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")
            print(f"Fold {fold} - Epoch {epoch}/{num_epochs} - Loss de Treino: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")
            
            # Validação
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            y_val_true = []
            y_val_pred = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE, non_blocking=True)
                    y_batch = y_batch.to(DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)
                    y_val_true.extend(y_batch.cpu().numpy())
                    y_val_pred.extend(preds.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            val_precision = precision_score(y_val_true, y_val_pred, average='macro', zero_division=0)
            val_recall = recall_score(y_val_true, y_val_pred, average='macro', zero_division=0)
            val_f1 = f1_score(y_val_true, y_val_pred, average='macro', zero_division=0)

            logging.info(f"Fold {fold} - Epoch {epoch}/{num_epochs} - Validação - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
            print(f"Fold {fold} - Epoch {epoch}/{num_epochs} - Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
            
            # Salvar checkpoint
            metrics = {
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'f1_score': val_f1
            }
            is_best = val_accuracy > best_accuracy
            if is_best:
                best_accuracy = val_accuracy
                best_metrics = metrics
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler_amp.state_dict(),
                'best_accuracy': best_accuracy,
                'best_metrics': best_metrics,
            }, is_best, checkpoint_dir, fold)
        
        return model, best_metrics
    except Exception as e:
        logging.error(f"Erro durante a preparação dos DataLoaders: {e}")
        raise
