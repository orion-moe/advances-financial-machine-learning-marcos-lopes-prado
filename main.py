import os
import gc
import numpy as np
import pandas as pd
import torch
import joblib
import shap
import warnings
import logging
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from training.hyperparameter_tuning import objective, get_optuna_study
from data.data_loader import connect_clickhouse, get_tick_bars_in_chunks
from data.preprocessing import (
    triple_barrier_method,
    fractional_difference,
    process_batch,
    calculate_vwap,
    calculate_moving_averages
)
from models.neural_net import NeuralNet
from training.train import prepare_dataloaders, train_model
from ensembling.ensembling import ensemble_predictions, train_and_collect_models
from reporting.report import generate_pdf_report
from utils.checkpoint import (
    save_fold_checkpoint, 
    load_fold_checkpoint, 
    save_ensemble_models, 
    load_ensemble_models,
    save_data_processing_checkpoint
)
from utils.logger import setup_logging
from config.config import (
    DEVICE, CHECKPOINT_DATA_PATH, FOLD_CHECKPOINT_PATH, ENSEMBLE_CHECKPOINT_PATH,
    OPTUNA_STORAGE, OPTUNA_STUDY_NAME, FEATURES
)
from torch.cuda import amp

# Configuração de Logging
setup_logging()

warnings.filterwarnings('ignore')

def analyze_feature_importance_pytorch(model, X, report_dir, batch_size=2048):
    try:
        model.eval()
        
        background_size = min(1000, X.shape[0])
        background = X.sample(n=background_size).values.astype(np.float32)
        
        def model_forward(x):
            x_tensor = torch.from_numpy(x).to(DEVICE).float()
            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()
        
        explainer = shap.Explainer(model_forward, background)
        
        shap_values = []
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        with tqdm(total=num_batches, desc="Computing SHAP values", unit=" batches") as pbar:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X.iloc[start_idx:end_idx].values.astype(np.float32)
                shap_batch = explainer(X_batch)
                shap_values.append(shap_batch.values)
                pbar.update(1)
        
        shap_values = np.vstack(shap_values)
        
        target_class = 1
        target_shap = shap_values[:, target_class, :]
        
        feature_importance = np.abs(target_shap).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        })
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        top_features = feature_importance_df['feature'].head(20).tolist()
        
        logging.info(f"Top 20 features based on SHAP: {top_features}")
        print(f"Top 20 features based on SHAP: {top_features}")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(target_shap[:, :20], X[top_features], plot_type="bar", show=False, class_names=[f"Class {c}" for c in range(3)])
        plt.tight_layout()
        feature_img_path = os.path.join(report_dir, f'feature_importance_class_{target_class}.png')
        plt.savefig(feature_img_path, dpi=300)
        plt.close()
        logging.info(f"Feature importance plot saved: {feature_img_path}")
        print(f"Feature importance plot saved: {feature_img_path}")
        
        return top_features
    except Exception as e:
        logging.error(f"Erro durante a análise de importância das features: {e}")
        raise

def recursive_feature_elimination(model, X, y, step=1, cv=5):
    from sklearn.feature_selection import RFECV
    rfecv = RFECV(estimator=model, step=step, cv=cv, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X, y)
    selected_features = X.columns[rfecv.support_].tolist()
    logging.info(f"Selected features after RFE: {selected_features}")
    print(f"Selected features after RFE: {selected_features}")
    return selected_features

def main_pytorch():
    try:
        chunk_size = 500000
        logging.info("Iniciando a consulta das tick bars para análise...")
        print("Consultando tick bars para análise...")

        client = connect_clickhouse()
        total_estimate = 0
        try:
            total_estimate = client.execute('SELECT count() FROM tick_bars')[0][0]
            logging.info(f"Total de registros estimados: {total_estimate}")
        except Exception as e:
            logging.error(f"Erro ao estimar o total de registros: {e}")
            print("Erro ao estimar o total de registros.")
            total_estimate = 0

        last_bar_time = load_data_processing_checkpoint()

        all_data = []
        removed_records = []
        report_dir = "report_output"
        checkpoint_dir = "checkpoints"
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        for df_chunk in get_tick_bars_in_chunks(client, chunk_size, total_estimate, last_bar_time):
            process_batch(df_chunk, all_data, removed_records)
            gc.collect()

        if os.path.exists(CHECKPOINT_DATA_PATH):
            os.remove(CHECKPOINT_DATA_PATH)

        data = pd.concat(all_data, ignore_index=True)
        del all_data
        gc.collect()

        logging.info("Curadoria de dados concluída.")
        print("Curadoria de dados concluída.")

        if removed_records:
            removed_data = pd.concat(removed_records, ignore_index=True)
            removed_data.to_csv(os.path.join(report_dir, 'removed_records.csv'), index=False)
            removed_data_count = len(removed_data)
            logging.info(f"Total de registros removidos: {removed_data_count}")
            print(f"Total de registros removidos: {removed_data_count}")
        else:
            removed_data_count = 0
            logging.info("Nenhum registro removido durante a curadoria de dados.")
            print("Nenhum registro removido durante a curadoria de dados.")

        # Feature Engineering
        logging.info("Iniciando Feature Engineering...")
        print("Iniciando Feature Engineering...")

        data['frac_diff'] = fractional_difference(data['close'], d=0.5)
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['log_return'].rolling(window=20).std()

        delta = data['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(window=14).mean()
        roll_down = down.rolling(window=14).mean()
        rs = roll_up / roll_down
        data['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        data = data.dropna()

        logging.info("Feature Engineering concluída.")
        print("Feature Engineering concluída.")

        # Labeling
        logging.info("Aplicando Triple Barrier Method para labeling...")
        print("Aplicando Triple Barrier Method para labeling...")
        data = triple_barrier_method(data, pt_sl=(0.02, 0.02), max_hold=10)
        logging.info("Labeling concluído.")
        print("Labeling concluído.")

        # Seleção de features
        data = data.dropna(subset=FEATURES + ['label'])

        X = data[FEATURES]
        y = data['label']

        # Codificar as labels para multi-classe
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)  # Transforma -1,0,1 em 0,1,2

        # Escalonamento das features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float32))
        X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

        # Divisão dos dados usando TimeSeriesSplit com Purging
        logging.info("Iniciando validação cruzada purged com PyTorch...")
        print("Iniciando validação cruzada purged com PyTorch...")
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        models = []

        completed_folds = load_fold_checkpoint()

        with tqdm(total=tscv.get_n_splits(), desc="Validação Cruzada", unit=" fold") as pbar_cv:
            for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled), 1):
                if fold <= completed_folds:
                    pbar_cv.update(1)
                    continue  # Pular folds já concluídos

                logging.info(f"Treinamento no fold {fold}")
                print(f"\nTreinamento no fold {fold}")
                X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
                y_train, y_test = y_encoded[train_index], y_encoded[test_index]

                # Iniciar o estudo Optuna com armazenamento persistente
                logging.info(f"Iniciando otimização de hiperparâmetros para o fold {fold}...")
                print(f"Iniciando otimização de hiperparâmetros para o fold {fold}...")
                
                study = get_optuna_study()
                study.optimize(
                    lambda trial: objective(trial, X_train, y_train, X_test, y_test), 
                    n_trials=5, 
                    n_jobs=1,
                    show_progress_bar=True
                )
                
                best_params = study.best_trial.params
                
                best_params['hidden_sizes'] = [
                    best_params.pop('hidden_size1'),
                    best_params.pop('hidden_size2'),
                    best_params.pop('hidden_size3')
                ]
                
                logging.info(f"Melhores parâmetros encontrados: {best_params}")
                print(f"Melhores parâmetros encontrados: {best_params}")

                # Selecionar as melhores features usando SHAP
                logging.info("Selecionando as melhores features com SHAP...")
                print("Selecionando as melhores features com SHAP...")
                model = NeuralNet(
                    input_size=X_train.shape[1],
                    hidden_sizes=best_params['hidden_sizes'],
                    num_classes=3,
                    dropout=best_params['dropout_rate']
                ).to(DEVICE)

                criterion = nn.CrossEntropyLoss()
                optimizer_model = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

                train_loader, val_loader = prepare_dataloaders(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    batch_size=8192
                )

                model, best_metrics = train_model(
                    model, 
                    train_loader, 
                    val_loader, 
                    criterion, 
                    optimizer_model, 
                    num_epochs=best_params['num_epochs'],
                    fold=fold,
                    checkpoint_dir=checkpoint_dir,
                    load_checkpoint=False
                )
                models.append(model)
                scores.append(best_metrics)

                # Analisar importância das features
                top_features = analyze_feature_importance_pytorch(model, X_train, report_dir)

                # Aplicar RFE com um modelo simples (Logistic Regression)
                logging.info("Aplicando Recursive Feature Elimination (RFE)...")
                print("Aplicando Recursive Feature Elimination (RFE)...")
                lr = LogisticRegression(max_iter=1000, n_jobs=-1)
                selected_features_rfe = recursive_feature_elimination(lr, X_train, y_train)
                
                # Combinar as features selecionadas por SHAP e RFE
                final_features = list(set(top_features).intersection(set(selected_features_rfe)))
                logging.info(f"Final selected features: {final_features}")
                print(f"Final selected features: {final_features}")

                if not final_features:
                    final_features = top_features[:10]
                    logging.info(f"Usando top 10 features de SHAP: {final_features}")
                    print(f"Usando top 10 features de SHAP: {final_features}")

                # Atualizar X_train e X_test com as features selecionadas
                X_train_selected = X_train[final_features]
                X_test_selected = X_test[final_features]

                # Re-escalar as features selecionadas
                scaler_selected = StandardScaler()
                X_train_scaled_selected = scaler_selected.fit_transform(X_train_selected.astype(np.float32))
                X_test_scaled_selected = scaler_selected.transform(X_test_selected.astype(np.float32))
                X_train_scaled_selected = pd.DataFrame(X_train_scaled_selected, columns=final_features)
                X_test_scaled_selected = pd.DataFrame(X_test_scaled_selected, columns=final_features)

                # Preparar DataLoaders novamente com as features selecionadas
                train_loader_selected, val_loader_selected = prepare_dataloaders(
                    X_train_scaled_selected,
                    y_train,
                    X_test_scaled_selected,
                    y_test,
                    batch_size=8192
                )

                # Treinar o modelo novamente com as features selecionadas
                model_selected = NeuralNet(
                    input_size=X_train_scaled_selected.shape[1],
                    hidden_sizes=best_params['hidden_sizes'],
                    num_classes=3,
                    dropout=best_params['dropout_rate']
                ).to(DEVICE)

                optimizer_selected = torch.optim.Adam(model_selected.parameters(), lr=best_params['learning_rate'])

                model_selected, best_metrics_selected = train_model(
                    model_selected, 
                    train_loader_selected, 
                    val_loader_selected, 
                    criterion, 
                    optimizer_selected, 
                    num_epochs=best_params['num_epochs'],
                    fold=fold,
                    checkpoint_dir=checkpoint_dir,
                    load_checkpoint=False
                )
                models.append(model_selected)
                scores.append(best_metrics_selected)

                pbar_cv.update(1)

                # Salvar o checkpoint do fold após concluir
                save_fold_checkpoint(fold)

        if os.path.exists(FOLD_CHECKPOINT_PATH):
            os.remove(FOLD_CHECKPOINT_PATH)

        metrics = pd.DataFrame(scores).mean()
        logging.info(f"Métricas médias de validação cruzada:\n{metrics}")
        print("\nMétricas médias de validação cruzada:")
        print(metrics)

        metrics_dict = metrics.to_dict()

        # Análise de importância das features (usando o melhor modelo)
        logging.info("Analisando a importância das features com o modelo final...")
        print("Analisando a importância das features com o modelo final...")
        best_fold_index = np.argmax([score['accuracy'] for score in scores])
        best_model = models[best_fold_index]
        top_features_final = analyze_feature_importance_pytorch(best_model, X_scaled, report_dir)

        # Selecionar as melhores features para o ensembling
        logging.info("Selecionando features para o ensembling...")
        print("Selecionando features para o ensembling...")
        lr_final = LogisticRegression(max_iter=1000, n_jobs=-1)
        selected_features_rfe_final = recursive_feature_elimination(lr_final, X_scaled, y_encoded)
        final_features_ensemble = list(set(top_features_final).intersection(set(selected_features_rfe_final)))
        logging.info(f"Final selected features for ensembling: {final_features_ensemble}")
        print(f"Final selected features for ensembling: {final_features_ensemble}")

        if not final_features_ensemble:
            final_features_ensemble = top_features_final[:10]
            logging.info(f"Usando top 10 features de SHAP para ensembling: {final_features_ensemble}")
            print(f"Usando top 10 features de SHAP para ensembling: {final_features_ensemble}")

        # Preparar os dados finais com as features selecionadas
        X_final = data[final_features_ensemble]
        y_final = data['label']
        y_final_encoded = le.transform(y_final)

        # Escalonar as features finais
        scaler_final = StandardScaler()
        X_final_scaled = scaler_final.fit_transform(X_final.astype(np.float32))
        X_final_scaled = pd.DataFrame(X_final_scaled, columns=final_features_ensemble)

        # Divisão dos dados para ensembling (usando TimeSeriesSplit novamente)
        tscv_final = TimeSeriesSplit(n_splits=5)
        ensemble_models = []
        ensemble_scores = []

        # Carregar os modelos do ensemble se existirem
        ensemble_models = load_ensemble_models(
            lambda: NeuralNet(input_size=len(final_features_ensemble), hidden_sizes=metrics_dict.get('hidden_sizes', [256, 128, 64]), num_classes=3, dropout=metrics_dict.get('dropout_rate', 0.3)),
            ENSEMBLE_CHECKPOINT_PATH
        )

        if not ensemble_models:
            with tqdm(total=tscv_final.get_n_splits(), desc="Ensembling Cross-Validation", unit=" fold") as pbar_cv:
                for fold, (train_index, test_index) in enumerate(tscv_final.split(X_final_scaled), 1):
                    logging.info(f"Ensembling Treinamento no fold {fold}")
                    print(f"\nEnsembling Treinamento no fold {fold}")
                    X_train_ens, X_test_ens = X_final_scaled.iloc[train_index], X_final_scaled.iloc[test_index]
                    y_train_ens, y_test_ens = y_final_encoded[train_index], y_final_encoded[test_index]

                    logging.info("Treinando múltiplos modelos para ensembling...")
                    print("Treinando múltiplos modelos para ensembling...")
                    models_fold = train_and_collect_models(
                        X_train_ens,
                        y_train_ens,
                        X_test_ens,
                        y_test_ens,
                        params=metrics_dict,
                        num_models=5
                    )
                    ensemble_models.extend(models_fold)
                    ensemble_scores.extend([metrics_dict] * len(models_fold))

                    pbar_cv.update(1)

            save_ensemble_models(ensemble_models)
        else:
            logging.info("Modelos de ensemble carregados com sucesso.")
            print("Modelos de ensemble carregados com sucesso.")

        # Backtesting
        logging.info("Iniciando Backtest com o ensemble de modelos PyTorch...")
        print("Iniciando Backtest com o ensemble de modelos PyTorch...")

        X_all = scaler_final.transform(X_final.astype(np.float32))
        X_all = pd.DataFrame(X_all, columns=final_features_ensemble)

        predictions = ensemble_predictions(ensemble_models, X_all)

        data = data.reset_index(drop=True)
        data['prediction'] = predictions

        data['strategy_returns'] = data['log_return'] * data['prediction'].shift(1)
        data['strategy_returns'] = data['strategy_returns'].fillna(0)
        data['cumulative_returns'] = data['strategy_returns'].cumsum().apply(np.exp)

        plt.figure(figsize=(14,7))
        plt.plot(data['bar_time'], data['cumulative_returns'], label='Strategy Returns')
        plt.plot(data['bar_time'], np.exp(data['log_return'].cumsum()), label='Market Returns')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.title('Backtest de Estratégia com PyTorch Ensemble')
        plt.tight_layout()
        cumulative_returns_path = os.path.join(report_dir, 'cumulative_returns.png')
        plt.savefig(cumulative_returns_path, dpi=300)
        plt.close()
        logging.info(f"Gráfico de retornos cumulativos salvo em: {cumulative_returns_path}")
        print(f"Gráfico de retornos cumulativos salvo em: {cumulative_returns_path}")

        sharpe_ratio = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(252*390)
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        metrics_dict['sharpe_ratio'] = sharpe_ratio

        # Gerar o relatório em PDF
        logging.info("Gerando relatório em PDF...")
        print("Gerando relatório em PDF...")
        generate_pdf_report(metrics_dict, report_dir, removed_data_count)

        # Salvar o scaler final e o melhor modelo treinado
        from utils.checkpoint import save_model
        save_model(ensemble_models[0], scaler_final, report_dir, fold=1, filename='ensemble_model.safetensors')

        logging.info("Backtest concluído com sucesso.")
        print("Backtest concluído com sucesso.")
    except Exception as e:
        logging.error(f"Erro durante a execução principal: {e}")
        print(f"Erro durante a execução principal: {e}")
        raise

if __name__ == "__main__":
    main_pytorch()
