from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np
from ModelOptimizer.Enums.Balances import BalanceTypes
from ModelOptimizer.Enums.ModelsType import ModelsType
from ModelOptimizer.Enums.Scalers import ScalerTypes

class BestModelPipeLine:
    def __init__(self, dataset, target_column, balance_analyser, data_scaler, outlier_analyzer, model_strategy):
        self.dataset = dataset
        self.target_column = target_column
        self.balance_analyser = balance_analyser
        self.data_scaler = data_scaler
        self.outlier_analyzer = outlier_analyzer
        self.model_strategy = model_strategy
        self.scores = {}
        self.best_model = None
        self.best_strategy = {}

    def evaluate_model(self, X, y, model):
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=1),
            'recall': make_scorer(recall_score, zero_division=1),
            'f1_score': make_scorer(f1_score, zero_division=1)
        }
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = {metric: [] for metric in scoring}

        for train_index, test_index in kfold.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_val)

            for metric, scorer in scoring.items():
                score = scorer._score_func(y_val, y_pred, **scorer._kwargs)
                scores[metric].append(score)

        return {metric: np.mean(values) for metric, values in scores.items()}
    
    def run_pipeline(self):
            best_score = -1

            # Teste de modelos
            for model_type in ModelsType:
                model_name = model_type.value
                model = self.model_strategy.ModelStrategy(model_name)
                print(f"Testando modelo {model_name}...")

                # Encontrar o melhor balanceamento
                dataset_balanced, balance_method = self.balance_analyser.find_best_balancing_strategy(self.dataset, self.target_column, BalanceTypes, model)
                
                # Encontrar o melhor tratamento de outliers
                dataset_transformed, transform_method = self.outlier_analyzer.find_best_outlier_strategy(dataset_balanced, self.target_column, model)

                # Encontrar a melhor normalização
                dataset_scaled, scaler_method = self.data_scaler.find_best_scaling_strategy(dataset_transformed, self.target_column, ScalerTypes, model)

                # Avaliar o modelo final
                X_scaled = dataset_scaled.drop(columns=[self.target_column])
                y_scaled = dataset_scaled[self.target_column]
                scores = self.evaluate_model(X_scaled, y_scaled, model)

                # Calcular o escore ponderado
                weighted_score = scores['accuracy'] + scores['precision'] + scores['recall'] + scores['f1_score']
                
                # Verificar se este é o melhor modelo
                print(f"Nota para o modelo {model_name} foi: {weighted_score}")
                if weighted_score > best_score:
                    best_score = weighted_score
                    self.best_model = model
                    self.best_strategy = (model_name, balance_method, transform_method, scaler_method)  # Substitua com as estratégias reais usadas

            print("Melhor estratégia encontrada:")
            print(f"Modelo: {self.best_strategy[0]}")
            print(f"Balanceamento: {self.best_strategy[1]}")
            print(f"Transformacao: {self.best_strategy[2]}")
            print(f"Scalonamento: {self.best_strategy[3]}")
            # Imprimir as estratégias de balanceamento, outlier e escalonamento reais
            print(f"Scores: {scores}")

            return self.best_model, balance_method, transform_method, scaler_method