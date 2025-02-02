from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

class DataScaler:
   def __init__(self, dataset, target_column, scaler_strategy):
       """
       Inicializa a classe com o dataset e a coluna alvo (target).
       """
       self.dataset = dataset
       self.target_column = target_column
       self.scaler_strategy = scaler_strategy
       self.evaluation_results = {}

   def split_data(self, test_size=0.02, val_size=0.03, random_state=42):
        """
        Divide os dados em treino, validação e teste de forma estratificada.
        """
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        val_relative_size = val_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val)

        return X_train, X_val, X_test, y_train, y_val, y_test


   def scale_data(self, method):
        """
        Aplica um metodo de escalonamento nos dados e retorna o dataset inteiro escalado.
        """
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        scaler = self.scaler_strategy.scale_strategy(method)
        X_scaled = scaler.fit_transform(X)
        scaled_dataset = pd.DataFrame(X_scaled, columns=X.columns)
        scaled_dataset[self.target_column] = y.values
        return scaled_dataset
   
   def cross_val_evaluate(self, model, method, cv=5):
        """
        Realiza validação cruzada e avalia o modelo.
        """
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        scaler = self.scaler_strategy.scale_strategy(method)
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=1),
            'recall': make_scorer(recall_score, zero_division=1),
            'f1_score': make_scorer(f1_score, zero_division=1)
        }

        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = {metric: [] for metric in scoring}

        for train_index, test_index in kfold.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            # Transformar apenas os dados de treino
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Manter os nomes das características
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)

            # Treinar o modelo com os dados transformados
            model_clone = clone(model)
            model_clone.fit(X_train_scaled, y_train)

            # Fazer previsões com os dados não transformados
            y_pred = model_clone.predict(X_val_scaled)

            # Avaliar as métricas
            for metric, scorer in scoring.items():
                score = scorer._score_func(y_val, y_pred, **scorer._kwargs)
                scores[metric].append(score)

        # Calcular a média das métricas
        mean_scores = {metric: np.mean(values) for metric, values in scores.items()}
        self.evaluate_cross_val_scores(mean_scores, method)

   def evaluate_cross_val_scores(self, scores, method):
        self.evaluation_results[method] = {
            "accuracy": round(scores['accuracy'].mean(), 2),
            "precision": round(scores['precision'].mean(), 2),
            "recall": round(scores['recall'].mean(), 2),
            "f1_score": round(scores['f1_score'].mean(), 2)
        }

        print(f"Resultados para {method}:")
        print(f"Acurácia: {scores['accuracy'].mean():.2f}")
        print(f"Precisão: {scores['precision'].mean():.2f}")
        print(f"Recall: {scores['recall'].mean():.2f}")
        print(f"F1-Score: {scores['f1_score'].mean():.2f}")
  
   def evaluate_model(self, y_true, y_pred, method):
        """
        Avaliar o modelo e armazenar os resultados.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\nResultados para {method}:")
        print(f"Acurácia: {accuracy:.2f}")
        print(f"Precisão: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

   def find_best_scaling_method(self):
        """
        Determina o melhor método de escalonamento com base no maior F1-Score.
        """
        if not self.evaluation_results:
            print("Nenhum resultado de avaliação disponível. Execute evaluate_model primeiro.")
            return None
        
        weights = {'accuracy': 1, 'precision': 1, 'recall': 1, 'f1_score': 1}
        best_methods = []
        best_score = -1

        for method, metrics in self.evaluation_results.items():
            score = (metrics['accuracy'] * weights['accuracy'] +
                    metrics['precision'] * weights['precision'] +
                    metrics['recall'] * weights['recall'] +
                    metrics['f1_score'] * weights['f1_score'])

            if score > best_score:
                best_score = score
                best_methods = [method] 
            elif score == best_score:
                best_methods.append(method) 

        if len(best_methods) > 1:
            print(f"Empate entre os seguintes métodos com pontuação {best_score:.2f}: {', '.join(best_methods)}")
        else:
            print(f"O melhor método é: {best_methods[0]} com pontuação {best_score:.2f}")

        return best_methods


