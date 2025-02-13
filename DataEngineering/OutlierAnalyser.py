import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from Enums import OutliersRemovers
from sklearn.base import clone
import pandas as pd
from Dictionaries.features_limits import features_limits

class OutlierAnalyzer:
    """
    Classe para análise de outliers em um dataset.
    """

    def __init__(self, outlier_detector_strategy, outliers_remover_strategy, dataset=None, target_column=None):
        """
        Inicializa a classe com um dataset.
        """
        self.dataset = dataset
        self.target_column = target_column
        self.outlier_detector_strategy = outlier_detector_strategy
        self.outlier_remover_strategy = outliers_remover_strategy

    def detect_outliers(self, outliers_detector_strategy):
        """
        Detecta outliers usando o método do Intervalo Interquartil (IQR) para todas as colunas numéricas.
        """
        outliers_detector = self.outlier_detector_strategy.outliers_detector_strategy(outliers_detector_strategy)
        outliers = outliers_detector(self.dataset)
        for column, indices in outliers.items():
            if indices:
                print(f"Coluna '{column}' possui {len(indices)} outliers: {indices}")
        return outliers

    def transform(self, outliers_remover_type, dataset=None, target_column=None):
        if dataset is None:
            dataset = self.dataset

        if target_column is None:
            target_column = self.target_column

        outlier_remover = self.outlier_remover_strategy.outliers_remover_strategy(outliers_remover_type)
        transformed_dataset = outlier_remover(dataset, target_column)
        return transformed_dataset
    
    def evaluate_model(self, X, y, model, transformer, cv=5):
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

            # Transformar os dados
            X_train_transformed = transformer.fit_transform(X_train)
            X_val_transformed = transformer.transform(X_val)

            # Manter os nomes das características
            X_train_scaled = pd.DataFrame(X_train_transformed, columns=X_train.columns)
            X_val_scaled = pd.DataFrame(X_val_transformed, columns=X_val.columns)

            # Treinar o modelo com os dados transformados e escalados
            model_clone = clone(model)
            model_clone.fit(X_train_scaled, y_train)

            # Fazer previsões com os dados transformados e escalados
            y_pred = model_clone.predict(X_val_scaled)

            # Avaliar as métricas
            for metric, scorer in scoring.items():
                score = scorer._score_func(y_val, y_pred, **scorer._kwargs)
                scores[metric].append(score)

        return {metric: np.mean(values) for metric, values in scores.items()}
    
    def compare_transformations(self, model):
        """
        Compara as transformações de dados com base nas métricas de avaliação.
        """
        weights = {'accuracy': 1, 'precision': 1, 'recall': 1, 'f1_score': 1}
        best_transformations = []
        best_score = -1

        transformations = {
            OutliersRemovers.OutliersRemovers.LOGARITIMO.value: FunctionTransformer(func=np.log1p, validate=True),
            OutliersRemovers.OutliersRemovers.RAIZ_QUADRADA.value: FunctionTransformer(func=np.sqrt, validate=True),
            OutliersRemovers.OutliersRemovers.YEOJOHNSON.value: PowerTransformer(method='yeo-johnson')
        }

        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        for name, transformer in transformations.items():
            score = self.evaluate_model(X, y, model, transformer, )
            weighted_score = (score['accuracy'] * weights['accuracy'] +
                            score['precision'] * weights['precision'] +
                            score['recall'] * weights['recall'] +
                            score['f1_score'] * weights['f1_score'])

            if weighted_score > best_score:
                best_score = weighted_score
                best_transformations = [name]
            elif weighted_score == best_score:
                best_transformations.append(name)

            print(f"{name} - Accuracy: {score['accuracy']:.4f}, Precision: {score['precision']:.4f}, Recall: {score['recall']:.4f}, F1-Score: {score['f1_score']:.4f}")

        if len(best_transformations) > 1:
            print(f"Empate entre as seguintes transformações com pontuação {best_score:.2f}: {', '.join(best_transformations)}")
        else:
            print(f"A melhor transformação é: {best_transformations[0]} com pontuação {best_score:.2f}")

        return best_transformations

    def check_custom_limits(self):
        """
        Verifica se há valores absurdos no dataset com base nos limites definidos em features_limits.
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para análise de outliers.")
            return {}

        outliers_detected = {}

        for column, (min_value, max_value) in features_limits.items():
            if column not in self.dataset.columns:
                print(f"A coluna '{column}' não existe no dataset.")
                continue
            
            # Identifica os índices com valores fora do intervalo aceitável
            outlier_indices = self.dataset[(self.dataset[column] < min_value) | (self.dataset[column] > max_value)].index
            outliers_list = outlier_indices.tolist()

            if outliers_list:
                print(f"Coluna '{column}' possui {len(outliers_list)} valores absurdos: {outliers_list}")
                outliers_detected[column] = outliers_list

        return outliers_detected
    
    def find_best_outlier_strategy(self, dataset, target_column, model):
            print("Iniciando avaliação das estratégias de tratamento de outliers...")
            weights = {'accuracy': 1, 'precision': 1, 'recall': 1, 'f1_score': 1}
            best_removal_methods = []
            best_score = -1

            removal_methods = {
                OutliersRemovers.OutliersRemovers.LOGARITIMO.value: FunctionTransformer(func=np.log1p, validate=True),
                OutliersRemovers.OutliersRemovers.RAIZ_QUADRADA.value: FunctionTransformer(func=np.sqrt, validate=True),
                OutliersRemovers.OutliersRemovers.YEOJOHNSON.value: PowerTransformer(method='yeo-johnson')
            }

            X = dataset.drop(columns=[target_column])
            y = dataset[target_column]

            for name, transformer in removal_methods.items():
                print(f"Avaliando método de remoção de outliers: {name}")
                score = self.evaluate_model(X, y, model, transformer)
                weighted_score = (score['accuracy'] * weights['accuracy'] +
                                score['precision'] * weights['precision'] +
                                score['recall'] * weights['recall'] +
                                score['f1_score'] * weights['f1_score'])

                if weighted_score > best_score:
                    best_score = weighted_score
                    best_removal_methods = [name]
                elif weighted_score == best_score:
                    best_removal_methods.append(name)

                print(f"{name} - Accuracy: {score['accuracy']:.4f}, Precision: {score['precision']:.4f}, Recall: {score['recall']:.4f}, F1-Score: {score['f1_score']:.4f}")

            if len(best_removal_methods) > 1:
                print(f"Empate entre as seguintes transformações com pontuação {best_score:.2f}: {', '.join(best_removal_methods)}")
            else:
                print(f"A melhor transformação é: {best_removal_methods[0]} com pontuação {best_score:.2f}")

            print("Avaliação concluída. A melhor estratégia foi:", best_removal_methods[0])
            dataset_transformed = self.transform(best_removal_methods[0], dataset, target_column)
            return dataset_transformed