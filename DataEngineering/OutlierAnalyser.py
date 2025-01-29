from readline import redisplay
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

class OutlierAnalyzer:
    """
    Classe para análise de outliers em um dataset.
    """

    def __init__(self, dataset, target_column):
        """
        Inicializa a classe com um dataset.
        """
        self.dataset = dataset
        self.target_column = target_column

    def detect_outliers_iqr(self):
        """
        Detecta outliers usando o método do Intervalo Interquartil (IQR) para todas as colunas numéricas.
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para análise de outliers.")
            return {}

        outliers = {}
        numeric_columns = self.dataset.select_dtypes(include=['number']).columns

        for column in numeric_columns:
            Q1 = self.dataset[column].quantile(0.25)
            Q3 = self.dataset[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_indices = self.dataset[(self.dataset[column] < lower_bound) | (self.dataset[column] > upper_bound)].index
            outliers[column] = outlier_indices.tolist()

        return outliers

    def detect_outliers_zscore(self, threshold=3):
        """
        Detecta outliers usando o método do z-score para todas as colunas numéricas.
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para análise de outliers.")
            return {}

        outliers = {}
        numeric_columns = self.dataset.select_dtypes(include=['number']).columns

        for column in numeric_columns:
            z_scores = (self.dataset[column] - self.dataset[column].mean()) / self.dataset[column].std()
            outlier_indices = self.dataset[(z_scores.abs() > threshold)].index
            outliers[column] = outlier_indices.tolist()

        return outliers
    
    def apply_log_transform(self):
        log_transformer = FunctionTransformer(func=np.log1p, validate=True)
        features = self.dataset.drop(columns=[self.target_column])
        target = self.dataset[self.target_column]
        transformed_features = log_transformer.transform(features)
        transformed_dataset = pd.concat([pd.DataFrame(transformed_features, columns=features.columns), target.reset_index(drop=True)], axis=1)
        return transformed_dataset

    def apply_sqrt_transform(self):
        sqrt_transformer = FunctionTransformer(func=np.sqrt, validate=True)
        transformed_dataset = sqrt_transformer.transform(self.dataset)
        return pd.DataFrame(transformed_dataset, columns=self.dataset.columns)

    def apply_yejohnson_transform(self):
        boxcox_transformer = PowerTransformer(method='yeo-johnson')
        transformed_dataset = boxcox_transformer.fit_transform(self.dataset)
        return pd.DataFrame(transformed_dataset, columns=self.dataset.columns)
    
    def evaluate_model(self, X, y, model, transformer):
        pipeline = Pipeline([
            ('transform', transformer),
            ('model', model)
        ])
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=1),
            'recall': make_scorer(recall_score, zero_division=1),
            'f1': make_scorer(f1_score, zero_division=1)
        }
        
        scores = cross_validate(pipeline, X, y, cv=5, scoring=scoring)
        return {
            'accuracy': scores['test_accuracy'].mean(),
            'precision': scores['test_precision'].mean(),
            'recall': scores['test_recall'].mean(),
            'f1_score': scores['test_f1'].mean()
        }
    
    def compare_transformations(self, model):
        """
        Compara as transformações de dados com base nas métricas de avaliação.
        """
        weights = {'accuracy': 1, 'precision': 1, 'recall': 1, 'f1_score': 1}
        best_transformations = []
        best_score = -1

        transformations = {
            'Log Transformation': FunctionTransformer(func=np.log1p, validate=True),
            'Sqrt Transformation': FunctionTransformer(func=np.sqrt, validate=True),
            'Box-Cox Transformation': PowerTransformer(method='yeo-johnson')
        }

        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        for name, transformer in transformations.items():
            score = self.evaluate_model(X, y, model, transformer)
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

    def check_custom_limits(self, min_value, max_value, column):
        """
        Verifica se os valores da coluna específica estão fora do intervalo definido por min_value e max_value.
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para análise de outliers.")
            return []

        # Verifica se a coluna existe no dataset
        if column not in self.dataset.columns:
            print(f"A coluna '{column}' não existe no dataset.")
            return []

        # Verifica se os valores estão fora do intervalo
        outlier_indices = self.dataset[(self.dataset[column] < min_value) | (self.dataset[column] > max_value)].index
        outliers = outlier_indices.tolist()
        print(f"Coluna '{column}' possui {len(outliers)} outliers: {outliers}")
        return outliers
