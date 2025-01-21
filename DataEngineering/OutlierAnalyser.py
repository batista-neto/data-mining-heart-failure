from readline import redisplay
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

class OutlierAnalyzer:
    """
    Classe para análise de outliers em um dataset.
    """

    def __init__(self, dataset):
        """
        Inicializa a classe com um dataset.
        """
        self.dataset = dataset

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
