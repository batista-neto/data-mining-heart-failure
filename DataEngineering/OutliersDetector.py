import pandas as pd
import numpy as np
from scipy.stats import zscore

class OutliersDetector:
    """
    Classe para identificação de outliers em um dataset.
    """
    def detect_outliers_iqr(self, dataset):
        """
        Detecta outliers usando o método do Intervalo Interquartil (IQR) para todas as colunas numéricas.
        """
        outliers = {}
        numeric_columns = dataset.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            Q1 = dataset[column].quantile(0.25)
            Q3 = dataset[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices = dataset[(dataset[column] < lower_bound) | (dataset[column] > upper_bound)].index
            outliers[column] = outlier_indices.tolist()
        return outliers

    def detect_outliers_zscore(self, dataset, threshold=3):
        """
        Detecta outliers usando o método do z-score para todas as colunas numéricas.
        """
        outliers = {}
        numeric_columns = dataset.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            z_scores = zscore(dataset[column].dropna())
            outlier_indices = dataset[np.abs(z_scores) > threshold].index
            outliers[column] = outlier_indices.tolist()
        return outliers