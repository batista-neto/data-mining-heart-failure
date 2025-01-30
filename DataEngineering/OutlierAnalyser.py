import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from Enums import OutliersRemovers

class OutlierAnalyzer:
    """
    Classe para análise de outliers em um dataset.
    """

    def __init__(self, dataset, target_column, outlier_detector_strategy, outliers_remover_strategy):
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

    def transform(self, outliers_remover_type):
        outlier_remover = self.outlier_remover_strategy.outliers_remover_strategy(outliers_remover_type)
        transformed_dataset = outlier_remover(self.dataset, self.target_column)
        return transformed_dataset
    
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
            OutliersRemovers.OutliersRemovers.LOGARITIMO.value: FunctionTransformer(func=np.log1p, validate=True),
            OutliersRemovers.OutliersRemovers.RAIZ_QUADRADA.value: FunctionTransformer(func=np.sqrt, validate=True),
            OutliersRemovers.OutliersRemovers.YEOJOHNSON.value: PowerTransformer(method='yeo-johnson')
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
