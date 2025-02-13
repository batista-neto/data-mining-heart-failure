from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from Enums.ModelsType import ModelsType

class ModelStrategy:
    def ModelStrategy(self, model_type):
        """
        Retorna o modelo que será usando para as análises
        """
        if model_type == ModelsType.LogisticRegression.value:
            model_type = LogisticRegression(solver='liblinear', random_state=42)
        elif model_type == ModelsType.RandomForest.value:
            model_type =  RandomForestClassifier(random_state=42)
        elif model_type == ModelsType.XGBoost.value:
            model_type =  XGBClassifier(random_state=42)
        elif model_type == ModelsType.KNN.value:
            model_type = KNeighborsClassifier()
        else:
            raise ValueError(f"Scaler '{model_type}' não é suportado.")

        return model_type