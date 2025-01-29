from sklearn.linear_model import LogisticRegression
from Enums.ModelsType import ModelsType

class ModelStrategy:
    def ModelStrategy(self, model_type):
        """
        Retorna o modelo que será usando para as análises
        """
        if model_type == ModelsType.LogisticRegression.value:
            model_type = LogisticRegression(solver='liblinear', random_state=42)
        else:
            raise ValueError(f"Scaler '{model_type}' não é suportado.")

        return model_type