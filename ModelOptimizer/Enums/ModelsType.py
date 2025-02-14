from enum import Enum

# Definindo um Enum
class ModelsType(Enum):
    LogisticRegression = "LogisticRegression"
    RandomForest = "RandomForest"
    XGBoost = "XGBoost"
    KNN = "KNN"