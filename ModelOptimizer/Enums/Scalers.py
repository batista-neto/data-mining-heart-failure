from enum import Enum

# Definindo um Enum
class ScalerTypes(Enum):
    MinMaxScaler = "minmax"
    StandardScaler = "standard"
    RobustScaler = "robust"
    MaxAbsScaler = "maxabs"
    Normalizer = "normalizer"
