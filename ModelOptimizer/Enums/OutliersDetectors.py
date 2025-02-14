from enum import Enum

# Definindo um Enum
class OutliersDetectors(Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
