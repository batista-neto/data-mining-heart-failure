from enum import Enum

# Definindo um Enum
class BalanceTypes(Enum):
    Oversampling = "oversampling"
    Undersampling = "undersampling"
    SMOTETomek = "smotetomek"
