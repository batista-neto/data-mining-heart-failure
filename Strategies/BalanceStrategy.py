from Enums import Balances
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

class BalanceStrategy:
    def balance_strategy(self, balance_type):
        """
        Escolha o tipo de Scaler que deseja utilizar.
        """
        if balance_type == Balances.BalanceTypes.Oversampling.value:
            balance = SMOTE(random_state=42)
        elif balance_type == Balances.BalanceTypes.Undersampling.value:
            balance = RandomUnderSampler(random_state=42)
        elif balance_type == Balances.BalanceTypes.SMOTETomek.value:
            balance = SMOTETomek(random_state=42)
        else:
            raise ValueError(f"Balancer '{balance_type}' não é suportado.")

        return balance
