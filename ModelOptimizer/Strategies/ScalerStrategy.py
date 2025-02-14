from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer
from ModelOptimizer.Enums import Scalers

class ScalerStrategy:
    def scale_strategy(self, scaler_type):
        """
        Escolha o tipo de Scaler que deseja utilizar.
        """
        if scaler_type == Scalers.ScalerTypes.MinMaxScaler.value:
            scaler = MinMaxScaler()
        elif scaler_type == Scalers.ScalerTypes.StandardScaler.value:
            scaler = StandardScaler()
        elif scaler_type == Scalers.ScalerTypes.RobustScaler.value:
            scaler = RobustScaler()
        elif scaler_type == Scalers.ScalerTypes.MaxAbsScaler.value:
            scaler = MaxAbsScaler()
        elif scaler_type == Scalers.ScalerTypes.Normalizer.value:
            scaler = Normalizer()
        else:
            raise ValueError(f"Scaler '{scaler_type}' não é suportado.")

        return scaler
