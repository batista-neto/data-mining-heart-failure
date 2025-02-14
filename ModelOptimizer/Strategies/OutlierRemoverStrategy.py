from ModelOptimizer.Enums import OutliersRemovers
from ModelOptimizer.DataEngineering.OutliersRemover import OutliersRemove

class OutlierRemoverStrategy:
    def outliers_remover_strategy(self, outliers_remover_type):
        """
        Escolha o tipo de Detecçao de outlier que deseja utilizar.
        """
        if outliers_remover_type == OutliersRemovers.OutliersRemovers.LOGARITIMO.value:
            outliers_detector = OutliersRemove().log_transform
        elif outliers_remover_type == OutliersRemovers.OutliersRemovers.RAIZ_QUADRADA.value:
            outliers_detector = OutliersRemove().sqrt_transform
        elif outliers_remover_type == OutliersRemovers.OutliersRemovers.YEOJOHNSON.value:
            outliers_detector = OutliersRemove().yeojohnson_transform
        else:
            raise ValueError(f"'{outliers_remover_type}' não é suportado.")

        return outliers_detector
