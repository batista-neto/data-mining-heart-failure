from Enums import OutliersDetectors
from DataEngineering.OutliersDetector import OutliersDetector

class OutlierDetectorStrategy:
    def outliers_detector_strategy(self, outliers_detector_type):
        """
        Escolha o tipo de Detecçao de outlier que deseja utilizar.
        """
        if outliers_detector_type == OutliersDetectors.OutliersDetectors.IQR.value:
            outliers_detector = OutliersDetector().detect_outliers_iqr
        elif outliers_detector_type == OutliersDetectors.OutliersDetectors.ZSCORE.value:
            outliers_detector = OutliersDetector().detect_outliers_zscore
        else:
            raise ValueError(f"'{outliers_detector_type}' não é suportado.")

        return outliers_detector
