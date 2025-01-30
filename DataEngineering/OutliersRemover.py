import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

class OutliersRemove:
    """
    Classe para identificação de outliers em um dataset.
    """
    def log_transform(self, dataset, target_column):
        log_transformer = FunctionTransformer(func=np.log1p, validate=True)
        features = dataset.drop(columns=[target_column])
        target = dataset[target_column]
        transformed_features = log_transformer.transform(features)
        transformed_dataset = pd.concat([pd.DataFrame(transformed_features, columns=features.columns), target.reset_index(drop=True)], axis=1)
        return transformed_dataset

    def sqrt_transform(self, dataset, target_column):
        sqrt_transformer = FunctionTransformer(func=np.sqrt, validate=True)
        transformed_dataset = sqrt_transformer.transform(dataset)
        return pd.DataFrame(transformed_dataset, columns=dataset.columns)

    def yeojohnson_transform(self, dataset, target_column):
        boxcox_transformer = PowerTransformer(method='yeo-johnson')
        transformed_dataset = boxcox_transformer.fit_transform(dataset)
        return pd.DataFrame(transformed_dataset, columns=dataset.columns)