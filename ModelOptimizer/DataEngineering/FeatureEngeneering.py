import pandas as pd

class FeatureEngineer:
    def __init__(self, dataset):
        """
        Inicializa a classe de engenharia de features.
        """
        self.dataset = dataset

    def add_interaction_features(self, coluns):
        """
        Adiciona features de interação (multiplicativas) ao dataset.
        """
        print("Adicionando features de interação...")
        self.dataset[f'{coluns[0]}_x_{coluns[1]}'] = self.dataset[coluns[0]] * self.dataset[coluns[1]]
        print("Features de interação adicionadas: ['age_x_ejection_fraction', 'serum_creatinine_x_serum_sodium']")

    def add_ratio_features(self):
        """
        Adiciona features de razão ao dataset.
        """
        print("Adicionando features de razão...")
        self.dataset['serum_creatinine_div_serum_sodium'] = self.dataset['serum_creatinine'] / self.dataset['serum_sodium']
        print("Feature de razão adicionada: ['serum_creatinine_div_serum_sodium']")

    def get_dataset(self):
        """
        Retorna o dataset atualizado.

        Returns:
            pd.DataFrame: Dataset com as novas features.
        """
        return self.dataset
