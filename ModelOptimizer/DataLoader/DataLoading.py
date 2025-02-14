import pandas as pd

class DataLoader:
    """
    Classe para carregar um dataset a partir de um arquivo CSV.
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Carrega o dataset e retorna como um DataFrame do pandas.
        """
        try:
            dataset = pd.read_csv(self.file_path)
            print("Dataset carregado com sucesso!")
            return dataset
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}")
            return None