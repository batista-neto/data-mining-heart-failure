class DataExplorer:
    """
    Classe para realizar uma análise inicial de um dataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def analyze(self):
        """
        Realiza a análise inicial do dataset e imprime os resultados.
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para análise.")
            return

        # Exibe as 5 primeiras linhas
        print("Primeiras linhas do dataset:")
        print(self.dataset.head())

        # Resumo estatístico
        print("\nResumo estatístico do dataset:")
        print(self.dataset.describe())

        # Informações do dataset
        print("\nInformações sobre os tipos de dados:")
        print(self.dataset.info())
