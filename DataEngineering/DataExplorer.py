from IPython.display import display

class DataExplorer:
    """
    Classe para realizar uma análise inicial de um dataset.
    """
    def analyze(self, dataset):
        """
        Realiza a análise inicial do dataset e imprime os resultados.
        """
        if dataset is None:
            print("Nenhum dataset fornecido para análise.")
            return

        # Exibe as 5 primeiras linhas
        print("Primeiras linhas do dataset:")
        display(dataset.head())

        # Resumo estatístico
        print("\nResumo estatístico do dataset:")
        display(dataset.describe())

        # Informações do dataset
        print("\nInformações sobre os tipos de dados:")
        dataset.info()
