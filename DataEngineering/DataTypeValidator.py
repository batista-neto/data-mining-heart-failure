class DataTypeValidator:
    """
    Classe para verificar e ajustar os tipos de dados de um dataset.
    """
    def __init__(self, dataset):
        """
        Inicializa a classe com um dataset.
        
        :param dataset: pandas.DataFrame, dataset para validação de tipos
        """
        self.dataset = dataset

    def validate_types(self):
        """
        Verifica os tipos de dados e sugere ajustes caso necessário.
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para validação.")
            return

        # Exibe os tipos de dados
        print("\nTipos de dados antes da validação:")
        print(self.dataset.dtypes)

        # Sugere ajustes se necessário (exemplo básico)
        for column in self.dataset.select_dtypes(include=['object']).columns:
            print(f"A coluna '{column}' é do tipo 'object'. Verifique se a conversão é necessária.")

        print("\nValidação concluída.")
