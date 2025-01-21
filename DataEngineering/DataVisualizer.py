import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    """
    Classe para análise visual de distribuições e identificação de outliers.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def plot_distributions(self):
        """
        Plota as distribuições de todas as colunas numéricas do dataset.
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para visualização.")
            return

        numeric_columns = self.dataset.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.dataset[column], kde=True, bins=30, color='blue')
            plt.title(f'Distribuição da Coluna: {column}')
            plt.xlabel(column)
            plt.ylabel('Frequência')
            plt.show()

    def plot_boxplots(self, columns):
        """
        Plota boxplots para as colunas especificadas no dataset.

        :param columns: list, lista de nomes de colunas para as quais os boxplots serão gerados
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para visualização.")
            return

        for column in columns:
            if column not in self.dataset.columns:
                print(f"A coluna '{column}' não existe no dataset.")
                continue
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.dataset[column], color='orange')
            plt.title(f'Boxplot da Coluna: {column}')
            plt.xlabel(column)
            plt.show()
