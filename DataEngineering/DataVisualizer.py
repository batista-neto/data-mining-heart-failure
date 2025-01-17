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

    def plot_boxplots(self):
        """
        Plota boxplots de todas as colunas numéricas do dataset para identificar possíveis outliers.
        """
        if self.dataset is None:
            print("Nenhum dataset fornecido para visualização.")
            return

        numeric_columns = self.dataset.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.dataset[column], color='orange')
            plt.title(f'Boxplot da Coluna: {column}')
            plt.xlabel(column)
            plt.show()
