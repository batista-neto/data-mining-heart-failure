import seaborn as sns
import matplotlib.pyplot as plt

class FeatureSelector:
    def __init__(self, dataset):
        """
        Classe para realizar a seleção de atributos.
        """
        self.dataset = dataset

    def analyze_correlation(self, threshold=0.8):
            """
            Analisa a correlação entre os atributos e retorna pares de atributos com alta correlação.
            """
            # Calcular a matriz de correlação
            correlation_matrix = self.dataset.corr()

            # Visualizar a matriz de correlação
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Matriz de Correlação", fontsize=16)
            plt.show()

            # Encontrar pares de atributos com alta correlação
            high_correlation_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > threshold:
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        correlation_value = correlation_matrix.iloc[i, j]
                        high_correlation_pairs.append((col1, col2, correlation_value))

            # Retornar os pares de alta correlação
            if high_correlation_pairs:
                print("Atributos com alta correlação (acima de 0.8):")
                for pair in high_correlation_pairs:
                    print(f"{pair[0]} e {pair[1]} - Correlação: {pair[2]:.2f}")
            else:
                print("Nenhum par de atributos com correlação alta encontrado.")

            return high_correlation_pairs
    
    def drop_columns(self, columns_to_drop):
        """
        Remove as colunas especificadas do dataset original.
        """
        print(f"Removendo as colunas: {columns_to_drop}")
        self.dataset = self.dataset.drop(columns=columns_to_drop, axis=1)

        return self.dataset


     

