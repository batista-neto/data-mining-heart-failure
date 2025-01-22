from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DataScaler:
   def __init__(self, dataset, target_column, scaler_strategy):
       """
       Inicializa a classe com o dataset e a coluna alvo (target).
       """
       self.dataset = dataset
       self.target_column = target_column
       self.scaler_strategy = scaler_strategy
       self.evaluation_results = {}

   def split_data(self, test_size=0.2, random_state=42):
       """
       Divide os dados em treino e teste.
       """
       X = self.dataset.drop(columns=[self.target_column])
       y = self.dataset[self.target_column]
       return train_test_split(X, y, test_size=test_size, random_state=random_state)


   def scale_data(self, X_train, X_test, method):
       """
       Aplica um metodo de escalonamento nos dados.
       """
       scaler = self.scaler_strategy.scale_strategy(method)
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)

       return X_train_scaled, X_test_scaled
  
   def evaluate_model(self, y_true, y_pred, method):
        """
        Avaliar o modelo e armazenar os resultados.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        self.evaluation_results[method] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(f"\nResultados para {method}:")
        print(f"Acurácia: {accuracy:.2f}")
        print(f"Precisão: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

   def find_best_scaling_method(self):
        """
        Determina o melhor método de escalonamento com base no maior F1-Score.
        """
        

        if not self.evaluation_results:
            print("Nenhum resultado de avaliação disponível. Execute evaluate_model primeiro.")
            return None
        
        weights = {'accuracy': 1, 'precision': 1, 'recall': 1, 'f1_score': 1}
        best_methods = []
        best_score = -1

        for method, metrics in self.evaluation_results.items():
            score = (metrics['accuracy'] * weights['accuracy'] +
                    metrics['precision'] * weights['precision'] +
                    metrics['recall'] * weights['recall'] +
                    metrics['f1_score'] * weights['f1_score'])


            if score > best_score:
                best_score = score
                best_methods = [method] 
            elif score == best_score:
                best_methods.append(method) 
        if len(best_methods) > 1:
            print(f"Empate entre os seguintes métodos com pontuação {best_score:.2f}: {', '.join(best_methods)}")
        else:
            print(f"O melhor método é: {best_methods[0]} com pontuação {best_score:.2f}")

        return best_methods


