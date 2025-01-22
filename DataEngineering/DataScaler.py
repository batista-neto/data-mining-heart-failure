from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DataScaler:
   def __init__(self, dataset, target_column):
       """
       Inicializa a classe com o dataset e a coluna alvo (target).
       """
       self.dataset = dataset
       self.target_column = target_column
       self.scalers = {
           'minmax': MinMaxScaler(),
           'robustscaler': RobustScaler(),
       }


   def split_data(self, test_size=0.2, random_state=42):
       """
       Divide os dados em treino e teste.
       """
       X = self.dataset.drop(columns=[self.target_column])
       y = self.dataset[self.target_column]
       return train_test_split(X, y, test_size=test_size, random_state=random_state)


   def scale_data(self, X_train, X_test, method):
       """
       Aplica o MinMax ou RobustScaler nos dados.
       """
       if method not in self.scalers:
           raise ValueError(f"Método inválido. Escolha entre 'minmax' ou 'robustscaler'.")


       scaler = self.scalers[method]
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)


       return X_train_scaled, X_test_scaled
  
   def evaluate_model(self, y_true, y_pred, method):
       """
       Avaliar o modelo
       """
       print(f"\nResultados para {method}:")
       print(f"Acurácia: {accuracy_score(y_true, y_pred):.2f}")
       print(f"Precisão: {precision_score(y_true, y_pred):.2f}")
       print(f"Recall: {recall_score(y_true, y_pred):.2f}")
       print(f"F1-Score: {f1_score(y_true, y_pred):.2f}")
