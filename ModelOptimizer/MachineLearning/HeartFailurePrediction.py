from readline import redisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
from IPython.display import display

class HeartFailurePrediction:
    def __init__(self, data, model, target):
        # Assumindo que 'data' já é um DataFrame do pandas
        self.data = data
        self.model = model
        self.target = target
    
    def preprocess_data(self):
        # Separar entre features e target
        X = self.data.drop(self.target, axis=1) 
        y = self.data[self.target]
        
        # Dividir em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalizar os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        # Treinar o modelo
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        # Fazer previsões
        predictions = self.model.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        # Exibir resultados
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        total_score = accuracy + precision + recall + f1
        print(f"Pontuação: {total_score:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nMatriz de Confusão:")
        print(conf_matrix)
        print("\nRelatório de Classificação:")
        print(report)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix,
            "classification_report": report
        }
    
    def get_model(self):
        # Retornar o modelo treinado
        return self.model