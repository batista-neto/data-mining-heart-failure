from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

class HeartFailurePrediction:
    def __init__(self, data, model, target):
        # Assumindo que 'data' já é um DataFrame do pandas
        self.data = data
        self.model = model
        self.target = target
    
    def preprocess_data(self):
        # Separar entre features e target
        X = self.data.drop(self.target, axis=1)  # Substitua 'target' pelo nome correto da coluna alvo
        y = self.data[self.target]  # Substitua 'target' pelo nome correto da coluna alvo
        
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
        
        # Avaliar o modelo
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return accuracy, report
    
    def get_model(self):
        # Retornar o modelo treinado
        return self.model