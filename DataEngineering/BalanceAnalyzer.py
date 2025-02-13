import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, make_scorer


class BalanceAnalyser:
    def __init__(self, balance_strategy, dataset=None, target_column=None):
        self.dataset = dataset
        self.target_column = target_column
        self.balance_strategy = balance_strategy
        self.results = {}


    def balance(self, dataset, target_column, method):
        if dataset is None:
            dataset = self.dataset

        if target_column is None:
            target_column = self.target_column

        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]

        balance = self.balance_strategy.balance_strategy(method)
        X_resampled, y_resampled = balance.fit_resample(X, y)
        balanced_dataset = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name=self.target_column)], axis=1)
        return balanced_dataset

    def evaluate_model(self, y_test, y_pred, method):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Resultados para {method}:")
        print(f"Acurácia: {accuracy}")
        print(f"Precisão: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")

        return self.results[method]

    def cross_val_evaluate(self, dataset, target_column, model, method, cv=5):
        """
        Realiza validação cruzada e avalia o modelo com diferentes métodos de balanceamento.
        """
        if dataset is None:
            dataset = self.dataset

        if target_column is None:
            target_column = self.target_column

        balance_strategy = self.balance_strategy.balance_strategy(method)

        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]

        pipeline = Pipeline([
            ('balance', balance_strategy),
            ('model', model)
        ])

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=1),
            'recall': make_scorer(recall_score, zero_division=1),
            'f1': make_scorer(f1_score, zero_division=1)
        }

        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_validate(pipeline, X, y, cv=kfold, scoring=scoring)
        self.evaluate_cross_val_scores(scores, method)

    def evaluate_cross_val_scores(self, scores, method):
        self.results[method] = {
            "accuracy": scores['test_accuracy'].mean(),
            "precision": scores['test_precision'].mean(),
            "recall": scores['test_recall'].mean(),
            "f1_score": scores['test_f1'].mean()
        }

        print(f"Resultados para {method}:")
        print(f"Acurácia: {scores['test_accuracy'].mean():.2f}")
        print(f"Precisão: {scores['test_precision'].mean():.2f}")
        print(f"Recall: {scores['test_recall'].mean():.2f}")
        print(f"F1-Score: {scores['test_f1'].mean():.2f}")

    def compare_strategies(self):
        """
        Compara as estratégias de balanceamento com base nas métricas de avaliação.
        """
        if not self.results:
            print("Nenhum resultado disponível. Execute as estratégias primeiro.")
            return None

        weights = {'accuracy': 1, 'precision': 1, 'recall': 1, 'f1_score': 1}
        best_methods = []
        best_score = -1

        for method, metrics in self.results.items():
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
    
    def _plot_distribution(self, class_counts, title):
        """
        Plota a distribuição das classes.
        """
        plt.figure(figsize=(8, 5))
        sns.barplot(x=class_counts.index, y=class_counts.values, color="skyblue")
        plt.title(title, fontsize=16)
        plt.xlabel("Classes", fontsize=14)
        plt.ylabel("Quantidade", fontsize=14)
        plt.xticks(ticks=range(len(class_counts.index)), labels=class_counts.index, fontsize=12)
        plt.show()

    def analyze_distribution(self, y_data=None, title="Distribuição da Variável-Alvo"):
        """
        Analisa e exibe a distribuição da variável-alvo ou de um conjunto de dados fornecido.
        """
        if y_data is None:
            # Caso y_data não seja fornecido, utiliza a variável-alvo do dataset
            class_counts = self.dataset[self.target_column].value_counts()
            print("Distribuição da variável-alvo:")
        else:
            # Caso y_data seja fornecido, utiliza esse conjunto de dados
            class_counts = pd.Series(y_data).value_counts()
            print(f"Distribuição fornecida:")
        
        print(class_counts)
        print("\nProporção das classes:")
        print(class_counts / class_counts.sum())

        self._plot_distribution(class_counts, title)
    
    def find_best_balancing_strategy(self, dataset, target_column, balance_types, model):
            print("Iniciando avaliação das estratégias de balanceamento...")
            for method in balance_types:
                method_name = method.value
                self.cross_val_evaluate(dataset, target_column, model, method_name)
            best_methods = self.compare_strategies()
            print("Avaliação concluída. O melhor étodo foi:", best_methods[0])
            dataset_balanced = self.balance(dataset, target_column, best_methods[0])
            return dataset_balanced

