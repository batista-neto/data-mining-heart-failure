import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix


class BalanceAnalyser:
    def __init__(self, dataset, target_column, balance_strategy):
        self.dataset = dataset
        self.target_column = target_column
        self.balance_strategy = balance_strategy
        self.results = {}

    def balance(self, X_train, y_train, method):
        smote = self.balance_strategy.balance_strategy(method)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    def evaluate_model(self, y_test, y_pred, method):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        self.results[method] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return self.results[method]

    def compare_strategies(self):
        if not self.results:
            print("Nenhum resultado disponível. Execute as estratégias primeiro.")
            return None

        comparison = pd.DataFrame(self.results).T
        print("Resultados comparativos:")
        print(comparison)

        best_strategy = comparison['f1_score'].idxmax()
        print(f"Melhor estratégia: {best_strategy} com F1-Score {comparison['f1_score'].max():.2f}")

        return best_strategy
    
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

    def evaluate_model_with_advanced_metrics(self, model, X_train, y_train, X_test, y_test, method_name):
        """
        Treina e avalia o modelo em métricas avançadas: ROC-AUC, PR-AUC, e Matriz de Confusão.
        """
        # Treinar o modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        # PR-AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba) if y_proba is not None else (None, None, None)
        pr_auc = auc(recall_vals, precision_vals) if y_proba is not None else "N/A"

        # Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)

        # Exibição de resultados
        print(f"\n[{method_name}] - {model.__class__.__name__}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"ROC-AUC: {roc_auc if roc_auc != 'N/A' else 'N/A'}")
        print(f"PR-AUC: {pr_auc if pr_auc != 'N/A' else 'N/A'}")
        print("Confusion Matrix:")
        print(cm)

        # Visualizar a Matriz de Confusão
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Não", "Sim"], yticklabels=["Não", "Sim"])
        plt.title(f"Matriz de Confusão ({method_name} - {model.__class__.__name__})")
        plt.xlabel("Previsto")
        plt.ylabel("Verdadeiro")
        plt.show()

        return {
            "method": method_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }
    
    def compare_with_advanced_metrics(self, model, X_train, y_train, X_test, y_test, methods):
        """
        Compara as estratégias de balanceamento usando métricas avançadas.
        """
        results = []

        for method in methods:
            print(f"\n--- Avaliando com {method} ---")
            X_train_balanced, y_train_balanced = self.balance(X_train, y_train, method)
            result = self.evaluate_model_with_advanced_metrics(model, X_train_balanced, y_train_balanced, X_test, y_test, method)
            results.append(result)

        print("\nResultados comparados:")
        for res in results:
            print(f"{res['method']} - ROC-AUC: {res['roc_auc']}, PR-AUC: {res['pr_auc']}, F1-Score: {res['f1_score']}")

        return results
