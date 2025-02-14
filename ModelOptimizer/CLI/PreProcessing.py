import argparse
import pandas as pd
from IPython.display import display
import sys
sys.path.append('/Users/develcode118/Documents/UFMA/Mineracao_dados/data-mining-heart-failure/ModelOptimizer')
from ModelOptimizer.DataEngineering.BalanceAnalyzer import BalanceAnalyser
from ModelOptimizer.Strategies.BalanceStrategy import BalanceStrategy
from ModelOptimizer.DataEngineering.OutlierAnalyser import OutlierAnalyzer
from ModelOptimizer.DataEngineering.DataScaler import DataScaler
from ModelOptimizer.Strategies.OutlierRemoverStrategy import OutlierRemoverStrategy
from ModelOptimizer.Strategies.OutlierDetectorStrategy import OutlierDetectorStrategy
from ModelOptimizer.Strategies.ScalerStrategy import ScalerStrategy
from ModelOptimizer.Strategies.ModelStrategy import ModelStrategy
from ModelOptimizer.MachineLearning.BestModelPipeLine import BestModelPipeLine

def balance(dataset=None, target_column=None, method=None, output=None):
    """
    Faz balanceamento do dataset utilizando a classe BalanceAnalyzer.
    
    :param dataset: DataFrame contendo os dados
    :param coluna_alvo: Nome da coluna alvo no dataset
    :param metodo_remocao: Método de balanceamento a ser utilizado
    :return: DataFrame balanceado 
    """

    parser = argparse.ArgumentParser()
    if dataset is None or target_column is None or method is None:
        parser = argparse.ArgumentParser(description="Balanceamento de dados")
        parser.add_argument('--data', type=str, required=True, help='Caminho para o arquivo CSV do dataset')
        parser.add_argument('--target', type=str, required=True, help='Nome da coluna alvo no dataset')
        parser.add_argument('--method', type=str, required=True, help='Método de balanceamento a ser utilizado')
        parser.add_argument('--output', type=str, default='dataset_balanced.csv', help='Nome do arquivo de saída')
        args = parser.parse_args()

        dataset = args.data
        target_column = args.target
        method = args.method
        output = args.output

    dataset = pd.read_csv(dataset)

    print("Inicializando balanceamento...")

    balance_strategy = BalanceStrategy() 

    balance_analyser = BalanceAnalyser(balance_strategy, dataset,target_column)

    balanced_data = balance_analyser.balance(method)

    balanced_data.to_csv(f"{output}.csv", index=False)

    display(balanced_data.head())


def transform(dataset=None, target_column=None, method=None, output=None):
    """
    Aplica a remoção de outliers no dataset utilizando a classe OutlierAnalyzer.
    
    :param dataset: DataFrame contendo os dados
    :param coluna_alvo: Nome da coluna alvo no dataset
    :param metodo_remocao: Método de remoção de outliers a ser utilizado
    :return: DataFrame transformado após remoção de outliers
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, required=True, help='Caminho para o arquivo CSV do dataset')
    parser.add_argument('--target', type=str, required=True, help='Nome da coluna alvo no dataset')
    parser.add_argument('--method', type=str, required=True, help='Método de transformacao a ser utilizado')
    parser.add_argument('--output', type=str, default='dataset_transformed.csv', help='Nome do arquivo de saída')

    args = parser.parse_args()

    if dataset is None:
        dataset = args.data
    if target_column is None:
        target_column = args.target
    if method is None:
        method = args.method
    if output is None:
        output = args.output

    dataset = pd.read_csv(dataset)

    print("Inicializando transformação...")

    outlier_remover_strategy = OutlierRemoverStrategy()
    outlier_detector_strategy = OutlierDetectorStrategy()

    analisador = OutlierAnalyzer(outlier_detector_strategy, outlier_remover_strategy, dataset, target_column)
    dataset_transformado = analisador.transform(method)

    dataset_transformado.to_csv(f"{output}.csv", index=False)  

    display(dataset_transformado.head())

def scale(dataset=None, target_column=None, method=None, output=None):
    """
    Aplica um método de escalonamento nos dados e retorna o dataset inteiro escalado.

    :param dataset: DataFrame contendo os dados
    :param coluna_alvo: Nome da coluna alvo no dataset
    :param metodo_remocao: Método de escalonamento a ser utilizado
    :return: DataFrame escalonado
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, required=True, help='Caminho para o arquivo CSV do dataset')
    parser.add_argument('--target', type=str, required=True, help='Nome da coluna alvo no dataset')
    parser.add_argument('--method', type=str, required=True, help='Método de escalonamento a ser utilizado')
    parser.add_argument('--output', type=str, default='dataset_scaled.csv', help='Nome do arquivo de saída')


    args = parser.parse_args()

    if dataset is None:
        dataset = args.data
    if target_column is None:
        target_column = args.target
    if method is None:
        method = args.method
    if output is None:
        output = args.output

    dataset = pd.read_csv(dataset)

    print("Inicializando escalonamento...")

    scaler_strategy = ScalerStrategy()
    data_scaler = DataScaler(scaler_strategy)

    dataset_scaled = data_scaler.scale_data(method, dataset, target_column)

    dataset_scaled.to_csv(f"{output}.csv", index=False)  

    display(dataset_scaled.head())

def pipeline(dataset=None, target_column=None):
    """
    Executa o pipeline que retorna a combinção de modelo, balanceamento, trasformação e escalamento que melhor pontuou.
    :param dataset: DataFrame contendo os dados
    :param coluna_alvo: Nome da coluna alvo no dataset
    :return: Modelo melhor pontuado
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, required=True, help='Caminho para o arquivo CSV do dataset')
    parser.add_argument('--target', type=str, required=True, help='Nome da coluna alvo no dataset')

    args = parser.parse_args()

    if dataset is None:
        dataset = args.data
    if target_column is None:
        target_column = args.target

    dataset = pd.read_csv(dataset)

    balance_strategy = BalanceStrategy()
    balance_analyser = BalanceAnalyser(balance_strategy)
    model_strategy = ModelStrategy()

    outlier_detector_strategy = OutlierDetectorStrategy()
    outlier_remover_strategy = OutlierRemoverStrategy()
    outlier_analyzer = OutlierAnalyzer(outlier_detector_strategy, outlier_remover_strategy)

    scaler_strategy = ScalerStrategy()
    data_scaler = DataScaler(scaler_strategy)

    best_model = BestModelPipeLine(dataset, target_column, balance_analyser, data_scaler, outlier_analyzer, model_strategy)

    best_model.run_pipeline()

def treatment(dataset=None, target_column=None, balance=None, transform=None, scale=None, output="output.csv"):
    """
    Faz balanceamento, tratamento de outlier e escalonamento, respectivamente, do dataset.
    
    :param dataset: DataFrame contendo os dados
    :param coluna_alvo: Nome da coluna alvo no dataset
    :param balace: Método de balanceamento a ser utilizado
    :param transform: Método de transformacao a ser utilizado
    :param scale: Método de escalonamento a ser utilizado
    :return: DataFrame tratado 
    """

    parser = argparse.ArgumentParser()
    if dataset is None or target_column is None or balance is None or transform is None or scale is None:
        parser.add_argument('--data', type=str, required=True, help='Caminho para o arquivo CSV do dataset')
        parser.add_argument('--target', type=str, required=True, help='Nome da coluna alvo no dataset')
        parser.add_argument('--balance', type=str, required=True, help='Método de balanceamento a ser utilizado')
        parser.add_argument('--transform', type=str, required=True, help='Método de transformacao a ser utilizado')
        parser.add_argument('--scale', type=str, required=True, help='Método de escalonamento a ser utilizado')
        parser.add_argument('--output', type=str, default='dataset_treatment.csv', help='Nome do arquivo de saída')
        
        args = parser.parse_args()
        dataset = args.data
        target_column = args.target
        balance = args.balance
        transform = args.transform
        scale = args.scale
        output = args.output

    dataset = pd.read_csv(dataset)

    print("Inicializando balanceamento...")
    balance_strategy = BalanceStrategy() 
    balance_analyser = BalanceAnalyser(balance_strategy, dataset, target_column)
    balanced_data = balance_analyser.balance(balance)

    print("Inicializando transformação...")
    outlier_remover_strategy = OutlierRemoverStrategy()
    outlier_detector_strategy = OutlierDetectorStrategy()
    analisador = OutlierAnalyzer(outlier_detector_strategy, outlier_remover_strategy, balanced_data, target_column)
    dataset_transformado = analisador.transform(transform)

    print("Inicializando escalonamento...")
    scaler_strategy = ScalerStrategy()
    data_scaler = DataScaler(scaler_strategy)
    dataset_scaled = data_scaler.scale_data(scale, dataset_transformado, target_column)

    dataset_scaled.to_csv(f"{output}.csv", index=False)

    display(dataset_scaled.head())
    return dataset_scaled

if __name__ == "__main__":
    import sys
    if 'balance' in sys.argv[0]:
        balance()
    elif 'transform' in sys.argv[0]:
        transform()
    elif 'scale' in sys.argv[0]:
        scale()
    elif 'pipeline' in sys.argv[0]:
        pipeline()
    elif 'treatment' in sys.argv[0]:
        treatment()
    else:
        print("Comando não reconhecido")