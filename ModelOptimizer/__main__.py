import argparse
from ModelOptimizer.CLI.PreProcessing import balance, transform, scale, pipeline, treatment

def main():
    parser = argparse.ArgumentParser(description="ModelOptimizer CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Subcomando: balance
    parser_balance = subparsers.add_parser("balance", help="Realiza balanceamento do dataset")
    parser_balance.add_argument("--data", type=str, required=True, help="Caminho do dataset CSV")
    parser_balance.add_argument("--target", type=str, required=True, help="Coluna alvo")
    parser_balance.add_argument("--method", type=str, required=True, help="Método de balanceamento")
    parser_balance.add_argument("--output", type=str, default="dataset_balanced.csv", help="Nome do arquivo de saída")

    # Subcomando: transform
    parser_transform = subparsers.add_parser("transform", help="Remove outliers do dataset")
    parser_transform.add_argument("--data", type=str, required=True, help="Caminho do dataset CSV")
    parser_transform.add_argument("--target", type=str, required=True, help="Coluna alvo")
    parser_transform.add_argument("--method", type=str, required=True, help="Método de transformação")
    parser_transform.add_argument("--output", type=str, default="dataset_transformed.csv", help="Nome do arquivo de saída")

    # Subcomando: scale
    parser_scale = subparsers.add_parser("scale", help="Escalona o dataset")
    parser_scale.add_argument("--data", type=str, required=True, help="Caminho do dataset CSV")
    parser_scale.add_argument("--target", type=str, required=True, help="Coluna alvo")
    parser_scale.add_argument("--method", type=str, required=True, help="Método de escalonamento")
    parser_scale.add_argument("--output", type=str, default="dataset_scaled.csv", help="Nome do arquivo de saída")

    # Subcomando: pipeline
    parser_pipeline = subparsers.add_parser("pipeline", help="Executa pipeline completo")
    parser_pipeline.add_argument("--data", type=str, required=True, help="Caminho do dataset CSV")
    parser_pipeline.add_argument("--target", type=str, required=True, help="Coluna alvo")

    # Subcomando: treatment
    parser_treatment = subparsers.add_parser("treatment", help="Executa tratamento completo do dataset")
    parser_treatment.add_argument("--data", type=str, required=True, help="Caminho do dataset CSV")
    parser_treatment.add_argument("--target", type=str, required=True, help="Coluna alvo")
    parser_treatment.add_argument("--balance", type=str, required=True, help="Método de balanceamento")
    parser_treatment.add_argument("--transform", type=str, required=True, help="Método de transformação")
    parser_treatment.add_argument("--scale", type=str, required=True, help="Método de escalonamento")
    parser_treatment.add_argument("--output", type=str, default="dataset_treatment.csv", help="Nome do arquivo de saída")

    args = parser.parse_args()

    if args.command == "balance":
        balance()
    elif args.command == "transform":
        transform()
    elif args.command == "scale":
        scale()
    elif args.command == "pipeline":
        pipeline()
    elif args.command == "treatment":
        treatment()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
