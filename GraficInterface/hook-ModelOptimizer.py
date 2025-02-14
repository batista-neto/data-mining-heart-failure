from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Coletar submódulos e arquivos de dados do ModelOptimizer
hiddenimports = collect_submodules('ModelOptimizer')
datas = collect_data_files('ModelOptimizer')

# Incluir os submódulos e arquivos de outras bibliotecas essenciais
libraries = [
    "imblearn",
    "sklearn",
    "matplotlib",
    "numpy",
    "pandas",
    "seaborn",
    "scipy",
    "xgboost",
    "joblib",
    "ipykernel",
    "ipython"
]

for lib in libraries:
    hiddenimports += collect_submodules(lib)
    datas += collect_data_files(lib)

# Incluir arquivos do matplotlib
datas += collect_data_files("matplotlib.backends")
hiddenimports += collect_submodules("matplotlib.backends")

# Incluir arquivos do scikit-learn
datas += collect_data_files("sklearn.utils._typedefs")
hiddenimports += collect_submodules("sklearn.utils._typedefs")

# Resolver possível erro do imblearn (VERSION.txt ausente)
datas += collect_data_files("imblearn", includes=["VERSION.txt"])

