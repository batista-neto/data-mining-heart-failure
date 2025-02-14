from setuptools import setup, find_packages

setup(
    name='ModelOptimizer',  # Nome do seu pacote
    version='0.1',  # Versão do seu pacote
    packages=find_packages(),  # Isso encontra todas as pastas com um __init__.py
    install_requires=[  # Dependências do seu pacote
        'appnope==0.1.4',
        'imbalanced-learn==0.13.0',
        'ipykernel==6.29.5',
        'ipython==8.31.0',
        'joblib==1.4.2',
        'matplotlib==3.10.0',
        'numpy==2.2.1',
        'pandas==2.2.3',
        'scikit-learn==1.6.1',
        'seaborn==0.13.2',
        'scipy==1.15.1',
        'xgboost==2.1.4',
    ],
    entry_points={ 
        'console_scripts': [
            'balance=ModelOptimizer.CLI.PreProcessing:balance',
            'transform=ModelOptimizer.CLI.PreProcessing:transform',
            'scale=ModelOptimizer.CLI.PreProcessing:scale',
            'pipeline=ModelOptimizer.CLI.PreProcessing:pipeline',
            'treatment=ModelOptimizer.CLI.PreProcessing:treatment',
        ],
    },
    author='Joao_Batista&Bruno',
    author_email='jbss.neto@discente.ufma.br',
    description='',
    long_description=open('README.md').read(),
    url='https://github.com/batista-neto/data-mining-heart-failure', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Versão mínima do Python
)