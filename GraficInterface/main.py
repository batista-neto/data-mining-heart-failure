import os
import sys
import tkinter as tk
from tkinter import ttk
import warnings
from pipeline import pipeline
from scale import scale
from transform import transform
from balance import balance
from treatment import treatment
import sys
import os


sys.path.append(os.path.abspath("/Users/develcode118/Documents/UFMA/Mineracao_dados/data-mining-heart-failure"))

import ModelOptimizer  

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class SuppressWarnings:
    def write(self, msg):
        pass 

sys.stderr = SuppressWarnings()

def main():
    root = tk.Tk()
    root.title("Model Optimizer")

    notebook = ttk.Notebook(root)  # Criando abas

    # Criar e adicionar as abas
    frame_balance = ttk.Frame(notebook)
    balance(frame_balance)
    notebook.add(frame_balance, text="Balanceamento")

    frame_transform = ttk.Frame(notebook)
    transform(frame_transform)
    notebook.add(frame_transform, text="Trasformação")

    frame_scaling = ttk.Frame(notebook)
    scale(frame_scaling)
    notebook.add(frame_scaling, text="Escalonamento")

    frame_pipeline = ttk.Frame(notebook)
    pipeline(frame_pipeline)
    notebook.add(frame_pipeline, text="Pipeline")

    frame_treatment = ttk.Frame(notebook)
    treatment(frame_treatment)
    notebook.add(frame_treatment, text="Tratamento")

    notebook.pack(expand=True, fill="both")
    
    root.mainloop()

if __name__ == "__main__":
    main()
