import os
import sys
import tkinter as tk
from tkinter import ttk
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
import pandas as pd

# Suprimir warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class SuppressWarnings:
    def write(self, msg):
        pass 

sys.stderr = SuppressWarnings()

def balance(frame):
    """Cria a interface para a execução do balanceamento dentro de um frame."""

    balance_file_label = tk.Label(frame, text="Selecione o arquivo CSV:")
    balance_file_label.pack(pady=5)

    balance_file_entry = tk.Entry(frame, width=50)
    balance_file_entry.pack(pady=5)

    balance_file_button = tk.Button(frame, text="Selecionar Arquivo", command=lambda: select_file(balance_file_entry, lambda fp: load_columns(fp, column_var_balance, column_menu_balance)))
    balance_file_button.pack(pady=5)

    column_var_balance = tk.StringVar(frame)
    column_menu_balance = tk.OptionMenu(frame, column_var_balance, "")
    column_label_balance = tk.Label(frame, text="Escolha a Coluna Alvo:")
    column_label_balance.pack()
    column_menu_balance.pack(pady=5)

    balance_methods = ["oversampling", "undersampling", "smotetomek"]
    balance_var = tk.StringVar(frame)
    balance_var.set(balance_methods[0])
    balance_label = tk.Label(frame, text="Escolha o Método de Balanceamento:")
    balance_label.pack()
    balance_menu = tk.OptionMenu(frame, balance_var, *balance_methods)
    balance_menu.pack(pady=5)

    output_label = tk.Label(frame, text="Nome do Arquivo de Saída:")
    output_label.pack()
    output_entry = tk.Entry(frame, width=50)
    output_entry.pack(pady=5)

    output_text_balance = tk.Text(frame, height=10, width=60)
    output_text_balance.pack(pady=5)
    output_text_balance.tag_configure("error", foreground="red")

    def run_balance():
        """Executa o balanceamento e mantém a mensagem 'EXECUTANDO...' até a finalização."""
        file_path = balance_file_entry.get()
        target_column = column_var_balance.get()
        balance_method = balance_var.get()
        output_file = output_entry.get()

        if not file_path or not target_column or not output_file:
            messagebox.showerror("Erro", "Por favor, preencha todos os campos antes de executar.")
            return

        output_text_balance.delete("1.0", tk.END)
        output_text_balance.see(tk.END)
        frame.update_idletasks()

        def execute():
            run_command_with_output(["balance", "--data", file_path, "--target", target_column, "--method", balance_method, "--output", output_file], output_text_balance)
            output_text_balance.insert(tk.END, "EXECUTANDO...\n")

        threading.Thread(target=execute, daemon=True).start()

    balance_button = tk.Button(frame, text="Executar Balanceamento", command=run_balance)
    balance_button.pack(pady=10)


def pipeline(frame):
    """Cria a interface para execução do pipeline dentro de um frame."""
    
    file_label = tk.Label(frame, text="Selecione o arquivo CSV:")
    file_label.pack(pady=5)

    file_entry = tk.Entry(frame, width=50)
    file_entry.pack(pady=5)

    file_button = tk.Button(frame, text="Selecionar Arquivo", command=lambda: select_file(file_entry, lambda fp: load_columns(fp, column_var, column_menu)))
    file_button.pack(pady=5)

    column_var = tk.StringVar(frame)
    column_menu = tk.OptionMenu(frame, column_var, "")
    column_label = tk.Label(frame, text="Escolha a Coluna Alvo:")
    column_label.pack()
    column_menu.pack(pady=5)

    output_text = tk.Text(frame, height=10, width=60)
    output_text.pack(pady=5)
    output_text.tag_configure("error", foreground="red")

    def run_pipeline():
        """Executa o pipeline e mantém a mensagem 'EXECUTANDO...' até a finalização."""
        file_path = file_entry.get()
        target_column = column_var.get()

        if not file_path or not target_column:
            messagebox.showerror("Erro", "Por favor, selecione um arquivo e escolha a coluna alvo.")
            return

        output_text.delete("1.0", tk.END)
        output_text.see(tk.END)
        frame.update_idletasks()

        def execute():
            run_command_with_output(["pipeline", "--data", file_path, "--target", target_column], output_text)
            output_text.insert(tk.END, "EXECUTANDO...\n") 

        threading.Thread(target=execute, daemon=True).start()

    run_button = tk.Button(frame, text="Executar Pipeline", command=run_pipeline)
    run_button.pack(pady=10)

def scale(frame):
    """Cria a interface para a execução do escalonamento dentro de um frame."""

    scaling_file_label = tk.Label(frame, text="Selecione o arquivo CSV:")
    scaling_file_label.pack(pady=5)

    scaling_file_entry = tk.Entry(frame, width=50)
    scaling_file_entry.pack(pady=5)

    scaling_file_button = tk.Button(frame, text="Selecionar Arquivo", command=lambda: select_file(scaling_file_entry, lambda fp: load_columns(fp, column_var_scaling, column_menu_scaling)))
    scaling_file_button.pack(pady=5)

    column_var_scaling = tk.StringVar(frame)
    column_menu_scaling = tk.OptionMenu(frame, column_var_scaling, "")
    column_label_scaling = tk.Label(frame, text="Escolha a Coluna Alvo:")
    column_label_scaling.pack()
    column_menu_scaling.pack(pady=5)

    scaling_methods = ["minmax", "standard", "robust", "maxabs", "normalizer"]
    scaling_var = tk.StringVar(frame)
    scaling_var.set(scaling_methods[0])
    scaling_label = tk.Label(frame, text="Escolha o Método de Escalonamento:")
    scaling_label.pack()
    scaling_menu = tk.OptionMenu(frame, scaling_var, *scaling_methods)
    scaling_menu.pack(pady=5)

    output_label = tk.Label(frame, text="Nome do Arquivo de Saída:")
    output_label.pack()
    output_entry = tk.Entry(frame, width=50)
    output_entry.pack(pady=5)

    output_text_scaling = tk.Text(frame, height=10, width=60)
    output_text_scaling.pack(pady=5)
    output_text_scaling.tag_configure("error", foreground="red")

    def run_scaling():
        """Executa o escalonamento e mantém a mensagem 'EXECUTANDO...' até a finalização."""
        file_path = scaling_file_entry.get()
        target_column = column_var_scaling.get()
        scaling_method = scaling_var.get()
        output_file = output_entry.get()

        if not file_path or not target_column or not output_file:
            messagebox.showerror("Erro", "Por favor, preencha todos os campos antes de executar.")
            return

        output_text_scaling.delete("1.0", tk.END)
        output_text_scaling.see(tk.END)
        frame.update_idletasks()

        def execute():
            run_command_with_output(["scale", "--data", file_path, "--target", target_column, "--method", scaling_method, "--output", output_file], output_text_scaling)
            output_text_scaling.insert(tk.END, "EXECUTANDO...\n") 

        threading.Thread(target=execute, daemon=True).start()

    scale_button = tk.Button(frame, text="Executar Escalonamento", command=run_scaling)
    scale_button.pack(pady=10)

def transform(frame):
    """Cria a interface para a execução da transformação dentro de um frame."""

    transform_file_label = tk.Label(frame, text="Selecione o arquivo CSV:")
    transform_file_label.pack(pady=5)

    transform_file_entry = tk.Entry(frame, width=50)
    transform_file_entry.pack(pady=5)

    transform_file_button = tk.Button(frame, text="Selecionar Arquivo", command=lambda: select_file(transform_file_entry, lambda fp: load_columns(fp, column_var_transform, column_menu_transform)))
    transform_file_button.pack(pady=5)

    column_var_transform = tk.StringVar(frame)
    column_menu_transform = tk.OptionMenu(frame, column_var_transform, "")
    column_label_transform = tk.Label(frame, text="Escolha a Coluna Alvo:")
    column_label_transform.pack()
    column_menu_transform.pack(pady=5)

    transform_methods = ["logaritimo", "raiz_quadrada", "yeojohnson"]
    transform_var = tk.StringVar(frame)
    transform_var.set(transform_methods[0])
    transform_label = tk.Label(frame, text="Escolha o Método de Transformação:")
    transform_label.pack()
    transform_menu = tk.OptionMenu(frame, transform_var, *transform_methods)
    transform_menu.pack(pady=5)

    output_label = tk.Label(frame, text="Nome do Arquivo de Saída:")
    output_label.pack()
    output_entry = tk.Entry(frame, width=50)
    output_entry.pack(pady=5)

    output_text_transform = tk.Text(frame, height=10, width=60)
    output_text_transform.pack(pady=5)
    output_text_transform.tag_configure("error", foreground="red")

    def run_transformation():
        """Executa a transformação e mantém a mensagem 'EXECUTANDO...' até a finalização."""
        file_path = transform_file_entry.get()
        target_column = column_var_transform.get()
        transform_method = transform_var.get()
        output_file = output_entry.get()

        if not file_path or not target_column or not output_file:
            messagebox.showerror("Erro", "Por favor, preencha todos os campos antes de executar.")
            return

        output_text_transform.delete("1.0", tk.END)
        output_text_transform.see(tk.END)
        frame.update_idletasks()

        def execute():
            run_command_with_output(["transform", "--data", file_path, "--target", target_column, "--method", transform_method, "--output", output_file], output_text_transform)
            output_text_transform.insert(tk.END, "EXECUTANDO...\n") 

        threading.Thread(target=execute, daemon=True).start()

    transform_button = tk.Button(frame, text="Executar Transformação", command=run_transformation)
    transform_button.pack(pady=10)

import tkinter as tk
from tkinter import messagebox
import threading
from utils import select_file, run_command_with_output, load_columns

def treatment(frame):
    """Cria a interface para a execução do tratamento dentro de um frame."""

    treatment_file_label = tk.Label(frame, text="Selecione o arquivo CSV:")
    treatment_file_label.pack(pady=5)

    treatment_file_entry = tk.Entry(frame, width=50)
    treatment_file_entry.pack(pady=5)

    treatment_file_button = tk.Button(frame, text="Selecionar Arquivo", command=lambda: select_file(treatment_file_entry, lambda fp: load_columns(fp, column_var_treatment, column_menu_treatment)))
    treatment_file_button.pack(pady=5)

    column_var_treatment = tk.StringVar(frame)
    column_menu_treatment = tk.OptionMenu(frame, column_var_treatment, "")
    column_label_treatment = tk.Label(frame, text="Escolha a Coluna Alvo:")
    column_label_treatment.pack()
    column_menu_treatment.pack(pady=5)

    balance_methods = ["oversampling", "undersampling", "smotetomek"]
    balance_var = tk.StringVar(frame)
    balance_var.set(balance_methods[0])
    balance_label = tk.Label(frame, text="Escolha o Método de Balanceamento:")
    balance_label.pack()
    balance_menu = tk.OptionMenu(frame, balance_var, *balance_methods)
    balance_menu.pack(pady=5)

    transform_methods = ["logaritimo", "raiz_quadrada", "yeojohnson"]
    transform_var = tk.StringVar(frame)
    transform_var.set(transform_methods[0])
    transform_label = tk.Label(frame, text="Escolha o Método de Transformação:")
    transform_label.pack()
    transform_menu = tk.OptionMenu(frame, transform_var, *transform_methods)
    transform_menu.pack(pady=5)

    scale_methods = ["minmax", "standard", "robust", "maxabs", "normalizer"]
    scale_var = tk.StringVar(frame)
    scale_var.set(scale_methods[0])
    scale_label = tk.Label(frame, text="Escolha o Método de Escalonamento:")
    scale_label.pack()
    scale_menu = tk.OptionMenu(frame, scale_var, *scale_methods)
    scale_menu.pack(pady=5)

    output_label = tk.Label(frame, text="Nome do Arquivo de Saída:")
    output_label.pack()
    output_entry = tk.Entry(frame, width=50)
    output_entry.pack(pady=5)

    output_text_treatment = tk.Text(frame, height=10, width=60)
    output_text_treatment.pack(pady=5)
    output_text_treatment.tag_configure("error", foreground="red")

    def run_treatment():
        """Executa o tratamento e mantém a mensagem 'EXECUTANDO...' até a finalização."""
        file_path = treatment_file_entry.get()
        target_column = column_var_treatment.get()
        balance_method = balance_var.get()
        transform_method = transform_var.get()
        scale_method = scale_var.get()
        output_file = output_entry.get()

        if not file_path or not target_column or not output_file:
            messagebox.showerror("Erro", "Por favor, preencha todos os campos antes de executar.")
            return

        output_text_treatment.delete("1.0", tk.END)
        output_text_treatment.see(tk.END)
        frame.update_idletasks()

        def execute():
            run_command_with_output([
                "treatment", "--data", file_path, "--target", target_column, 
                "--balance", balance_method, "--transform", transform_method, 
                "--scale", scale_method, "--output", output_file
            ], output_text_treatment)
            output_text_treatment.insert(tk.END, "EXECUTANDO...\n")  # Mensagem ao finalizar

        threading.Thread(target=execute, daemon=True).start()

    treatment_button = tk.Button(frame, text="Executar Tratamento", command=run_treatment)
    treatment_button.pack(pady=10)

def select_file(entry_widget, load_columns_func=None):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)
        if load_columns_func:
            load_columns_func(file_path)

def load_columns(file_path, column_var, column_menu):
    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        if columns:
            column_var.set(columns[0])
            column_menu["menu"].delete(0, "end")
            for col in columns:
                column_menu["menu"].add_command(label=col, command=tk._setit(column_var, col))
        else:
            messagebox.showerror("Erro", "O arquivo CSV não contém colunas.")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao ler o CSV: {str(e)}")

def run_command_with_output(command, output_widget):
    def execute():
        try:
            output_widget.delete("1.0", tk.END)  # Limpa o terminal anterior
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Lendo a saída do processo em tempo real
            for line in process.stdout:
                output_widget.insert(tk.END, line)
                output_widget.see(tk.END)  # Rolagem automática

            for line in process.stderr:
                output_widget.insert(tk.END, line, "error")
                output_widget.see(tk.END)

            process.wait()

            if process.returncode == 0:
                messagebox.showinfo("Sucesso", "Processo concluído com sucesso!")
            else:
                messagebox.showerror("Erro", "Erro ao executar o comando. Verifique a saída.")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao executar o comando: {str(e)}")

    threading.Thread(target=execute, daemon=True).start()

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