import tkinter as tk
from tkinter import messagebox
import threading
from utils import select_file, run_command_with_output, load_columns

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
