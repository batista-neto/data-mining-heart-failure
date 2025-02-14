import tkinter as tk
from tkinter import messagebox
import threading
from utils import select_file, run_command_with_output, load_columns

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
