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
