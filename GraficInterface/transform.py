import tkinter as tk
from tkinter import messagebox
import threading
from utils import select_file, run_command_with_output, load_columns

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
