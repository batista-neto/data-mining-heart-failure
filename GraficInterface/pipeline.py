import tkinter as tk
from tkinter import messagebox
import threading
from utils import select_file, run_command_with_output, load_columns

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
