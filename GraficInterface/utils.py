import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
import pandas as pd

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
