import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

root = tk.Tk()
root.geometry("1920x1080")
notebook = ttk.Notebook(root)
tab1 = tk.Frame(notebook,width=1850,height=980)
tab2 = tk.Frame(notebook,width=1850,height=980)
notebook.add(tab1,text="Graphs")
notebook.add(tab2, text="Fractals")
notebook.pack(expand=True, fill="both", padx=20,pady=20)
root.mainloop()