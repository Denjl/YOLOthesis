import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import subprocess
import threading

# Spúšťa detekciu kontaminácie pomocou vybraného modelu a videa
def run_predict():
    model_name = model_var.get()  # Získanie názvu vybraného modelu
    video_path = video_var.get()  # Získanie cesty k vybranému videu

    # Kontrola, či boli vybrané oba potrebné vstupy
    if not model_name or not video_path:
        messagebox.showerror("Error", "Please select both a model and a video file.")
        return

     # Aktualizácia stavu a spustenie indikátora priebehu
    status_var.set("Running...")
    progress_bar.start()

    # Spustenie subprocessu na vykonanie detekcie
    def run_subprocess():
        try:
            # Spustenie skriptu predict.py s vybranými parametrami
            subprocess.run(
                ["python", "predict.py", f"model={model_name}.pt", f"source={video_path}", "show=False"],
                check=True
            )
            # Aktualizácia stavu po úspešnom dokončení
            status_var.set("Done!")
            messagebox.showinfo("Success", "Detection completed successfully!")
        except subprocess.CalledProcessError as e:
            # Spracovanie chyby, ak sa vyskytne
            status_var.set("Error during prediction")
            messagebox.showerror("Error", f"Error running predict.py: {e}")
        finally:
            # Zastavenie indikátora priebehu v každom prípade
            progress_bar.stop()

    # Spustenie subprocess v samostatnom vlákne
    threading.Thread(target=run_subprocess).start()

# Funkcia na výber videa pomocou dialógového okna
def select_video():
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4"), ("All Files", "*.*")]
    )
    if video_path:
        video_var.set(video_path)

# Nastavenie hlavného okna aplikácie
root = tk.Tk()
root.title("YOLO Contamination Detection")

# Rozbaľovací zoznam pre výber modelu YOLO
tk.Label(root, text="Select YOLOv8 Model:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
model_var = tk.StringVar(value="yolov8s")
model_options = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
tk.OptionMenu(root, model_var, *model_options).grid(row=0, column=1, padx=10, pady=10, sticky="ew")

# Sekcia pre výber video súboru
tk.Label(root, text="Select Video File:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
video_var = tk.StringVar()
tk.Entry(root, textvariable=video_var, width=50).grid(row=1, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_video).grid(row=1, column=2, padx=10, pady=10)

# Sekcia pre zobrazenie stavu a indikátora priebehu
status_var = tk.StringVar(value="")
tk.Label(root, textvariable=status_var, fg="blue").grid(row=2, column=0, columnspan=3, pady=5)
progress_bar = ttk.Progressbar(root, mode='indeterminate')
progress_bar.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10)

# Tlačidlo pre spustenie detekcie kontaminácie
tk.Button(root, text="Run Contamination Detection", command=run_predict, bg="lightblue").grid(row=4, column=0, columnspan=3, pady=10)

# Spustenie hlavnej slučky aplikácie
root.mainloop()
