import os
import shutil
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import psutil
import ctypes

# -------- CONFIG -------- #
DEST_DIR = os.path.join(os.getenv('APPDATA'), 'nvda')
BUNDLED_PLUGIN_DIR_NAME = "plugin_files"

INSTALL_ITEMS = {
    "Rehnuma Akbar (Male)": "celestial_x",
    "Rehnuma Zainab (Female)": "sada-e-niswan",
    "Ryan (Male)": "ryan",
    "Amy (Female)": "amy"
}

# -------- FUNCTIONS -------- #
def is_nvda_running():
    for proc in psutil.process_iter(attrs=['name']):
        if proc.info['name'] and 'nvda.exe' in proc.info['name'].lower():
            return True
    return False

def is_nvda_installed():
    appdata_exists = os.path.exists(DEST_DIR)
    possible_paths = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "NVDA", "nvda.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "NVDA", "nvda.exe"),
    ]
    exe_found = any(os.path.exists(path) for path in possible_paths)
    return appdata_exists and exe_found

def preview_voice(voice_display_name):
    # Map display name to folder and wav file
    voice_to_folder = {
        "Rehnuma Akbar (Male)": "celestial_x",
        "Rehnuma Zainab (Female)": "sada-e-niswan",
        "Ryan (Male)": "ryan",
        "Amy (Female)": "amy"
    }
    
    try:
        base_dir = getattr(sys, '_MEIPASS', os.path.abspath("."))
        folder = voice_to_folder.get(voice_display_name)
        if not folder:
            messagebox.showerror("Error", f"No preview available for {voice_display_name}")
            return
        wav_path = os.path.join(base_dir, BUNDLED_PLUGIN_DIR_NAME, folder, "preview.wav")
        
        if not os.path.exists(wav_path):
            messagebox.showerror("Error", f"Preview audio not found: {wav_path}")
            return
        
        play_audio(wav_path)
        
    except Exception as e:
        messagebox.showerror("Preview Error", f"Cannot preview audio:\n{e}")


def play_audio(wav_file_path):
    import simpleaudio as sa

    try:
        wave_obj = sa.WaveObject.from_wave_file(wav_file_path)
        play_obj = wave_obj.play()
        # No blocking: it will play in background
    except Exception as e:
        messagebox.showerror("Audio Error", f"Cannot play audio:\n{e}")

def count_total_files(source_dir):
    total = 0
    for root, dirs, files in os.walk(source_dir):
        total += len(files)
    return total

def copy_with_progress(source_dir, dest_dir):
    total_files = count_total_files(source_dir)
    copied_files = 0

    os.makedirs(dest_dir, exist_ok=True)

    for root_dir, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root_dir, source_dir)
        target_root = os.path.join(dest_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            src_file = os.path.join(root_dir, file)
            dst_file = os.path.join(target_root, file)
            shutil.copy2(src_file, dst_file)
            copied_files += 1
            progress = int((copied_files / total_files) * 100)
            progress_var.set(progress)
            progress_label.config(text=f"{progress}%")
            progress_bar.update()

def install_folder(folder_name):
    if not is_nvda_installed():
        messagebox.showerror("NVDA Not Found", "NVDA is not installed. Please install NVDA first.")
        return
    if is_nvda_running():
        ctypes.windll.user32.MessageBoxW(0, "Please close NVDA before installing.", "Error", 0)
        return
    try:
        base_dir = getattr(sys, '_MEIPASS', os.path.abspath("."))
        source_dir = os.path.join(base_dir, BUNDLED_PLUGIN_DIR_NAME, folder_name)
        if not os.path.exists(source_dir):
            messagebox.showerror("Error", f"Folder not found: {source_dir}")
            return

        os.makedirs(DEST_DIR, exist_ok=True)

        for btn in start_buttons.values():
            btn.config(state='disabled')

        threading.Thread(target=lambda: copy_and_finish(source_dir, folder_name), daemon=True).start()

    except Exception as e:
        messagebox.showerror("Error", f"Installation failed: {e}")

def copy_and_finish(source_dir, folder_name):
    copy_with_progress(source_dir, DEST_DIR)
    root.after(0, lambda: finish_install(folder_name))


def finish_install(folder_name):
    mapping= {
          "celestial_x":"Akbar",
         "sada-e-niswan": "Zainab",
        "ryan":"Ryan" ,
       "amy": "Amy"
    }
    messagebox.showinfo("Success", f"{mapping[folder_name].capitalize()} voice installed successfully.")
    for btn in start_buttons.values():
        btn.config(state='normal')
    progress_var.set(0)
    progress_label.config(text="0%")

def close_app():
    root.quit()

# -------- GUI SETUP -------- #
root = tk.Tk()
root.title("Rehnuma Awaz - Urdu TTS Installer")
root.geometry("540x670")
root.configure(bg="white")
root.resizable(False, False)

# --- Modern Flat Styles --- #
style = ttk.Style()
style.theme_use('clam')

style.configure("TButton",
                font=('Segoe UI', 10, 'bold'),
                padding=6,
                relief="flat",
                borderwidth=0)
style.map("TButton",
          background=[('active', '#43A047')],
          foreground=[('disabled', 'gray')])

style.configure("Core.TButton",
                font=('Segoe UI', 10, 'bold'),
                padding=6,
                relief="flat",
                background="#2196F3",
                foreground="white",
                borderwidth=0)
style.map("Core.TButton",
          background=[('active', '#1976D2')])

style.configure("Preview.TButton",
                font=('Segoe UI', 10, 'bold'),
                padding=6,
                relief="flat",
                background="#03A9F4",
                foreground="white",
                borderwidth=0)
style.map("Preview.TButton",
          background=[('active', '#0288D1')])

style.configure("Install.TButton",
                font=('Segoe UI', 10, 'bold'),
                padding=6,
                relief="flat",
                background="#4CAF50",
                foreground="white",
                borderwidth=0)
style.map("Install.TButton",
          background=[('active', '#388E3C')])

style.configure("Close.TButton",
                font=('Segoe UI', 10, 'bold'),
                padding=6,
                relief="flat",
                background="#9E9E9E",
                foreground="white",
                borderwidth=0)
style.map("Close.TButton",
          background=[('active', '#757575')])

style.configure("green.Horizontal.TProgressbar",
    troughcolor='white',   # Background of the progress bar
    background='#4CAF50',   # Fill color (green)
    thickness=20            # Optional: make it thicker
)

# Labels normal (no forced background anymore)
style.configure("TLabel", font=('Segoe UI', 11))

main_frame = ttk.Frame(root, padding="20 20 20 20")
main_frame.pack(fill="both", expand=True)

def section_separator():
    sep = ttk.Separator(main_frame, orient="horizontal")
    sep.pack(fill="x", pady=15)

def add_section_header(text):
    header = ttk.Label(main_frame, text=text, font=('Segoe UI', 13, 'bold'))
    header.pack(anchor="w", pady=(10, 5))

def add_voice_row(voice_name, install_folder_name):
    row = ttk.Frame(main_frame)
    row.pack(fill="x", pady=8)
    ttk.Label(row, text=voice_name, width=25).pack(side="left")
    preview_btn = ttk.Button(row, text="Preview", style="Preview.TButton", command=lambda v=voice_name: preview_voice(v))
    preview_btn.pack(side="right", padx=5)
    install_btn = ttk.Button(row, text="Install", style="Install.TButton", command=lambda v=install_folder_name: install_folder(v))
    install_btn.pack(side="right", padx=5)
    start_buttons[install_folder_name] = install_btn

start_buttons = {}

# -- Core NVDA Plugin Section -- #
core_row = ttk.Frame(main_frame)
core_row.pack(fill="x", pady=10)
section_separator()

# Urdu Voices
add_section_header("Urdu Voices")
for voice in ["Rehnuma Akbar (Male)", "Rehnuma Zainab (Female)"]:
    add_voice_row(voice, INSTALL_ITEMS[voice])

section_separator()

# English Voices
add_section_header("English Voices")
for voice in ["Ryan (Male)", "Amy (Female)"]:
    add_voice_row(voice, INSTALL_ITEMS[voice])

section_separator()

# Progress Bar
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100, length=450, style="green.Horizontal.TProgressbar")
progress_bar.pack(pady=(20, 5))
progress_label = ttk.Label(main_frame, text="0%")
progress_label.pack()

# Close Button
close_btn = ttk.Button(main_frame, text="Close", style="Close.TButton", command=close_app)
close_btn.pack(pady=(30, 0))

root.mainloop()
