import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import sys
import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import time
import cv2
import numpy as np
from pipeline import PipelineOptions, run_pipeline

class ConsoleRedirector:
    def __init__(self, gui_instance, text_widget):
        self.gui_instance = gui_instance
        self.text_widget = text_widget
        self.last_was_progress = False
        
    def write(self, text):
        if "[PROGRESS:" in text:
            try:
                progress = int(text.split("[PROGRESS:")[1].split("]")[0])
                self.gui_instance.root.after(0, self.gui_instance.update_progress, progress)
            except:
                pass
            self.last_was_progress = True
            return
            
        if text == "\n" and self.last_was_progress:
            self.last_was_progress = False
            return
            
        self.last_was_progress = False
        self.gui_instance.root.after(0, self._safe_write, text)
        
    def _safe_write(self, text):
        try:
            self.text_widget.insert(tk.END, text)
            self.text_widget.see(tk.END)
            self.text_widget.update_idletasks()
        except Exception:
            pass
        
    def flush(self):
        pass

class FocusStackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive FocusStack Pro")
        self.root.geometry("600x600")
        
        # Folder selection
        frame_top = tk.Frame(root)
        frame_top.pack(pady=10, padx=10, fill=tk.X)
        
        self.lbl_folder = tk.Label(frame_top, text="No folder selected", fg="gray")
        self.lbl_folder.pack(side=tk.LEFT, padx=10)
        
        btn_browse = tk.Button(frame_top, text="Browse Images...", command=self.browse_folder)
        btn_browse.pack(side=tk.RIGHT)
        
        self.folder_path = None
        
        # Processing Mode
        frame_mode = tk.Frame(root)
        frame_mode.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(frame_mode, text="Processing Mode:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        self.proc_mode_var = tk.StringVar(value="Single Folder")
        self.combo_mode = ttk.Combobox(
            frame_mode,
            textvariable=self.proc_mode_var,
            values=["Single Folder", "Batch (All Subfolders)"],
            state="readonly",
            width=30
        )
        self.combo_mode.pack(side=tk.LEFT, padx=5)
        
        # Stacking Method Selection
        frame_method = tk.Frame(root)
        frame_method.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(frame_method, text="Method:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        self.method_var = tk.StringVar(value="Laplacian Pyramid (Best Quality)")
        self.combo_method = ttk.Combobox(
            frame_method, 
            textvariable=self.method_var, 
            values=["Laplacian Pyramid (Best Quality)", "Weighted Average (Fastest for Huge Images)", "Strict Depth Map"],
            state="readonly",
            width=30
        )
        self.combo_method.pack(side=tk.LEFT, padx=5)
        
        tk.Label(frame_method, text="Kernel Size:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        self.kernel_var = tk.IntVar(value=1) # 1 = Auto mode (resolution-adaptive kernel)
        self.spin_kernel = tk.Spinbox(frame_method, from_=1, to=31, increment=2, textvariable=self.kernel_var, width=5, font=("Arial", 10))
        self.spin_kernel.pack(side=tk.LEFT)
        
        tk.Label(frame_method, text="Pyramid:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        self.pyramid_var = tk.IntVar(value=5)
        self.spin_pyramid = tk.Spinbox(frame_method, from_=1, to=12, textvariable=self.pyramid_var, width=5, font=("Arial", 10))
        self.spin_pyramid.pack(side=tk.LEFT)
        
        # Energy Scale Selection
        frame_energy = tk.Frame(root)
        frame_energy.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(frame_energy, text="Focus Map Scale:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        self.energy_scale_var = tk.StringVar(value="100% (Maximum Quality - Helicon Equivalent)")
        self.combo_energy = ttk.Combobox(
            frame_energy, 
            textvariable=self.energy_scale_var, 
            values=["25% (Fastest)", "50% (Balanced)", "100% (Maximum Quality - Helicon Equivalent)"],
            state="readonly",
            width=45
        )
        self.combo_energy.pack(side=tk.LEFT, padx=5)
        
        # Max Power Checkbox
        frame_mid = tk.Frame(root)
        frame_mid.pack(pady=10, padx=10, fill=tk.X)
        
        self.max_power_var = tk.BooleanVar()
        self.chk_max_power = tk.Checkbutton(
            frame_mid, 
            text="In-Memory Caching (Saves disk read/write, requires 16GB+ RAM)", 
            variable=self.max_power_var,
            font=("Arial", 10, "bold"),
            fg="blue"
        )
        self.chk_max_power.pack(anchor=tk.W)
        
        self.crop_var = tk.BooleanVar(value=True)
        self.chk_crop = tk.Checkbutton(
            frame_mid, 
            text="Crop Aligned Images (Uncheck to keep original size and avoid losing data)", 
            variable=self.crop_var,
            font=("Arial", 10, "bold"),
            fg="green"
        )
        self.chk_crop.pack(anchor=tk.W)
        
        self.deghost_var = tk.BooleanVar(value=False) # Disabled by default for speed
        
        frame_deghost = tk.Frame(frame_mid)
        frame_deghost.pack(anchor=tk.W, fill=tk.X)
        
        self.chk_deghost = tk.Checkbutton(
            frame_deghost, 
            text="Optical Flow De-Ghosting (Prevents subject movement halos, SLOW)", 
            variable=self.deghost_var,
            font=("Arial", 10, "bold"),
            fg="purple"
        )
        self.chk_deghost.pack(side=tk.LEFT)
        
        self.deghost_quality_var = tk.StringVar(value="Medium (Fast)")
        self.combo_deghost_quality = ttk.Combobox(
            frame_deghost,
            textvariable=self.deghost_quality_var,
            values=["High Quality (Slow)", "Medium (Fast)", "Fastest (Low Quality)"],
            state="readonly",
            width=20
        )
        self.combo_deghost_quality.pack(side=tk.LEFT, padx=5)
        
        self.exposure_var = tk.BooleanVar(value=False) # Disabled by default to prevent color/brightness shifts
        self.chk_exposure = tk.Checkbutton(
            frame_mid, 
            text="Normalize Exposure (Fixes frame brightness differences, can cause halos on dark backgrounds)", 
            variable=self.exposure_var,
            font=("Arial", 10, "bold"),
            fg="brown"
        )
        self.chk_exposure.pack(anchor=tk.W)
        
        # Progress Bar and ETA
        frame_prog = tk.Frame(root)
        frame_prog.pack(pady=5, padx=10, fill=tk.X)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame_prog, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.lbl_eta = tk.Label(frame_prog, text="ETA: --", font=("Arial", 9, "bold"))
        self.lbl_eta.pack(side=tk.RIGHT)
        self.start_time = None
        
        # Start button
        self.btn_start = tk.Button(root, text="Start Stacking!", font=("Arial", 12, "bold"), bg="green", fg="white", command=self.start_stacking)
        self.btn_start.pack(pady=10, fill=tk.X, padx=20)
        
        # Console output
        self.console = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15)
        self.console.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Redirect print and errors
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self, self.console)
        sys.stderr = ConsoleRedirector(self, self.console)
        print("Welcome to Adaptive FocusStack Pro!")
        print("Waiting for folder selection...")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        # Restore stdout/stderr when the window is closed
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.root.destroy()

    def update_progress(self, val):
        self.progress_var.set(val)
        if self.start_time and val > 0 and val < 100:
            elapsed = time.time() - self.start_time
            total_est = (elapsed / val) * 100
            rem = total_est - elapsed
            self.lbl_eta.config(text=f"ETA: {int(rem)}s")
        elif val >= 100:
            self.lbl_eta.config(text="Done!")
        self.root.update_idletasks()

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            self.lbl_folder.config(text=self.folder_path, fg="black")
            print(f"Selected: {self.folder_path}")

    def start_stacking(self):
        if not self.folder_path:
            print("Error: Please select a folder first!")
            return
            
        self.btn_start.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.lbl_eta.config(text="ETA: --")
        self.start_time = time.time()
        # Run in background thread to avoid freezing GUI
        threading.Thread(target=self.run_engine, daemon=True).start()

    def run_engine(self):
        try:
            import re
            folders_to_process = []
            if "Batch" in self.proc_mode_var.get():
                for d in os.listdir(self.folder_path):
                    full_d = os.path.join(self.folder_path, d)
                    # Skip the newly created outputs directory to prevent recursive loop
                    if os.path.isdir(full_d) and os.path.basename(full_d) != "outputs":
                        folders_to_process.append(full_d)
                if not folders_to_process:
                    print("No subfolders found for batch processing.")
                    return
            else:
                folders_to_process = [self.folder_path]
                
            for current_folder in folders_to_process:
                print(f"\n==============================================")
                print(f"Processing Folder: {current_folder}")
                print(f"==============================================")
                
                try:
                    self._process_single_folder(current_folder, re)
                except Exception as e:
                    print(f"\nCRITICAL ERROR in {current_folder}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing to next folder...")
                    
        except Exception as e:
            print(f"\nFATAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
            print("=== Finished All Tasks ===")
            
    def _process_single_folder(self, folder_path, re):
        image_paths = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png'))
        ]
        image_paths.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
        
        if not image_paths:
            print(f"No valid images found in folder: {folder_path}")
            return
            
        print(f"\n=== Starting Run with {len(image_paths)} images ===")
        
        # Setup Pipeline Options
        options = PipelineOptions()
        options.in_memory = self.max_power_var.get()
        options.enable_crop = self.crop_var.get()
        options.deghost = self.deghost_var.get()
        
        dq = self.deghost_quality_var.get()
        if "High" in dq:
            options.deghost_preset = "high"
        elif "Fastest" in dq:
            options.deghost_preset = "fast"
        else:
            options.deghost_preset = "medium"
            
        options.exposure_norm = self.exposure_var.get()
        options.kernel_size = int(self.kernel_var.get())
        options.pyramid_levels = int(self.pyramid_var.get())
        
        # Determine Method
        sel_method = self.method_var.get()
        if "Average" in sel_method:
            options.method = "weighted_average"
        elif "Depth" in sel_method:
            options.method = "depth_map"
        else:
            options.method = "pyramid"
            
        try:
            options.kernel_size = int(self.kernel_var.get())
            # kernel=1 means auto mode; values >1 are user-specified
            if options.kernel_size > 1 and options.kernel_size % 2 == 0:
                options.kernel_size += 1
        except Exception:
            options.kernel_size = 1  # Auto mode
            
        try:
            options.pyramid_levels = int(self.pyramid_var.get())
            if options.pyramid_levels < 1: options.pyramid_levels = 1
        except Exception:
            options.pyramid_levels = 5
            
        # Determine Energy Scale
        sel_scale = self.energy_scale_var.get()
        if "100%" in sel_scale:
            options.energy_scale = 1.0
        elif "50%" in sel_scale:
            options.energy_scale = 0.5
        else:
            options.energy_scale = 0.25
            
        # In batch mode, we put outputs in the parent directory of the current folder
        if "Batch" in self.proc_mode_var.get():
            options.output_dir = os.path.join(os.path.dirname(folder_path), "outputs")
            
        try:
            run_pipeline(folder_path, options)
            print("\n--- Folder Processing Complete ---")
        except Exception as e:
            print(f"\nFATAL ERROR processing folder {folder_path}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = FocusStackGUI(root)
    root.mainloop()
