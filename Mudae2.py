import json
import logging
import os
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Dict, Tuple, Optional, Any, Callable
from collections import defaultdict

import numpy as np
import cv2 # opencv-python
import pyautogui
import pytesseract
from PIL import Image, ImageTk, ImageGrab

# Attempt to import PyTorch related libraries
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # Create dummy classes/functions if PyTorch is not available
    # This allows the Sorter tab to load but disable AI features.
    class models: # type: ignore
        class efficientnet_b0: # type: ignore
            def __init__(self, weights=None): pass
            def children(self): return []
        class EfficientNet_B0_Weights: # type: ignore
            DEFAULT = None

    class transforms: # type: ignore
        class Compose: # type: ignore
            def __init__(self, *args, **kwargs): pass
        class Resize: # type: ignore
            def __init__(self, *args, **kwargs): pass
        class CenterCrop: # type: ignore
            def __init__(self, *args, **kwargs): pass
        class ToTensor: # type: ignore
            def __init__(self, *args, **kwargs): pass
        class Normalize: # type: ignore
            def __init__(self, *args, **kwargs): pass
    
    class torch: # type: ignore
        class nn: # type: ignore
            class Sequential: # type: ignore
                def __init__(self, *args, **kwargs): pass
                def to(self, device): return self
                def eval(self): pass
        
        def device(self, dev_str): return None
        # Corrected dummy for torch.cuda
        class cuda: # type: ignore
            @staticmethod
            def is_available(): return False

        def no_grad(self): # type: ignore
            class NoGradContext:
                def __enter__(self): pass
                def __exit__(self, type, value, traceback): pass
            return NoGradContext()

# Scipy is only used by the Sorter for pdist, squareform
try:
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Dummy pdist and squareform if scipy is not available
    def pdist(data, metric='euclidean'): # type: ignore
        # Simplified pdist that returns zeros, Sorter will handle this
        num_vectors = data.shape[0]
        if num_vectors < 2:
            return np.array([])
        # This is not a real pdist, just a placeholder
        return np.zeros(num_vectors * (num_vectors - 1) // 2)

    def squareform(vec): # type: ignore
        # Simplified squareform
        if vec.size == 0:
            return np.array([[]])
        # Determine n from m = n*(n-1)/2
        # n^2 - n - 2m = 0.  n = (1 + sqrt(1 + 8m)) / 2
        m = vec.size
        n_float = (1 + np.sqrt(1 + 8 * m)) / 2
        if not n_float.is_integer():
                raise ValueError("vec argument must be a 1D condensed distance matrix (invalid size).")
        n = int(n_float)
        if n * (n - 1) // 2 != m: # Double check
            raise ValueError("vec argument must be a 1D condensed distance matrix.")
        
        matrix = np.zeros((n, n))
        # This is not a real squareform fill, just a placeholder
        return matrix


# --- Constants & Configuration ---
APP_NAME = "Mudae Power Tools"
APP_GEOMETRY = "1100x850"
APP_MIN_SIZE = (950, 750)

CONFIG_FILE_NAME = "mudae_power_tools_config.json"
DEFAULT_OUTPUT_FOLDER_NAME = "MudaeCharacters" # Scraper output, Sorter input
DB_FILE_NAME = "character_show_db.json" # Shared character-show database

# Scraper Defaults
DEFAULT_DHASH_SIZE = 8
DEFAULT_HAMMING_DISTANCE_NEW_SCREENSHOT = 2
DEFAULT_HAMMING_DISTANCE_DUPLICATE_CARD = 5
DEFAULT_MIN_CARD_AREA = 100 # Lowered significantly to be less restrictive
DEFAULT_MIN_CARD_ASPECT_RATIO = 0.5 # Widened range
DEFAULT_MAX_CARD_ASPECT_RATIO = 5.0 # Widened range
DEFAULT_OCR_LANGUAGE = "eng"
DEFAULT_TESSERACT_CMD_PATH = ""
DEFAULT_OUTPUT_IMAGE_FORMAT = "PNG"
DEFAULT_JPEG_QUALITY = 90
OCR_CONFIG_SPARSE = '--psm 11' # Sparse text. Find as much text as possible.
# Increased threshold for consecutive similar frames before auto-stop
CONSECUTIVE_SIMILAR_FRAMES_THRESHOLD = 10


# Sorter Defaults
VALID_IMAGE_EXTENSIONS_SORTER = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
DEFAULT_AI_MODEL_SORTER = "efficientnet_b0"
DEFAULT_FEATURE_SIZE_SORTER = 1280 # Corresponds to EfficientNet B0
UNKNOWN_SHOW_SORTER = "Unknown Show"
DISCORD_COMMAND_CHAR_LIMIT_SORTER = 1900

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tooltip Helper Class ---
class Tooltip:
    """Create a tooltip for a given widget."""
    def __init__(self, widget, text='widget info', delay=500, wraplength=300):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self.tooltip_window = None
        self._id_after = None
        widget.bind("<Enter>", self.schedule_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)
        widget.bind("<ButtonPress>", self.hide_tooltip) # Hide on click as well

    def schedule_tooltip(self, event=None):
        self.hide_tooltip() # Cancel any existing scheduled tooltip
        self._id_after = self.widget.after(self.delay, self.show_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window: return # Already showing

        # Calculate position
        x_bbox, y_bbox, _, _ = self.widget.bbox("insert") # Get bbox relative to widget
        x = self.widget.winfo_rootx() + x_bbox + 25 # Add widget's root coords for global positioning
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5 # Place below the widget

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # No window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                            background='#ffffe0', relief='solid', borderwidth=1,
                            font=("tahoma", "8", "normal"), wraplength=self.wraplength)
        label.pack(ipadx=2, ipady=2)

    def hide_tooltip(self, event=None):
        if self._id_after:
            self.widget.after_cancel(self._id_after)
            self._id_after = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# --- Progress Dialog Helper Class (from MudaeVisualSorter) ---
class ProgressDialog(tk.Toplevel):
    """A simple progress dialog window."""
    def __init__(self, parent, title="Processing...", total_items=100):
        super().__init__(parent)
        self.title(title)
        self.geometry("500x150")
        self.transient(parent) # Show on top of the parent window
        self.grab_set() # Modal behavior
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close_button) # Handle manual close

        self.progress_label = ttk.Label(self, text="Initializing...")
        self.progress_label.pack(pady=10, padx=10, fill="x")

        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=450, mode="determinate")
        self.progress_bar.pack(pady=10, padx=10)
        self.progress_bar["maximum"] = total_items

        self.status_label = ttk.Label(self, text="Starting...", wraplength=450)
        self.status_label.pack(pady=5, padx=10, fill="x")
        
        self.cancelled = False
        self.update_idletasks() # Ensure widgets are drawn

    def _on_close_button(self):
        # This is a basic way to signal cancellation.
        # The long-running task needs to check self.cancelled periodically.
        if messagebox.askyesno("Cancel?", "Are you sure you want to cancel this operation?", parent=self):
            self.cancelled = True
            self.destroy() # Or self.close_dialog() if you add more logic there

    def update_progress(self, current_item: int, status_text: str):
        """Updates the progress bar and status text."""
        if not self.winfo_exists(): return
        self.progress_bar["value"] = current_item
        self.status_label.config(text=status_text)
        self.update() # Force GUI update

    def set_phase(self, phase_text: str):
        """Updates the main progress label for different phases."""
        if not self.winfo_exists(): return
        self.progress_label.config(text=phase_text)
        self.update()

    def close_dialog(self): # Renamed from 'close' to avoid conflict with tk.Toplevel.close
        """Closes the progress dialog."""
        if self.winfo_exists():
            self.destroy()

# --- Region Selector (from MudaeImageScraper) ---
class RegionSelector:
    def __init__(self, parent: Optional[tk.Tk] = None, title: str = "Select Region"):
        self.start_x: int = 0
        self.start_y: int = 0
        self.end_x: int = 0
        self.end_y: int = 0
        self.selecting: bool = False
        self.selected_region: Optional[Tuple[int, int, int, int]] = None
        self.parent_window = parent
        self.title_text = title # Store title for reuse

        if self.parent_window and self.parent_window.winfo_exists():
            self.root = tk.Toplevel(self.parent_window)
        else:
            self.root = tk.Tk()
            self.parent_window = None

        self.root.title(title)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.configure(background='black')
        self.root.wait_visibility() # Ensure window is visible before grab_set
        self.root.grab_set()

        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.instructions_text_id = self.canvas.create_text(
            screen_width // 2, screen_height // 2,
            text=f"{title}\nClick and drag to select the area.\nPress ESC to cancel.",
            fill="white", font=("Arial", 24, "bold"), justify="center", tags="instructions"
        )
        self.rect_id: Optional[int] = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Escape>", self.on_cancel)
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close_button)

    def _handle_close_button(self):
        logging.info("Region selection cancelled via window close button.")
        self.selected_region = None
        self._destroy_selector(focus_parent=True)

    def on_press(self, event: tk.Event):
        self.selecting = True
        self.start_x = event.x
        self.start_y = event.y
        self.canvas.delete("instructions")
        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=3, fill='red', stipple='gray25'
        )

    def on_drag(self, event: tk.Event):
        if not self.selecting: return
        self.end_x = event.x
        self.end_y = event.y
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, self.end_x, self.end_y)

    def on_release(self, event: tk.Event):
        if not self.selecting: return
        self.end_x = event.x
        self.end_y = event.y
        self.selecting = False

        x1, y1 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
        x2, y2 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)

        if x2 - x1 < 10 or y2 - y1 < 10:
            messagebox.showwarning("Invalid Region", "Selected region is too small. Please try again.", parent=self.root)
            self.selected_region = None
            if self.rect_id: self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.instructions_text_id = self.canvas.create_text(
                self.root.winfo_screenwidth() // 2, self.root.winfo_screenheight() // 2,
                text=f"{self.title_text}\nClick and drag to select the area.\nPress ESC to cancel.",
                fill="white", font=("Arial", 24, "bold"), justify="center", tags="instructions"
            )
            return

        self.selected_region = (x1, y1, x2, y2)
        self.canvas.create_text(
            self.root.winfo_screenwidth() // 2, self.root.winfo_screenheight() // 2,
            text="Region selected! Closing selector...",
            fill="lime green", font=("Arial", 24, "bold"), tags="confirmation"
        )
        self.root.after(750, lambda: self._destroy_selector(focus_parent=True))

    def on_cancel(self, event: Optional[tk.Event] = None):
        logging.info("Region selection cancelled by user (ESC).")
        self.selected_region = None
        self._destroy_selector(focus_parent=True)

    def _destroy_selector(self, focus_parent: bool = True):
        if self.root.winfo_exists():
            self.root.grab_release()
            self.root.destroy()
        if focus_parent and self.parent_window and self.parent_window.winfo_exists():
            self.parent_window.deiconify()
            self.parent_window.lift()
            self.parent_window.focus_force()

    def run(self) -> Optional[Tuple[int, int, int, int]]:
        if self.root.winfo_exists():
            self.root.deiconify()
            self.root.wait_window(self.root) # Block until selector window is closed
        return self.selected_region

# --- Manual Card Selector Dialog ---
class ManualCardSelector(tk.Toplevel):
    def __init__(self, parent, screenshot_image: Image.Image, title="Manually Select Card"):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title(title)
        
        self.screenshot_image = screenshot_image # Store original image
        self.cropped_image_result: Optional[Image.Image] = None
        self.rect_coords: Optional[Tuple[int, int, int, int]] = None 
        self.start_x, self.start_y = 0, 0
        self.current_rect_id = None

        # Frame for canvas and scrollbars
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, background="lightgrey", cursor="crosshair")
        
        # Add scrollbars
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tk_screenshot_image = ImageTk.PhotoImage(self.screenshot_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_screenshot_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Escape>", self._on_cancel_event)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Confirm Selection", command=self._on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel (Skip Screenshot)", command=self._on_cancel).pack(side=tk.LEFT, padx=5)
        
        instr_label = ttk.Label(self, text="Click and drag on the image to select the card area.")
        instr_label.pack(pady=(0,5))

        # Window sizing and positioning
        self.update_idletasks() 

        img_w, img_h = self.screenshot_image.size
        max_canvas_width = parent.winfo_screenwidth() * 0.7 
        max_canvas_height = parent.winfo_screenheight() * 0.6 # Adjusted for better visibility

        canvas_display_width = min(img_w, max_canvas_width)
        canvas_display_height = min(img_h, max_canvas_height)
        
        chrome_width_estimate = 40 
        chrome_height_estimate = 120 # Increased for instructions and buttons

        final_win_width = int(canvas_display_width + chrome_width_estimate)
        final_win_height = int(canvas_display_height + chrome_height_estimate)
        
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        
        x_pos = parent_x + (parent_w - final_win_width) // 2
        y_pos = parent_y + (parent_h - final_win_height) // 2
        
        self.geometry(f"{final_win_width}x{final_win_height}+{max(0,x_pos)}+{max(0,y_pos)}")
        self.minsize(300, 200) # Minimum window size

    def _on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
        self.current_rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def _on_drag(self, event):
        if not self.current_rect_id: return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def _on_release(self, event):
        if not self.current_rect_id: return
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        x1 = int(min(self.start_x, end_x))
        y1 = int(min(self.start_y, end_y))
        x2 = int(max(self.start_x, end_x))
        y2 = int(max(self.start_y, end_y))

        img_w, img_h = self.screenshot_image.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            messagebox.showwarning("Small Selection", "Selected area is too small. Please try again.", parent=self)
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            self.rect_coords = None
        else:
            self.rect_coords = (x1, y1, x2, y2)
            self.canvas.coords(self.current_rect_id, x1, y1, x2, y2) # Update to valid coords

    def _on_confirm(self):
        if self.rect_coords:
            try:
                self.cropped_image_result = self.screenshot_image.crop(self.rect_coords)
                logging.info(f"Manual selection confirmed: {self.rect_coords}")
                self.destroy()
            except Exception as e:
                logging.error(f"Error cropping image manually: {e}")
                messagebox.showerror("Crop Error", f"Could not crop the image: {e}", parent=self)
                self.cropped_image_result = None
                self.destroy()
        else:
            messagebox.showwarning("No Selection", "Please select a region on the image first.", parent=self)

    def _on_cancel_event(self, event=None):
        self._on_cancel()

    def _on_cancel(self):
        logging.info("Manual card selection cancelled by user.")
        self.cropped_image_result = None
        self.destroy()

    def run(self) -> Optional[Image.Image]:
        self.wait_window()
        return self.cropped_image_result

# --- Stop Capture Window (from MudaeImageScraper) ---
class StopCaptureWindow:
    def __init__(self, parent: tk.Tk, stop_callback: Callable):
        self.stop_callback = stop_callback
        self.root = tk.Toplevel(parent)
        self.root.title("Capture Control")
        # Position it at top-left, adjust as needed
        self.root.geometry(f"220x100+{parent.winfo_rootx()+50}+{parent.winfo_rooty()+50}")
        self.root.attributes("-topmost", True)

        stop_button = ttk.Button(self.root, text="Stop Capture (ESC or Close)", command=self.stop_capture_action)
        stop_button.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.root.bind("<Escape>", self.stop_capture_action)
        self.root.protocol("WM_DELETE_WINDOW", self.stop_capture_action)

    def stop_capture_action(self, event: Optional[tk.Event] = None):
        self.stop_callback()
        self.close_window()

    def show(self):
        if self.root.winfo_exists():
            self.root.deiconify()

    def close_window(self):
        if self.root.winfo_exists():
            self.root.destroy()

# --- Advanced Settings Dialog (from MudaeImageScraper, for Scraper settings) ---
class ScraperAdvancedSettingsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, app_instance: 'MudaePowerToolsApp'):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.app = app_instance # Main app instance
        self.title("Scraper Advanced Settings")
        self.geometry("450x400") # Adjusted size
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.vars: Dict[str, tk.StringVar] = {}

        frame = ttk.Frame(self, padding="10")
        frame.pack(expand=True, fill=tk.BOTH)

        settings_map = {
            "dhash_size": "dHash Size (e.g., 8):",
            "hamming_distance_new_screenshot": "New Screenshot Hamming Threshold (e.g., 2):",
            "hamming_distance_duplicate_card": "Duplicate Card Hamming Threshold (e.g., 5):",
            "min_card_area": "Min Card Area (pixels, e.g., 8000):",
            "min_card_aspect_ratio": "Min Card Aspect Ratio (H/W, e.g., 1.2):",
            "max_card_aspect_ratio": "Max Card Aspect Ratio (H/W, e.g., 3.0):"
        }

        row_idx = 0
        for key, label_text in settings_map.items():
            ttk.Label(frame, text=label_text).grid(row=row_idx, column=0, sticky="w", pady=3, padx=5)
            self.vars[key] = tk.StringVar(value=str(getattr(self.app, key))) # Get from main app
            entry = ttk.Entry(frame, textvariable=self.vars[key], width=10)
            entry.grid(row=row_idx, column=1, sticky="ew", pady=3, padx=5)
            Tooltip(entry, f"Current value: {getattr(self.app, key)}. Lower Hamming means more similar.")
            row_idx += 1

        frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=row_idx, column=0, columnspan=2, pady=20)

        save_btn = ttk.Button(button_frame, text="Save & Close", command=self._on_save)
        save_btn.pack(side=tk.LEFT, padx=10)
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side=tk.LEFT, padx=10)
        reset_btn = ttk.Button(button_frame, text="Reset to Defaults", command=self._on_reset_defaults)
        reset_btn.pack(side=tk.LEFT, padx=10)

    def _on_save(self):
        try:
            for key, var in self.vars.items():
                value_str = var.get()
                if key in ["min_card_aspect_ratio", "max_card_aspect_ratio"]:
                    value = float(value_str)
                else:
                    value = int(value_str)
                setattr(self.app, key, value) # Set on main app
            self.app._save_config() # Main app saves its config
            self.app.status_var.set("Scraper advanced settings saved.")
            logging.info("Scraper advanced settings updated and saved.")
            self.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please ensure all values are correct numbers.", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings: {e}", parent=self)
            logging.error(f"Error saving scraper advanced settings: {e}")

    def _on_cancel(self):
        self.destroy()

    def _on_reset_defaults(self):
        self.vars["dhash_size"].set(str(DEFAULT_DHASH_SIZE))
        self.vars["hamming_distance_new_screenshot"].set(str(DEFAULT_HAMMING_DISTANCE_NEW_SCREENSHOT))
        self.vars["hamming_distance_duplicate_card"].set(str(DEFAULT_HAMMING_DISTANCE_DUPLICATE_CARD))
        self.vars["min_card_area"].set(str(DEFAULT_MIN_CARD_AREA))
        self.vars["min_card_aspect_ratio"].set(str(DEFAULT_MIN_CARD_ASPECT_RATIO))
        self.vars["max_card_aspect_ratio"].set(str(DEFAULT_MAX_CARD_ASPECT_RATIO))
        self.app.status_var.set("Scraper advanced settings reset. Click 'Save & Close' to apply.")

# --- Helper Functions (dHash, sanitize_filename) ---
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s.-]', '', name).strip()
    name = re.sub(r'[-\s]+', '_', name)
    return name if name else "unknown_character"

def dhash(image: Image.Image, hash_size: int = DEFAULT_DHASH_SIZE) -> Optional[int]:
    if image is None:
        logging.error("dHash received a None image.")
        return None
    try:
        image = image.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = np.array(image)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])
    except Exception as e:
        logging.error(f"Error calculating dHash: {e}")
        return None

def hamming_distance(hash1: Optional[int], hash2: Optional[int]) -> int:
    if hash1 is None or hash2 is None:
        return float('inf') # Treat None hashes as infinitely different
    return bin(hash1 ^ hash2).count('1')


# --- Main Application Class ---
class MudaePowerToolsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry(APP_GEOMETRY)
        self.root.minsize(APP_MIN_SIZE[0], APP_MIN_SIZE[1])

        # --- Initialize Shared Variables & Configurable Parameters ---
        # Scraper related (will be primarily managed by ScraperTab but stored here for config)
        self.output_folder: str = os.path.join(os.path.expanduser("~"), DEFAULT_OUTPUT_FOLDER_NAME)
        self.capture_region: Optional[Tuple[int, int, int, int]] = None
        self.next_button_region: Optional[Tuple[int, int, int, int]] = None
        self.character_name_sub_region: Optional[Tuple[int, int, int, int]] = None
        self.show_name_sub_region: Optional[Tuple[int, int, int, int]] = None
        self.dhash_size: int = DEFAULT_DHASH_SIZE
        self.hamming_distance_new_screenshot: int = DEFAULT_HAMMING_DISTANCE_NEW_SCREENSHOT
        self.hamming_distance_duplicate_card: int = DEFAULT_HAMMING_DISTANCE_DUPLICATE_CARD
        self.min_card_area: int = DEFAULT_MIN_CARD_AREA
        self.min_card_aspect_ratio: float = DEFAULT_MIN_CARD_ASPECT_RATIO
        self.max_card_aspect_ratio: float = DEFAULT_MAX_CARD_ASPECT_RATIO
        self.ocr_language: str = DEFAULT_OCR_LANGUAGE
        self.tesseract_cmd_path: str = DEFAULT_TESSERACT_CMD_PATH
        self.output_image_format: str = DEFAULT_OUTPUT_IMAGE_FORMAT
        self.jpeg_quality: int = DEFAULT_JPEG_QUALITY
        
        # Shared database
        self.character_show_db: Dict[str, str] = {} # {'char_name': 'show_name'}
        self.db_file_path: str = os.path.join(self.output_folder, DB_FILE_NAME) # Initial path

        # Sorter related (will be primarily managed by SorterTab)
        # AI model related attributes for Sorter
        self.ai_model_sorter: Optional[torch.nn.Sequential] = None
        self.ai_transform_sorter: Optional[transforms.Compose] = None
        self.ai_device_sorter: Optional[torch.device] = None
        self.ai_feature_size_sorter: int = DEFAULT_FEATURE_SIZE_SORTER
        self.model_queue_sorter: queue.Queue = queue.Queue()
        self.model_loading_thread_sorter: Optional[threading.Thread] = None
        
        # Tkinter variables for status, etc.
        self.status_var = tk.StringVar(value="Initializing...")
        self._initial_tesseract_error_shown_startup = False


        self._load_config() # Load combined config
        self._apply_tesseract_path_global(log_success=False) # Apply Tesseract path from config
        self._load_character_db_global() # Load shared DB

        self._setup_styles()
        self._create_main_widgets()
        
        # Initial Tesseract check after UI is up
        self._check_tesseract_availability_global(from_manual_apply=False) 
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_app)

    def _setup_styles(self):
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes: style.theme_use("clam")
        elif "vista" in available_themes: style.theme_use("vista")
        elif "aqua" in available_themes: style.theme_use("aqua") # macOS
        
        style.configure("TButton", padding=6, relief="flat", font=("Arial", 10))
        style.configure("Small.TButton", padding=3, font=("Arial", 8))
        style.configure("TLabel", padding=2, font=("Arial", 10))
        style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        style.configure("Header.TLabel", font=("Arial", 10, "bold"))
        style.configure("TEntry", padding=5, font=("Arial", 10))
        style.configure("TLabelframe", padding=10) # Font set by labelwidget
        style.configure("TLabelframe.Label", font=("Arial", 11, "bold")) # Make LabelFrame titles a bit bigger
        style.configure("TNotebook.Tab", font=("Arial", 10, "bold"), padding=[8, 4])


    def _load_config(self):
        try:
            if os.path.exists(CONFIG_FILE_NAME):
                with open(CONFIG_FILE_NAME, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Scraper settings
                self.output_folder = config.get("output_folder", self.output_folder)
                self.capture_region = tuple(config.get("capture_region")) if config.get("capture_region") else None
                self.next_button_region = tuple(config.get("next_button_region")) if config.get("next_button_region") else None
                self.character_name_sub_region = tuple(config.get("character_name_sub_region")) if config.get("character_name_sub_region") else None
                self.show_name_sub_region = tuple(config.get("show_name_sub_region")) if config.get("show_name_sub_region") else None
                self.dhash_size = config.get("dhash_size", DEFAULT_DHASH_SIZE)
                self.hamming_distance_new_screenshot = config.get("hamming_distance_new_screenshot", DEFAULT_HAMMING_DISTANCE_NEW_SCREENSHOT)
                self.hamming_distance_duplicate_card = config.get("hamming_distance_duplicate_card", DEFAULT_HAMMING_DISTANCE_DUPLICATE_CARD)
                self.min_card_area = config.get("min_card_area", DEFAULT_MIN_CARD_AREA)
                self.min_card_aspect_ratio = config.get("min_card_aspect_ratio", DEFAULT_MIN_CARD_ASPECT_RATIO)
                self.max_card_aspect_ratio = config.get("max_card_aspect_ratio", DEFAULT_MAX_CARD_ASPECT_RATIO)
                self.ocr_language = config.get("ocr_language", DEFAULT_OCR_LANGUAGE)
                self.tesseract_cmd_path = config.get("tesseract_cmd_path", DEFAULT_TESSERACT_CMD_PATH)
                self.output_image_format = config.get("output_image_format", DEFAULT_OUTPUT_IMAGE_FORMAT)
                self.jpeg_quality = config.get("jpeg_quality", DEFAULT_JPEG_QUALITY)
                
                # Sorter settings (if any directly on main app - most are on SorterTab)
                # e.g., self.sorter_some_setting = config.get("sorter_some_setting", DEFAULT_SORTER_SETTING)

                logging.info(f"Configuration loaded from {CONFIG_FILE_NAME}")
            else:
                self.status_var.set("No config file found. Using defaults. Config will be saved on exit or change.")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            messagebox.showerror("Config Error", f"Could not load configuration: {e}\nDefault settings will be used.", parent=self.root if self.root.winfo_exists() else None)
        
        # Update DB path based on potentially loaded output_folder
        self.db_file_path = os.path.join(self.output_folder, DB_FILE_NAME)

    def _save_config(self, show_status: bool = False):
        config = {
            # Scraper settings
            "output_folder": self.output_folder,
            "capture_region": self.capture_region,
            "next_button_region": self.next_button_region,
            "character_name_sub_region": self.character_name_sub_region,
            "show_name_sub_region": self.show_name_sub_region,
            "dhash_size": self.dhash_size,
            "hamming_distance_new_screenshot": self.hamming_distance_new_screenshot,
            "hamming_distance_duplicate_card": self.hamming_distance_duplicate_card,
            "min_card_area": self.min_card_area,
            "min_card_aspect_ratio": self.min_card_aspect_ratio,
            "max_card_aspect_ratio": self.max_card_aspect_ratio,
            "ocr_language": self.ocr_language,
            "tesseract_cmd_path": self.tesseract_cmd_path,
            "output_image_format": self.output_image_format,
            "jpeg_quality": self.jpeg_quality,
            # Add Sorter specific settings if they are on main app
        }
        try:
            config_dir = os.path.dirname(CONFIG_FILE_NAME)
            if config_dir and not os.path.exists(config_dir): # Ensure directory for config exists if not in CWD
                    os.makedirs(config_dir, exist_ok=True) # Make parent dirs if they don't exist
            with open(CONFIG_FILE_NAME, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logging.info(f"Configuration saved to {CONFIG_FILE_NAME}")
            if show_status and self.root.winfo_exists(): self.status_var.set("Configuration saved.")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            if self.root.winfo_exists(): self.status_var.set(f"Error saving config: {e}")

    def _load_character_db_global(self):
        try:
            if os.path.exists(self.db_file_path):
                with open(self.db_file_path, 'r', encoding='utf-8') as f:
                    self.character_show_db = json.load(f)
                logging.info(f"Loaded {len(self.character_show_db)} char-show mappings from {self.db_file_path}")
                self.status_var.set(f"Loaded {len(self.character_show_db)} char-show DB entries.")
            else:
                logging.info(f"Character DB file not found at {self.db_file_path}. Will be created if data processed.")
                self.character_show_db = {} # Ensure it's an empty dict if file not found
                self.status_var.set("Character DB not found. Will be created on data processing.")
        except Exception as e:
            logging.error(f"Error loading character-show database: {e}")
            self.character_show_db = {}
            self.status_var.set(f"Error loading character DB: {e}")

    def _save_character_db_global(self):
        try:
            os.makedirs(os.path.dirname(self.db_file_path), exist_ok=True)
            with open(self.db_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.character_show_db, f, indent=2, sort_keys=True, ensure_ascii=False)
            logging.info(f"Saved {len(self.character_show_db)} char-show mappings to {self.db_file_path}")
            self.status_var.set(f"Character DB saved with {len(self.character_show_db)} entries.")
        except Exception as e:
            logging.error(f"Error saving character-show database: {e}")
            if self.root.winfo_exists(): self.status_var.set(f"Error saving character DB: {e}")

    def _apply_tesseract_path_global(self, log_success: bool = True):
        """Applies the Tesseract path from self.tesseract_cmd_path to pytesseract."""
        path_to_apply = self.tesseract_cmd_path.strip()
        current_pytesseract_cmd = getattr(pytesseract, 'tesseract_cmd', 'tesseract')

        if path_to_apply and os.path.isfile(path_to_apply):
            if current_pytesseract_cmd != path_to_apply:
                try:
                    pytesseract.tesseract_cmd = path_to_apply
                    if log_success: logging.info(f"Pytesseract command set to: {path_to_apply}")
                    return True
                except Exception as e:
                    logging.error(f"Failed to assign Pytesseract command path '{path_to_apply}': {e}")
                    if log_success and self.root.winfo_exists():
                        messagebox.showwarning("Tesseract Path", f"Could not assign Tesseract path:\n{path_to_apply}\n{e}", parent=self.root)
                    return False
            else: # Path is already set
                if log_success: logging.info(f"Pytesseract command already set to: {path_to_apply}")
                return True
        elif path_to_apply: # Path specified but not a valid file
            if log_success:
                logging.warning(f"Configured Tesseract path is not a valid file: {path_to_apply}")
                if self.root.winfo_exists():
                    messagebox.showwarning("Invalid Tesseract Path", f"The path specified for Tesseract is not a valid file:\n{path_to_apply}", parent=self.root)
            return False
        else: # No path, reset to default
            if current_pytesseract_cmd != 'tesseract':
                pytesseract.tesseract_cmd = 'tesseract'
                if log_success: logging.info("Tesseract path cleared. Pytesseract will search system PATH.")
            return True

    def _check_tesseract_availability_global(self, from_manual_apply: bool = False):
        """Checks Tesseract availability and updates status. Shared by ScraperTab."""
        try:
            self._apply_tesseract_path_global(log_success=False) # Ensure current path is applied
            version = pytesseract.get_tesseract_version()
            status_msg = f"Tesseract {version} found and configured."
            if from_manual_apply and self.root.winfo_exists():
                messagebox.showinfo("Tesseract Status", status_msg, parent=self.root)
            self.status_var.set(status_msg)
            logging.info(status_msg)
            return True
        except pytesseract.TesseractNotFoundError:
            err_msg_log = "TesseractNotFoundError received."
            user_msg_title = "Tesseract Not Found"
            current_effective_cmd = getattr(pytesseract, 'tesseract_cmd', 'tesseract')

            if self.tesseract_cmd_path and os.path.isfile(self.tesseract_cmd_path):
                user_msg_detail = (f"The Tesseract path is set to:\n'{self.tesseract_cmd_path}'\n\n"
                                    f"However, Pytesseract could not execute Tesseract. Common reasons:\n"
                                    f"1. Corrupted Tesseract installation (missing DLLs).\n"
                                    f"2. 'tessdata' folder not found by Tesseract.\n"
                                    f"3. Insufficient permissions.\n\n"
                                    f"Recommendations:\n"
                                    f"  a) Verify Tesseract works from command line: `\"{self.tesseract_cmd_path}\" --version`\n"
                                    f"  b) Ensure 'tessdata' folder is correctly placed.")
                err_msg_log += f" Path '{self.tesseract_cmd_path}' configured and exists. Effective cmd: '{current_effective_cmd}'."
            else:
                user_msg_detail = ("Pytesseract could not find Tesseract OCR.\n\n"
                                    "Please ensure Tesseract is installed and either:\n"
                                    "1. Added to system PATH, OR\n"
                                    "2. Full path to 'tesseract.exe' specified in Scraper settings.\n\n")
                if self.tesseract_cmd_path:
                    user_msg_detail += f"Specified path was:\n'{self.tesseract_cmd_path}'\nThis path seems invalid."
                    err_msg_log += f" Configured path '{self.tesseract_cmd_path}' invalid."
                else:
                    user_msg_detail += "No specific path provided; search in system PATH failed."
                err_msg_log += f" Effective cmd: '{current_effective_cmd}'."

            self.status_var.set("Tesseract not found/failed. Check Scraper settings/installation.")
            logging.error(err_msg_log)

            show_error_dialog = from_manual_apply or not self._initial_tesseract_error_shown_startup
            if show_error_dialog and self.root.winfo_exists():
                messagebox.showerror(user_msg_title, user_msg_detail, parent=self.root)
            if not from_manual_apply: self._initial_tesseract_error_shown_startup = True
            return False
        except Exception as e:
            err_msg = f"Unexpected error checking Tesseract: {str(e)}"
            self.status_var.set(f"Tesseract check error: {str(e)[:50]}...")
            logging.error(f"{err_msg}. Effective cmd: '{getattr(pytesseract, 'tesseract_cmd', 'tesseract')}'. Specified: '{self.tesseract_cmd_path}'.", exc_info=True)
            if from_manual_apply and self.root.winfo_exists():
                messagebox.showerror("Tesseract Error", err_msg, parent=self.root)
            return False

    def _create_main_widgets(self):
        # Main frame
        main_app_frame = ttk.Frame(self.root, padding="5")
        main_app_frame.pack(fill=tk.BOTH, expand=True)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_app_frame)
        
        # Create Scraper Tab
        self.scraper_tab_frame = ScraperTab(self.notebook, self)
        self.notebook.add(self.scraper_tab_frame, text="Scraper & Extractor")

        # Create Sorter Tab
        self.sorter_tab_frame = SorterTab(self.notebook, self)
        self.notebook.add(self.sorter_tab_frame, text="Visual Sorter & Commands")
        
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # Status Bar
        status_bar = ttk.Label(main_app_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5,2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0), padx=5)
        
        self.status_var.set("Ready.") # Initial ready state after loading

    def _on_close_app(self):
        # Check if scraper is capturing
        if hasattr(self.scraper_tab_frame, 'is_capturing') and self.scraper_tab_frame.is_capturing:
            if messagebox.askyesno("Capture in Progress", "Scraper is currently capturing. Exit anyway?", parent=self.root):
                self.scraper_tab_frame._stop_capture_flag_set() # Tell scraper to stop
            else:
                return # Don't close

        # Check if sorter AI model is loading
        if self.model_loading_thread_sorter and self.model_loading_thread_sorter.is_alive():
                if messagebox.askyesno("AI Model Loading", "Sorter AI model is loading. Exit anyway?", parent=self.root):
                    # Threads are daemonic, will exit with main. No specific stop needed here for that.
                    pass
                else:
                    return

        self._save_config(show_status=False) # Save final config
        self._save_character_db_global() # Save final DB
        if self.root.winfo_exists():
            self.root.destroy()

# --- ScraperTab Class ---
class ScraperTab(ttk.Frame):
    def __init__(self, parent_notebook: ttk.Notebook, main_app: MudaePowerToolsApp):
        super().__init__(parent_notebook)
        self.main_app = main_app # Reference to the main application instance

        # Scraper specific state
        self.screenshots: List[Image.Image] = []
        self.current_image_index: int = 0
        self.is_capturing: bool = False
        self.stop_capture_window: Optional[StopCaptureWindow] = None
        self.consecutive_similar_frames: int = 0
        self.last_captured_dhash: Optional[int] = None
        self.tk_image_preview: Optional[ImageTk.PhotoImage] = None

        # Tkinter variables for UI elements in this tab
        self.folder_path_var = tk.StringVar(value=self.main_app.output_folder)
        self.tesseract_path_var = tk.StringVar(value=self.main_app.tesseract_cmd_path)
        self.ocr_lang_var = tk.StringVar(value=self.main_app.ocr_language)
        self.image_format_var = tk.StringVar(value=self.main_app.output_image_format)
        self.jpeg_quality_var = tk.StringVar(value=str(self.main_app.jpeg_quality))

        self.region_info_var = tk.StringVar()
        self.next_button_info_var = tk.StringVar()
        self.char_name_info_var = tk.StringVar()
        self.show_name_info_var = tk.StringVar()
        self.image_info_var = tk.StringVar(value="No screenshots captured.")

        self._create_scraper_widgets()
        self._update_all_scraper_ui_elements() # Initial UI state

    def _create_scraper_widgets(self):
        # Main frame for this tab
        tab_main_frame = ttk.Frame(self, padding="10")
        tab_main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Configuration Frame ---
        config_frame = ttk.LabelFrame(tab_main_frame, text="Scraper Setup & Configuration")
        config_frame.pack(fill="x", pady=10)
        config_frame.columnconfigure(1, weight=1) # Make entry widgets expand

        # Output Folder
        ttk.Label(config_frame, text="Output Folder:", style="Header.TLabel").grid(row=0, column=0, padx=5, pady=3, sticky="w")
        folder_entry = ttk.Entry(config_frame, textvariable=self.folder_path_var, width=60)
        folder_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")
        browse_btn = ttk.Button(config_frame, text="Browse", command=self._browse_folder_scraper)
        browse_btn.grid(row=0, column=2, padx=5, pady=3)
        Tooltip(browse_btn, "Select the folder where extracted images will be saved.")

        # Tesseract Path
        ttk.Label(config_frame, text="Tesseract Path:", style="Header.TLabel").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        tesseract_entry = ttk.Entry(config_frame, textvariable=self.tesseract_path_var, width=60)
        tesseract_entry.grid(row=1, column=1, padx=5, pady=3, sticky="ew")
        tesseract_apply_btn = ttk.Button(config_frame, text="Apply & Test", command=self._apply_and_test_tesseract_path_scraper)
        tesseract_apply_btn.grid(row=1, column=2, padx=5, pady=3)
        Tooltip(tesseract_entry, "Optional: Full path to tesseract.exe. Leave blank to use system PATH.")
        Tooltip(tesseract_apply_btn, "Apply the Tesseract path and test its availability.")

        # OCR Language
        ttk.Label(config_frame, text="OCR Language(s):", style="Header.TLabel").grid(row=2, column=0, padx=5, pady=3, sticky="w")
        ocr_lang_entry = ttk.Entry(config_frame, textvariable=self.ocr_lang_var, width=15)
        ocr_lang_entry.grid(row=2, column=1, padx=5, pady=3, sticky="w")
        self.ocr_lang_var.trace_add("write", self._on_scraper_param_change)
        Tooltip(ocr_lang_entry, "Tesseract language codes (e.g., 'eng', 'jpn', 'eng+jpn').")

        # Output Image Format
        ttk.Label(config_frame, text="Save Image As:", style="Header.TLabel").grid(row=3, column=0, padx=5, pady=3, sticky="w")
        format_options = ["PNG", "JPG"] # Case matters for PIL
        format_menu = ttk.OptionMenu(config_frame, self.image_format_var, self.main_app.output_image_format, *format_options, command=self._on_image_format_change_scraper)
        format_menu.grid(row=3, column=1, padx=5, pady=3, sticky="w")

        self.jpeg_quality_label = ttk.Label(config_frame, text="JPEG Quality:", style="Header.TLabel")
        self.jpeg_quality_entry = ttk.Entry(config_frame, textvariable=self.jpeg_quality_var, width=5)
        self.jpeg_quality_var.trace_add("write", self._on_scraper_param_change)
        Tooltip(self.jpeg_quality_entry, "JPEG quality (1-100, higher is better).")
        # Visibility handled by _update_jpeg_quality_field_visibility_scraper

        # Advanced Settings Button
        adv_settings_btn = ttk.Button(config_frame, text="Advanced Scraper Settings...", command=self._open_scraper_advanced_settings)
        adv_settings_btn.grid(row=4, column=0, columnspan=3, pady=10)
        Tooltip(adv_settings_btn, "Configure dHash size, Hamming thresholds, and card detection parameters.")

        # Region Info Labels
        region_info_frame = ttk.Frame(config_frame)
        region_info_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10,0))
        region_info_frame.columnconfigure(1, weight=1)
        
        region_labels_texts = [
            ("List Region:", self.region_info_var, "list"),
            ("Next Button:", self.next_button_info_var, "next_button"),
            ("Char Name Area:", self.char_name_info_var, "char_name"),
            ("Show Name Area:", self.show_name_info_var, "show_name")
        ]
        for r_idx, (text, var, region_key) in enumerate(region_labels_texts):
            ttk.Label(region_info_frame, text=text, style="Header.TLabel").grid(row=r_idx, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(region_info_frame, textvariable=var).grid(row=r_idx, column=1, sticky="w", padx=5, pady=2)
            ttk.Button(region_info_frame, text="Clear", style="Small.TButton", 
                        command=lambda k=region_key: self._clear_selected_region_scraper(k)).grid(row=r_idx, column=2, padx=5, pady=2)
        
        # --- Control Buttons Frame ---
        controls_frame = ttk.Frame(tab_main_frame) # Use tab_main_frame
        controls_frame.pack(pady=10, fill="x")
        btn_container = ttk.Frame(controls_frame)
        btn_container.pack() # Center the buttons

        self.select_list_region_btn = ttk.Button(btn_container, text="1. Select List Region", command=lambda: self._select_generic_region_scraper("list"))
        self.select_list_region_btn.pack(side="left", padx=5, pady=5)
        self.select_next_btn_region_btn = ttk.Button(btn_container, text="2. Select Next Button", command=lambda: self._select_generic_region_scraper("next_button"))
        self.select_next_btn_region_btn.pack(side="left", padx=5, pady=5)
        
        sub_region_btn_container = ttk.Frame(controls_frame) # New container for optional buttons
        sub_region_btn_container.pack(pady=5)

        self.select_char_name_btn = ttk.Button(sub_region_btn_container, text="Select Char Name Area (Optional)", command=lambda: self._select_generic_region_scraper("char_name"))
        self.select_char_name_btn.pack(side="left", padx=5, pady=5)
        self.select_show_name_btn = ttk.Button(sub_region_btn_container, text="Select Show Name Area (Optional)", command=lambda: self._select_generic_region_scraper("show_name"))
        self.select_show_name_btn.pack(side="left", padx=5, pady=5)

        action_buttons_frame = ttk.Frame(tab_main_frame) # Use tab_main_frame
        action_buttons_frame.pack(pady=10, fill="x")
        action_btn_container = ttk.Frame(action_buttons_frame)
        action_btn_container.pack()

        self.capture_btn = ttk.Button(action_btn_container, text="3. Start Capture", command=self._start_capture_process_scraper)
        self.capture_btn.pack(side="left", padx=5, pady=5)
        self.process_btn = ttk.Button(action_btn_container, text="4. Extract Images", command=self._extract_images_from_screenshots_scraper)
        self.process_btn.pack(side="left", padx=5, pady=5)
        self.clear_ss_btn = ttk.Button(action_btn_container, text="Clear Screenshots", command=self._clear_screenshots_scraper)
        self.clear_ss_btn.pack(side="left", padx=5, pady=5)
        Tooltip(self.clear_ss_btn, "Clears all captured screenshots from memory.")

        # --- Preview Frame ---
        preview_outer_frame = ttk.LabelFrame(tab_main_frame, text="Screenshot Preview") # Use tab_main_frame
        preview_outer_frame.pack(fill="both", expand=True, pady=10)
        self.preview_canvas = tk.Canvas(preview_outer_frame, bg="gray20", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        self.preview_canvas.bind("<Configure>", self._on_preview_canvas_resize_scraper)

        nav_frame = ttk.Frame(preview_outer_frame)
        nav_frame.pack(fill="x", pady=5)
        self.prev_btn = ttk.Button(nav_frame, text="<< Previous", command=self._show_previous_image_scraper)
        self.prev_btn.pack(side="left", padx=10)
        ttk.Label(nav_frame, textvariable=self.image_info_var, anchor="center").pack(side="left", padx=10, fill="x", expand=True)
        self.next_btn = ttk.Button(nav_frame, text="Next >>", command=self._show_next_image_scraper)
        self.next_btn.pack(side="right", padx=10)

        Tooltip(self.select_list_region_btn, "Define the main rectangular area where Mudae character cards appear.")
        Tooltip(self.select_next_btn_region_btn, "Define the area of the 'next page' button in Discord for auto-scrolling.")
        Tooltip(self.select_char_name_btn, "Optional: Define where character names are on a card. Select relative to a card at screen's top-left.")
        Tooltip(self.select_show_name_btn, "Optional: Define where show names are on a card. Select relative to a card at screen's top-left.")
        Tooltip(self.capture_btn, "Start capturing screenshots. Requires List Region to be selected.")
        Tooltip(self.process_btn, "Extract individual character images from captured screenshots.")

    def _update_all_scraper_ui_elements(self):
        """Update all UI elements in the ScraperTab that reflect current state/config."""
        self.folder_path_var.set(self.main_app.output_folder)
        self.tesseract_path_var.set(self.main_app.tesseract_cmd_path)
        self.ocr_lang_var.set(self.main_app.ocr_language)
        self.image_format_var.set(self.main_app.output_image_format)
        self.jpeg_quality_var.set(str(self.main_app.jpeg_quality))
        
        self._update_jpeg_quality_field_visibility_scraper()
        self._update_region_info_labels_scraper()
        self._update_button_states_scraper()
        self._display_current_image_scraper()

    def _update_region_info_labels_scraper(self):
        self.region_info_var.set(self._get_region_text_scraper(self.main_app.capture_region))
        self.next_button_info_var.set(self._get_region_text_scraper(self.main_app.next_button_region))
        self.char_name_info_var.set(self._get_region_text_scraper(self.main_app.character_name_sub_region))
        self.show_name_info_var.set(self._get_region_text_scraper(self.main_app.show_name_sub_region))
        
    def _get_region_text_scraper(self, region: Optional[Tuple[int,int,int,int]]) -> str:
        if region:
            x1, y1, x2, y2 = region
            return f"{x2-x1}x{y2-y1} at ({x1},{y1})"
        return "Not Selected"

    def _update_button_states_scraper(self):
        list_region_selected = self.main_app.capture_region is not None
        screenshots_exist = bool(self.screenshots)

        self.capture_btn.config(state=tk.NORMAL if list_region_selected else tk.DISABLED)
        self.process_btn.config(state=tk.NORMAL if screenshots_exist else tk.DISABLED)
        self.clear_ss_btn.config(state=tk.NORMAL if screenshots_exist else tk.DISABLED)
        self.prev_btn.config(state=tk.NORMAL if screenshots_exist and len(self.screenshots) > 1 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if screenshots_exist and len(self.screenshots) > 1 else tk.DISABLED)

    def _browse_folder_scraper(self):
        initial_dir = self.main_app.output_folder if os.path.isdir(self.main_app.output_folder) else os.path.expanduser("~")
        folder_path = filedialog.askdirectory(title="Select Output Folder", initialdir=initial_dir, parent=self.main_app.root)
        if folder_path:
            self.main_app.output_folder = folder_path
            self.folder_path_var.set(folder_path)
            self.main_app.db_file_path = os.path.join(self.main_app.output_folder, DB_FILE_NAME) # Update global DB path
            self.main_app._load_character_db_global() # Reload DB from new location
            self.main_app.status_var.set(f"Output folder set to: {folder_path}")
            logging.info(f"Output folder changed to: {folder_path}")
            self.main_app._save_config()
            # Also update the SorterTab's default folder if it was using the old default
            if hasattr(self.main_app, 'sorter_tab_frame'):
                self.main_app.sorter_tab_frame.update_default_image_folder(folder_path)


    def _apply_and_test_tesseract_path_scraper(self):
        path_from_ui = self.tesseract_path_var.get().strip()
        self.main_app.tesseract_cmd_path = path_from_ui # Update main app's variable
        self.main_app._apply_tesseract_path_global(log_success=True) # Apply globally
        self.main_app._check_tesseract_availability_global(from_manual_apply=True) # Check globally
        self.main_app._save_config(show_status=True)

    def _on_image_format_change_scraper(self, *args):
        self.main_app.output_image_format = self.image_format_var.get()
        self._update_jpeg_quality_field_visibility_scraper()
        self.main_app._save_config(show_status=True)

    def _update_jpeg_quality_field_visibility_scraper(self):
        # Grid positions might need slight adjustment if layout changes
        if self.image_format_var.get() == "JPG":
            self.jpeg_quality_label.grid(row=3, column=1, padx=(100,5), pady=3, sticky="e")
            self.jpeg_quality_entry.grid(row=3, column=1, padx=(170,5), pady=3, sticky="e")
        else:
            self.jpeg_quality_label.grid_remove()
            self.jpeg_quality_entry.grid_remove()
            
    def _on_scraper_param_change(self, *args):
        """Generic handler for Scraper config changes that need saving."""
        try:
            jq_str = self.jpeg_quality_var.get()
            if jq_str: # Only validate if not empty
                jq = int(jq_str)
                if not (1 <= jq <= 100):
                    self.jpeg_quality_var.set(str(DEFAULT_JPEG_QUALITY)) # Reset to default
                    if self.main_app.root.winfo_exists(): 
                        messagebox.showwarning("Invalid Quality", "JPEG quality must be between 1 and 100.", parent=self.main_app.root)
                self.main_app.jpeg_quality = int(self.jpeg_quality_var.get())
        except ValueError: # If entry is not a valid int (e.g. empty or text)
            self.jpeg_quality_var.set(str(self.main_app.jpeg_quality)) # Revert to current app value
            pass # Don't show error for transient non-integer states during typing

        self.main_app.ocr_language = self.ocr_lang_var.get()
        # Potentially other params here
        self.main_app._save_config(show_status=True)

    def _open_scraper_advanced_settings(self):
        ScraperAdvancedSettingsDialog(self.main_app.root, self.main_app)

    def _clear_selected_region_scraper(self, region_type: str):
        region_name_map = {
            "list": "List Region", "next_button": "Next Button Region",
            "char_name": "Character Name Sub-Region", "show_name": "Show Name Sub-Region"
        }
        if messagebox.askyesno("Confirm Clear", f"Are you sure you want to clear the {region_name_map.get(region_type, 'region')} selection?", parent=self.main_app.root):
            if region_type == "list": self.main_app.capture_region = None
            elif region_type == "next_button": self.main_app.next_button_region = None
            elif region_type == "char_name": self.main_app.character_name_sub_region = None
            elif region_type == "show_name": self.main_app.show_name_sub_region = None
            
            self.main_app.status_var.set(f"{region_name_map.get(region_type, 'Region').split('(')[0].strip()} selection cleared.")
            logging.info(f"{region_type.replace('_', ' ').title()} selection cleared.")
            self._update_region_info_labels_scraper()
            self._update_button_states_scraper()
            self.main_app._save_config()

    def _select_generic_region_scraper(self, region_type: str):
        titles = {
            "list": "Select Mudae List Region",
            "next_button": "Select 'Next Page' Button Region",
            "char_name": "Select Character Name Area (Relative to Card)",
            "show_name": "Select Show Name Area (Relative to Card)"
        }
        title = titles.get(region_type, "Select Region")
        base_message = f"The main window will hide. {title}.\nPress ESC to cancel."
        
        info_message = base_message
        if region_type in ["char_name", "show_name"]:
            area_type = "Character Name" if region_type == "char_name" else "Show Name"
            info_message = (
                f"The selection window will appear. Please select the area for the {area_type}.\n\n"
                "IMPORTANT: Imagine a single character card is at the TOP-LEFT CORNER (0,0) "
                "of the selection screen. Draw your rectangle relative to that imagined card.\n\n"
                "This selected area will be applied to each extracted card image."
            )
        
        messagebox.showinfo(title, info_message, parent=self.main_app.root)
        self.main_app.root.withdraw() # Hide main app window
        
        selector = RegionSelector(parent=self.main_app.root, title=title)
        region = selector.run() # This blocks until selector is done

        if region:
            if region_type == "list": self.main_app.capture_region = region
            elif region_type == "next_button": self.main_app.next_button_region = region
            elif region_type == "char_name": self.main_app.character_name_sub_region = region
            elif region_type == "show_name": self.main_app.show_name_sub_region = region
            self.main_app.status_var.set(f"{title.split('(')[0].strip()} selected: {region}")
            logging.info(f"{region_type.replace('_', ' ').title()} region selected: {region}")
        else:
            self.main_app.status_var.set(f"{title.split('(')[0].strip()} selection cancelled.")
            logging.info(f"{region_type.replace('_', ' ').title()} selection cancelled.")

        self._update_region_info_labels_scraper()
        self._update_button_states_scraper()
        self.main_app._save_config() # Save config after region change

    def _start_capture_process_scraper(self):
        if not self.main_app.capture_region:
            messagebox.showwarning("No Region", "Please select the Mudae list region first.", parent=self.main_app.root)
            return
        if not self.main_app._check_tesseract_availability_global(from_manual_apply=False):
            if not messagebox.askyesno("Tesseract Issue", "Tesseract not properly configured. OCR for names/shows might fail.\n\nContinue capture anyway?", parent=self.main_app.root, icon='warning'):
                return

        self.screenshots = []
        self.current_image_index = 0
        self.last_captured_dhash = None
        self.consecutive_similar_frames = 0
        self._display_current_image_scraper()
        self.image_info_var.set("Preparing...")

        # Corrected default values for messagebox.askyesno
        auto_scroll_default_str = messagebox.YES if self.main_app.next_button_region else messagebox.NO
        auto_scroll = messagebox.askyesno(
            "Auto Scroll",
            "Enable automatic scrolling using the 'Next Button' region?\n"
            f"(Next Button region: {'Set' if self.main_app.next_button_region else 'Not Set - Manual scroll only'})",
            default="yes" if self.main_app.next_button_region else "no", parent=self.main_app.root
        )

        if auto_scroll and not self.main_app.next_button_region:
            messagebox.showinfo("Next Button Needed", "Auto-scroll cannot be enabled as 'Next Button' region is not selected.", parent=self.main_app.root)
            auto_scroll = False

        instructions = (
            "After clicking OK, switch to Discord and navigate to your Mudae list.\n"
            "The tool will begin capturing screenshots.\n"
            f"{'It will attempt to auto-scroll. ' if auto_scroll else 'Manually scroll through your list. '}"
            "A small 'Stop Capture' window will appear; use it or press ESC to stop."
        )
        messagebox.showinfo("Capture Instructions", instructions, parent=self.main_app.root)

        self.is_capturing = True
        # self.main_app.root.iconify() # Minimize main window - Removed to see if it affects stop window visibility
        logging.info("Creating StopCaptureWindow...")
        self.stop_capture_window = StopCaptureWindow(self.main_app.root, self._stop_capture_flag_set)
        logging.info("StopCaptureWindow created. Attempting to show and focus.")
        self.stop_capture_window.show()
        # Explicitly raise and focus the stop window
        if self.stop_capture_window.root.winfo_exists():
                self.stop_capture_window.root.lift()
                self.stop_capture_window.root.attributes('-topmost', True) # Ensure it stays on top
                self.stop_capture_window.root.after(10, lambda: self.stop_capture_window.root.attributes('-topmost', False)) # Remove topmost after a short delay
                self.stop_capture_window.root.focus_force()

        self.main_app.status_var.set("Capturing... (Use Stop window or ESC to halt)")
        self.main_app.root.after(2000, lambda: self._capture_loop_iteration_scraper(auto_scroll))


    def _stop_capture_flag_set(self):
        if self.is_capturing:
            self.is_capturing = False
            logging.info("Capture stop signal received.")
            # Finalize will be called by the loop checking self.is_capturing

    def _capture_loop_iteration_scraper(self, auto_scroll: bool):
        if not self.is_capturing:
            self._finalize_capture_scraper()
            return

        delay_ms = 750 if auto_scroll else 1500 # Shorter delay for auto, longer for manual

        try:
            x1, y1, x2, y2 = self.main_app.capture_region # type: ignore
            current_pil_image = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            current_dhash_val = dhash(current_pil_image, self.main_app.dhash_size)

            if current_dhash_val is None:
                logging.warning("Failed to calculate dHash. Skipping frame.")
                if self.is_capturing: self.main_app.root.after(delay_ms, lambda: self._capture_loop_iteration_scraper(auto_scroll))
                else: self._finalize_capture_scraper()
                return

            is_new_image = True
            dist = float('inf')
            if self.last_captured_dhash is not None:
                dist = hamming_distance(current_dhash_val, self.last_captured_dhash)
                if dist <= self.main_app.hamming_distance_new_screenshot:
                    self.consecutive_similar_frames += 1
                    is_new_image = False
                    # Use the constant for threshold
                    if self.consecutive_similar_frames > CONSECUTIVE_SIMILAR_FRAMES_THRESHOLD and auto_scroll: 
                        logging.info(f"Too many similar frames ({self.consecutive_similar_frames}) with auto-scroll, assuming end or stuck.")
                        self.is_capturing = False # Signal to stop
                else:
                    self.consecutive_similar_frames = 0
            else: # First image
                self.consecutive_similar_frames = 0

            if is_new_image or not self.screenshots:
                self.screenshots.append(current_pil_image)
                self.last_captured_dhash = current_dhash_val
                self.consecutive_similar_frames = 0 # Reset on new image
                logging.info(f"Captured screenshot #{len(self.screenshots)}. dHash: {current_dhash_val}, Dist from prev: {dist if dist != float('inf') else 'N/A'}")
                self.image_info_var.set(f"Captured: {len(self.screenshots)}")
                if self.main_app.root.winfo_exists(): self.main_app.root.update_idletasks()

            else: # Similar image
                logging.info(f"Skipped similar screenshot (dHash: {current_dhash_val}, dist: {dist}, similar frames: {self.consecutive_similar_frames}).")


            if auto_scroll and self.main_app.next_button_region and self.is_capturing:
                btn_x1, btn_y1, btn_x2, btn_y2 = self.main_app.next_button_region
                click_x, click_y = (btn_x1 + btn_x2) // 2, (btn_y1 + btn_y2) // 2
                
                current_mouse_pos = pyautogui.position()
                pyautogui.moveTo(click_x, click_y, duration=0.1)
                pyautogui.click()
                time.sleep(0.3) # Brief pause after click for page to load
                pyautogui.moveTo(current_mouse_pos[0], current_mouse_pos[1], duration=0.1) # Move mouse back
                logging.info(f"Auto-clicked next button at ({click_x}, {click_y})")

            if self.is_capturing:
                self.main_app.root.after(delay_ms, lambda: self._capture_loop_iteration_scraper(auto_scroll))
            else: # Capture was stopped (either by flag or consecutive similar frames)
                self._finalize_capture_scraper()

        except Exception as e:
            logging.error(f"Error during capture iteration: {e}", exc_info=True)
            if self.main_app.root.winfo_exists(): # Check if root still exists
                    messagebox.showerror("Capture Error", f"An error occurred: {e}", parent=self.main_app.root)
            self.is_capturing = False # Ensure capture stops on error
            self._finalize_capture_scraper()


    def _finalize_capture_scraper(self):
        if self.stop_capture_window:
            self.stop_capture_window.close_window()
            self.stop_capture_window = None

        if self.main_app.root.winfo_exists():
            self.main_app.root.deiconify() # Restore main window
            self.main_app.root.lift()
            self.main_app.root.focus_force()
            if self.screenshots:
                self.main_app.status_var.set(f"Capture finished. {len(self.screenshots)} screenshots taken.")
                self.current_image_index = 0 # Reset to first image for preview
                self._display_current_image_scraper()
            else:
                self.main_app.status_var.set("Capture finished. No new screenshots were taken.")
            self._update_button_states_scraper()
        logging.info("Capture process finalized.")

    def _clear_screenshots_scraper(self):
        if not self.screenshots:
            messagebox.showinfo("No Screenshots", "There are no screenshots to clear.", parent=self.main_app.root)
            return
        if messagebox.askyesno("Confirm Clear", f"Are you sure you want to clear all {len(self.screenshots)} captured screenshots from memory?", parent=self.main_app.root):
            self.screenshots = []
            self.current_image_index = 0
            self.last_captured_dhash = None
            self.consecutive_similar_frames = 0
            self._display_current_image_scraper() # Update preview to show no image
            self.main_app.status_var.set("Screenshots cleared.")
            self._update_button_states_scraper()
            logging.info("All screenshots cleared from memory.")

    def _on_preview_canvas_resize_scraper(self, event: Optional[tk.Event] = None):
        if self.main_app.root.winfo_exists() and self.preview_canvas.winfo_width() > 1 and self.preview_canvas.winfo_height() > 1:
            self._display_current_image_scraper()

    def _display_current_image_scraper(self):
        if not self.main_app.root.winfo_exists(): return

        if not self.screenshots or not (0 <= self.current_image_index < len(self.screenshots)):
            self.preview_canvas.delete("all")
            self.image_info_var.set("No images to display.")
            cw, ch = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
            if cw > 1 and ch > 1 :
                self.preview_canvas.create_text(cw//2, ch//2, text="No Image", fill="white", font=("Arial", 16))
            self.tk_image_preview = None # Clear reference
            if self.main_app.root.winfo_exists(): self._update_button_states_scraper()
            return

        pil_image = self.screenshots[self.current_image_index]
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1: # Canvas not ready
            if self.main_app.root.winfo_exists(): self.main_app.root.after(100, self._display_current_image_scraper)
            return

        img_width, img_height = pil_image.size
        # Calculate ratio to fit image within canvas while maintaining aspect ratio
        ratio = min(canvas_width / img_width, canvas_height / img_height) if img_width > 0 and img_height > 0 else 1
        ratio = max(0.01, ratio) # Ensure ratio is positive

        new_width = max(1, int(img_width * ratio))
        new_height = max(1, int(img_height * ratio))

        try:
            resized_img = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.tk_image_preview = ImageTk.PhotoImage(resized_img) # Keep reference
        except Exception as e:
            logging.error(f"Error resizing image for preview: {e}")
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(canvas_width//2, canvas_height//2, text="Error Displaying Image", fill="red")
            self.tk_image_preview = None
            return

        self.preview_canvas.delete("all") # Clear previous image
        # Calculate position to center the image on the canvas
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        self.preview_canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_image_preview)
        self.image_info_var.set(f"Screenshot {self.current_image_index + 1} of {len(self.screenshots)}")
        if self.main_app.root.winfo_exists(): self._update_button_states_scraper()

    def _show_previous_image_scraper(self):
        if self.screenshots and len(self.screenshots) > 1:
            self.current_image_index = (self.current_image_index - 1 + len(self.screenshots)) % len(self.screenshots)
            self._display_current_image_scraper()

    def _show_next_image_scraper(self):
        if self.screenshots and len(self.screenshots) > 1:
            self.current_image_index = (self.current_image_index + 1) % len(self.screenshots)
            self._display_current_image_scraper()

    def _ocr_sub_region_scraper(self, image_pil: Image.Image, sub_region: Optional[Tuple[int,int,int,int]]) -> Optional[str]:
        if not sub_region: return None
        try:
            img_width, img_height = image_pil.size
            x1, y1, x2, y2 = sub_region
            # Ensure crop coordinates are within image bounds
            crop_x1, crop_y1 = max(0, x1), max(0, y1)
            crop_x2, crop_y2 = min(img_width, x2), min(img_height, y2)

            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2: # Invalid crop region
                logging.warning(f"Sub-region {sub_region} is invalid or outside image bounds {image_pil.size}")
                return None

            cropped_image = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            text = pytesseract.image_to_string(cropped_image, lang=self.main_app.ocr_language, config=OCR_CONFIG_SPARSE).strip()
            text = re.sub(r'\s+', ' ', text) # Normalize whitespace
            return text if len(text) > 1 else None # Basic filter for very short/empty OCR results
        except pytesseract.TesseractError as te:
            logging.error(f"Tesseract OCR error in sub-region (lang: {self.main_app.ocr_language}): {te}")
            self.main_app.status_var.set(f"Tesseract OCR error: {str(te)[:50]}...")
        except Exception as e:
            logging.error(f"Error during sub-region OCR: {e}")
        return None

    def _ocr_default_card_regions_scraper(self, card_pil: Image.Image, region_type: str = "name") -> Optional[str]:
        """OCR default regions if specific sub-regions are not defined or fail."""
        try:
            height, width = card_pil.height, card_pil.width
            if region_type == "name": # Top 20% for name
                crop_box = (0, 0, width, int(height * 0.20))
            elif region_type == "show": # Bottom 20% for show
                crop_box = (0, int(height * 0.80), width, height)
            else: return None

            region_img = card_pil.crop(crop_box)
            text = pytesseract.image_to_string(region_img, lang=self.main_app.ocr_language, config=OCR_CONFIG_SPARSE).strip()
            text = re.sub(r'\s+', ' ', text)
            return text if len(text) > 2 else None # Slightly more stringent filter for defaults
        except pytesseract.TesseractError as te:
            logging.error(f"Tesseract OCR error in default region (lang: {self.main_app.ocr_language}, type: {region_type}): {te}")
            self.main_app.status_var.set(f"Tesseract OCR (default) error: {str(te)[:50]}...")
        except Exception as e:
            logging.error(f"Error in default OCR for {region_type}: {e}")
        return None

    def _detect_character_cards_opencv_scraper(self, screenshot_cv: np.ndarray) -> List[np.ndarray]:
        """Detects character cards in a screenshot using OpenCV."""
        cards = []
        if screenshot_cv is None or screenshot_cv.size == 0:
            logging.warning("Received empty image for card detection.")
            return cards

        gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Small blur
        # Adjust Canny thresholds if needed, these are common starting points
        edges = cv2.Canny(blurred, 50, 150) 
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logging.debug(f"Found {len(contours)} initial contours.") # Changed to debug for less noise

        for contour in contours:
            area = cv2.contourArea(contour)
            # The area filter is now less strict as per original defaults,
            # but pixel count will be the main decider later.
            if area < self.main_app.min_card_area: 
                # logging.debug(f"Skipping contour with area {area} (min required: {self.main_app.min_card_area})")
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                # logging.debug("Skipping contour with zero width or height.")
                continue # Avoid division by zero
            aspect_ratio = float(h) / w

            # Aspect ratio filter is also less strict.
            if not (self.main_app.min_card_aspect_ratio <= aspect_ratio <= self.main_app.max_card_aspect_ratio):
                # logging.debug(f"Skipping contour with aspect ratio {aspect_ratio:.2f} (required: [{self.main_app.min_card_aspect_ratio:.1f}-{self.main_app.max_card_aspect_ratio:.1f}])")
                continue
            
            # Extract the card image (no padding, direct crop from bounding box)
            # Ensure coordinates are within screenshot bounds
            x_c, y_c = max(0, x), max(0, y)
            w_c = min(screenshot_cv.shape[1] - x_c, w) # Adjust width if it goes out of bounds
            h_c = min(screenshot_cv.shape[0] - y_c, h) # Adjust height

            if w_c > 10 and h_c > 10: # Basic check for valid dimensions
                card_img_cv = screenshot_cv[y_c : y_c + h_c, x_c : x_c + w_c]
                cards.append(card_img_cv)
            # else:
                # logging.debug(f"Skipping contour with small dimensions {w_c}x{h_c}.")
        
        logging.info(f"Detected {len(cards)} potential cards meeting basic criteria.")
        return cards

    def _extract_images_from_screenshots_scraper(self):
        if not self.screenshots:
            messagebox.showwarning("No Screenshots", "No screenshots available to process.", parent=self.main_app.root)
            return
        if not self.main_app._check_tesseract_availability_global(from_manual_apply=False):
            if not messagebox.askyesno("Tesseract Issue", "Tesseract not properly configured. OCR for names/shows might fail.\n\nContinue extraction anyway?", parent=self.main_app.root, icon='warning'):
                return

        os.makedirs(self.main_app.output_folder, exist_ok=True)
        self.main_app.status_var.set("Starting image extraction...")

        prog_dialog = ProgressDialog(self.main_app.root, title="Extracting Images", total_items=len(self.screenshots))
        extracted_count = 0
        processed_card_hashes: Dict[int, str] = {}  # Store dHash of SAVED cards to avoid duplicates

        for i, screenshot_pil in enumerate(self.screenshots): # Loop through each screenshot
            if prog_dialog.cancelled:
                logging.info("Extraction cancelled by user via progress dialog.")
                break
            prog_dialog.update_progress(i, f"Screenshot {i + 1}/{len(self.screenshots)}: Detecting cards...")

            screenshot_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
            detected_cards_cv = self._detect_character_cards_opencv_scraper(screenshot_cv)
            
            card_pil_to_process: Optional[Image.Image] = None # Initialize

            if detected_cards_cv:
                prog_dialog.set_phase(f"SS {i + 1}: Found {len(detected_cards_cv)} cards. Identifying best card by pixel count...")
                current_best_card_pil_auto: Optional[Image.Image] = None
                current_max_pixel_count_auto: int = 0

                for card_cv_candidate in detected_cards_cv:
                    if prog_dialog.cancelled: break 
                    try:
                        card_pil_candidate = Image.fromarray(cv2.cvtColor(card_cv_candidate, cv2.COLOR_BGR2RGB))
                        pixel_count = card_pil_candidate.width * card_pil_candidate.height
                        if pixel_count > current_max_pixel_count_auto:
                            current_max_pixel_count_auto = pixel_count
                            current_best_card_pil_auto = card_pil_candidate
                    except Exception as e_candidate:
                        logging.error(f"Error evaluating card candidate in SS {i + 1}: {e_candidate}", exc_info=True)
                if prog_dialog.cancelled: break
                
                if current_best_card_pil_auto:
                    card_pil_to_process = current_best_card_pil_auto
                    prog_dialog.set_phase(f"SS {i + 1}: Processing best auto-detected card ({card_pil_to_process.width}x{card_pil_to_process.height}px)...")

            if not card_pil_to_process: # Automatic detection failed or no cards found
                logging.info(f"Automatic card detection failed for SS {i + 1}. Prompting for manual selection.")
                prog_dialog.set_phase(f"SS {i + 1}: Auto-detect failed. Waiting for manual card selection...")
                
                if self.main_app.root.state() == 'iconic': # Ensure main window is visible for modal
                    self.main_app.root.deiconify()
                
                manual_selector = ManualCardSelector(self.main_app.root, screenshot_pil, 
                                                     title=f"Manually Select Card in Screenshot {i+1}")
                manually_cropped_pil = manual_selector.run() # This blocks

                if manually_cropped_pil:
                    card_pil_to_process = manually_cropped_pil
                    logging.info(f"Card manually selected for SS {i + 1} ({card_pil_to_process.width}x{card_pil_to_process.height}px).")
                    prog_dialog.set_phase(f"SS {i + 1}: Processing manually selected card...")
                else:
                    logging.info(f"Manual card selection cancelled or failed for SS {i + 1}. Skipping this screenshot.")
                    prog_dialog.set_phase(f"SS {i + 1}: Manual selection skipped.")
                    continue # Skip to the next screenshot
            
            if not card_pil_to_process:
                logging.error(f"SS {i+1}: No card to process after auto/manual selection. This indicates an issue.")
                continue

            # --- Process the selected card (either auto-best or manual) ---
            card_dhash_val = dhash(card_pil_to_process, self.main_app.dhash_size)
            if card_dhash_val is not None:
                is_duplicate = False
                for existing_hash, filename_of_existing in processed_card_hashes.items():
                    if hamming_distance(card_dhash_val, existing_hash) < self.main_app.hamming_distance_duplicate_card:
                        is_duplicate = True
                        logging.info(
                            f"Card from SS {i+1} (dHash {card_dhash_val}, {card_pil_to_process.width}x{card_pil_to_process.height}px) "
                            f"is a duplicate of '{filename_of_existing}' (dHash {existing_hash}). Skipping."
                        )
                        break
                if is_duplicate:
                    continue 
            else: 
                logging.warning(f"Failed to calculate dHash for the card in SS {i + 1}. Skipping duplicate check, will attempt to save.")

            try:
                char_name_text, show_name_text = "unknown_char", UNKNOWN_SHOW_SORTER
                if self.main_app.character_name_sub_region:
                    ocr_char = self._ocr_sub_region_scraper(card_pil_to_process, self.main_app.character_name_sub_region)
                    if ocr_char: char_name_text = ocr_char
                if self.main_app.show_name_sub_region:
                    ocr_show = self._ocr_sub_region_scraper(card_pil_to_process, self.main_app.show_name_sub_region)
                    if ocr_show: show_name_text = ocr_show
                
                if char_name_text == "unknown_char" and not self.main_app.character_name_sub_region:
                    ocr_char_default = self._ocr_default_card_regions_scraper(card_pil_to_process, "name")
                    if ocr_char_default: char_name_text = ocr_char_default
                if show_name_text == UNKNOWN_SHOW_SORTER and not self.main_app.show_name_sub_region:
                    ocr_show_default = self._ocr_default_card_regions_scraper(card_pil_to_process, "show")
                    if ocr_show_default: show_name_text = ocr_show_default

                sane_char_name = sanitize_filename(char_name_text)
                sane_show_name = sanitize_filename(show_name_text)

                base_filename = f"{sane_show_name}_{sane_char_name}" if sane_show_name != UNKNOWN_SHOW_SORTER else sane_char_name
                file_extension = ".jpg" if self.main_app.output_image_format == "JPG" else ".png"
                filename = f"{base_filename}{file_extension}"
                filepath = os.path.join(self.main_app.output_folder, filename)

                counter = 1
                while os.path.exists(filepath):
                    filename = f"{base_filename}_{counter}{file_extension}"
                    filepath = os.path.join(self.main_app.output_folder, filename)
                    counter += 1
                
                save_params = {}
                img_to_save = card_pil_to_process
                if self.main_app.output_image_format == "JPG":
                    save_params['format'] = 'JPEG'
                    save_params['quality'] = self.main_app.jpeg_quality
                    if img_to_save.mode == 'RGBA': 
                        img_to_save = img_to_save.convert('RGB')
                else: 
                    save_params['format'] = 'PNG'

                img_to_save.save(filepath, **save_params)
                extracted_count += 1
                if card_dhash_val is not None: 
                    processed_card_hashes[card_dhash_val] = filename

                prog_dialog.update_progress(i, f"SS {i + 1}: Saved {filename} ({card_pil_to_process.width}x{card_pil_to_process.height}px)")

                if sane_char_name != "unknown_char" and sane_show_name != UNKNOWN_SHOW_SORTER:
                    if sane_char_name not in self.main_app.character_show_db or \
                       self.main_app.character_show_db.get(sane_char_name, UNKNOWN_SHOW_SORTER) == UNKNOWN_SHOW_SORTER:
                        self.main_app.character_show_db[sane_char_name] = sane_show_name
                        logging.info(f"DB: Added/Updated '{sane_char_name}' -> '{sane_show_name}'")
            
            except Exception as e_process_card:
                logging.error(f"Error processing/saving card from SS {i + 1}: {e_process_card}", exc_info=True)
                prog_dialog.update_progress(i, f"SS {i + 1}: Error saving card")
                time.sleep(0.1) 
        # End of loop for screenshots

        prog_dialog.close_dialog()
        self.main_app._save_character_db_global()

        msg = f"Extraction complete! Saved {extracted_count} images to:\n{self.main_app.output_folder}" if extracted_count > 0 else "Extraction complete. No new images extracted."
        messagebox.showinfo("Extraction Complete", msg, parent=self.main_app.root)
        self.main_app.status_var.set(f"Saved {extracted_count} images." if extracted_count > 0 else "No new images extracted.")
        logging.info(f"Image extraction finished. Total saved: {extracted_count}")


# --- SorterTab Class ---
class SorterTab(ttk.Frame):
    def __init__(self, parent_notebook: ttk.Notebook, main_app: MudaePowerToolsApp):
        super().__init__(parent_notebook)
        self.main_app = main_app

        # Sorter specific state
        self.character_data_sorter: Dict[str, Dict[str, Any]] = {} # {'char_name': {'image_path': ..., 'ai_features': ..., 'color_features': ...}}
        
        # Tkinter variables for UI elements in this tab
        self.sorter_image_folder_var = tk.StringVar(value=self.main_app.output_folder) # Default to scraper output
        self.first_character_var_sorter = tk.StringVar(value="")
        self.sort_within_groups_var_sorter = tk.BooleanVar(value=True)
        self.sort_groups_by_size_var_sorter = tk.BooleanVar(value=True)
        self.cmd_type_var_sorter = tk.StringVar(value="sm") # Default to $sm

        self._create_sorter_widgets()
        self._update_sorter_ui_elements() # Initial UI state

        # Check for PyTorch and Scipy availability and inform user
        if not PYTORCH_AVAILABLE:
            self.main_app.status_var.set("Sorter Warning: PyTorch not found. AI visual sorting disabled.")
            messagebox.showwarning("Sorter Dependency Missing",
                                    "PyTorch (torch and torchvision) is not installed.\n"
                                    "AI-based visual similarity sorting will be disabled.\n"
                                    "Sorting will rely on color features only.",
                                    parent=self.main_app.root)
        if not SCIPY_AVAILABLE:
            self.main_app.status_var.set("Sorter Warning: SciPy not found. Visual sorting might be impaired.")
            messagebox.showwarning("Sorter Dependency Missing",
                                    "SciPy is not installed.\n"
                                    "Visual similarity calculations might be impaired or less accurate.",
                                    parent=self.main_app.root)


    def _create_sorter_widgets(self):
        tab_main_frame = ttk.Frame(self, padding="10")
        tab_main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Title ---
        title_label = ttk.Label(tab_main_frame, text="Mudae Visual Similarity Sorter & Command Generator", style="Title.TLabel")
        title_label.pack(pady=(0, 15))

        # --- Folder Selection ---
        folder_frame = ttk.Frame(tab_main_frame)
        folder_frame.pack(fill=tk.X, pady=5)
        ttk.Label(folder_frame, text="Image Folder:", style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 5))
        folder_entry = ttk.Entry(folder_frame, textvariable=self.sorter_image_folder_var, width=60)
        folder_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(folder_frame, text="Browse", command=self._browse_folder_sorter).pack(side=tk.LEFT, padx=5)
        Tooltip(folder_entry, "Folder containing character images (e.g., from the Scraper tab).")

        # --- First Character (FM) ---
        fm_frame = ttk.Frame(tab_main_frame)
        fm_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fm_frame, text="First Character (Optional):", style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 5))
        fm_entry = ttk.Entry(fm_frame, textvariable=self.first_character_var_sorter, width=40)
        fm_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        Tooltip(fm_entry, "If specified, this character and their show will be sorted first.")

        # --- Sorting Options ---
        options_frame = ttk.LabelFrame(tab_main_frame, text="Sorting Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)

        cb_visual = ttk.Checkbutton(options_frame, text="Sort by visual similarity within each show", variable=self.sort_within_groups_var_sorter)
        cb_visual.pack(anchor=tk.W, pady=2)
        Tooltip(cb_visual, "If checked, characters within the same show are sorted by visual similarity. Otherwise, original order (e.g., alphabetical by filename) is kept within shows.")
        
        cb_size = ttk.Checkbutton(options_frame, text="Sort shows by number of characters (largest first)", variable=self.sort_groups_by_size_var_sorter)
        cb_size.pack(anchor=tk.W, pady=2)
        Tooltip(cb_size, "If checked, show groups are sorted by the number of characters they contain (descending).")

        cmd_type_frame = ttk.Frame(options_frame)
        cmd_type_frame.pack(fill=tk.X, pady=(8,0))
        ttk.Label(cmd_type_frame, text="Command Type:", style="Header.TLabel").pack(side=tk.LEFT, padx=(0,10))
        rb_sm = ttk.Radiobutton(cmd_type_frame, text="$sm (Sort Marriage)", variable=self.cmd_type_var_sorter, value="sm")
        rb_sm.pack(side=tk.LEFT, padx=5)
        Tooltip(rb_sm, "Generates $sm commands. Sorts your entire list based on the provided order.")
        rb_smpos = ttk.Radiobutton(cmd_type_frame, text="$smpos (Sort from Position)", variable=self.cmd_type_var_sorter, value="smpos")
        rb_smpos.pack(side=tk.LEFT, padx=5)
        Tooltip(rb_smpos, "Generates $smpos commands. Sorts characters starting from the position of the last character in the previous $smpos command batch.")
        
        # --- Action Buttons ---
        action_button_frame = ttk.Frame(tab_main_frame)
        action_button_frame.pack(pady=10)
        
        self.load_analyze_btn = ttk.Button(action_button_frame, text="Load & Analyze Images", command=self._load_and_analyze_action_sorter)
        self.load_analyze_btn.pack(side=tk.LEFT, padx=10)
        Tooltip(self.load_analyze_btn, "Loads images from the selected folder, extracts features, and identifies shows.")

        self.preview_sort_btn = ttk.Button(action_button_frame, text="Preview Sort", command=self._preview_sort_action_sorter, state=tk.DISABLED)
        self.preview_sort_btn.pack(side=tk.LEFT, padx=10)
        Tooltip(self.preview_sort_btn, "Shows a preview of the sorted characters with thumbnails. Requires images to be loaded and analyzed first.")

        self.generate_cmd_btn = ttk.Button(action_button_frame, text="Generate Sort Command", command=self._generate_commands_action_sorter, state=tk.DISABLED)
        self.generate_cmd_btn.pack(side=tk.LEFT, padx=10)
        Tooltip(self.generate_cmd_btn, "Generates the Mudae sort commands based on the current settings and analyzed data.")
        
        # --- Output Area ---
        output_frame = ttk.LabelFrame(tab_main_frame, text="Sort Command (Copy & Paste to Discord)", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        scroll = ttk.Scrollbar(output_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text_area_sorter = tk.Text(output_frame, wrap=tk.WORD, yscrollcommand=scroll.set, height=10, relief=tk.SOLID, borderwidth=1, font=("Courier New", 9))
        self.result_text_area_sorter.pack(fill=tk.BOTH, expand=True, padx=(0,5))
        scroll.config(command=self.result_text_area_sorter.yview)

    def _update_sorter_ui_elements(self):
        # Update states of buttons etc. based on Sorter's internal state
        # Check if the necessary session data exists for enabling buttons
        data_analyzed_for_session = (
            hasattr(self, 'processed_char_names_sorter_session') and
            self.processed_char_names_sorter_session is not None
        )
        self.preview_sort_btn.config(state=tk.NORMAL if data_analyzed_for_session else tk.DISABLED)
        self.generate_cmd_btn.config(state=tk.NORMAL if data_analyzed_for_session else tk.DISABLED)
        
        # Update image folder path if main app's output folder changed
        current_main_output_folder = self.main_app.output_folder
        if self.sorter_image_folder_var.get() != current_main_output_folder:
                # If the sorter's folder is currently empty or was the one previously set by scraper default, update it.
            if not self.sorter_image_folder_var.get() or \
                self.sorter_image_folder_var.get() == getattr(self, '_last_known_scraper_output_folder', self.main_app.output_folder): # Check against current default too
                self.sorter_image_folder_var.set(current_main_output_folder)
        
        # Store the current main output folder as the last known default from scraper for next check
        setattr(self, '_last_known_scraper_output_folder', current_main_output_folder)


    def _browse_folder_sorter(self):
        initial_dir = self.sorter_image_folder_var.get() if os.path.isdir(self.sorter_image_folder_var.get()) else self.main_app.output_folder
        folder_selected = filedialog.askdirectory(title="Select Folder with Character Images", initialdir=initial_dir, parent=self.main_app.root)
        if folder_selected:
            self.sorter_image_folder_var.set(folder_selected)
            self.main_app.status_var.set(f"Sorter image folder set: {folder_selected}")
            # Clear previous sorter data if folder changes
            self.character_data_sorter.clear()
            # Clear session data as well
            if hasattr(self, 'processed_char_names_sorter_session'): del self.processed_char_names_sorter_session
            if hasattr(self, 'char_to_show_map_sorter_session'): del self.char_to_show_map_sorter_session
            if hasattr(self, 'char_to_path_map_sorter_session'): del self.char_to_path_map_sorter_session
            if hasattr(self, 'ai_features_list_sorter_session'): del self.ai_features_list_sorter_session
            if hasattr(self, 'color_features_list_sorter_session'): del self.color_features_list_sorter_session
            self._update_sorter_ui_elements()

    def _setup_ai_model_sorter(self):
        """Initializes and loads the AI model in a separate thread for Sorter."""
        if not PYTORCH_AVAILABLE:
            self.main_app.status_var.set("PyTorch not available. AI features disabled for Sorter.")
            self.main_app.ai_model_sorter = None
            return

        self.main_app.status_var.set("Loading Sorter AI model...")
        self.main_app.ai_device_sorter = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.main_app.model_loading_thread_sorter = threading.Thread(target=self._load_model_worker_sorter, daemon=True)
        self.main_app.model_loading_thread_sorter.start()
        self.main_app.root.after(100, self._check_model_loading_status_sorter)

    def _load_model_worker_sorter(self):
        """Worker function to load the AI model for Sorter."""
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.main_app.ai_model_sorter = torch.nn.Sequential(*list(model.children())[:-1]) # Remove classifier
            self.main_app.ai_model_sorter = self.main_app.ai_model_sorter.to(self.main_app.ai_device_sorter) # type: ignore
            self.main_app.ai_model_sorter.eval()

            self.main_app.ai_transform_sorter = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.main_app.ai_feature_size_sorter = DEFAULT_FEATURE_SIZE_SORTER
            self.main_app.model_queue_sorter.put(("success", f"EfficientNet_B0 loaded ({self.main_app.ai_device_sorter})."))
        except Exception as e:
            logging.error(f"Failed to load Sorter AI model: {e}", exc_info=True)
            self.main_app.model_queue_sorter.put(("error", f"Failed to load Sorter AI model: {e}"))

    def _check_model_loading_status_sorter(self):
        """Checks queue for Sorter AI model loading status."""
        try:
            status_type, message = self.main_app.model_queue_sorter.get_nowait()
            if status_type == "success":
                self.main_app.status_var.set(f"Sorter: Ready - {message}")
            else: # error
                self.main_app.status_var.set(f"Sorter: {message}")
                if self.main_app.root.winfo_exists():
                    messagebox.showwarning("Sorter AI Model Error", f"{message}\n\nVisual sorting will rely on color features only.", parent=self.main_app.root)
                self.main_app.ai_model_sorter = None # Ensure model is None
        except queue.Empty:
            if self.main_app.model_loading_thread_sorter and self.main_app.model_loading_thread_sorter.is_alive():
                self.main_app.root.after(100, self._check_model_loading_status_sorter)
            elif not self.main_app.ai_model_sorter and PYTORCH_AVAILABLE: # Thread finished but model not set
                if not self.main_app.status_var.get().startswith("Sorter: Failed to load AI model"):
                        self.main_app.status_var.set("Sorter: AI model loading failed. Using color features.")
        except Exception as e:
            self.main_app.status_var.set(f"Sorter: Error checking model status: {e}")
            self.main_app.ai_model_sorter = None

    def _extract_ai_features_sorter(self, image_path: str) -> Optional[np.ndarray]:
        if not self.main_app.ai_model_sorter or not self.main_app.ai_transform_sorter or not self.main_app.ai_device_sorter or not PYTORCH_AVAILABLE:
            return None
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.main_app.ai_transform_sorter(img).unsqueeze(0).to(self.main_app.ai_device_sorter)
            with torch.no_grad():
                features = self.main_app.ai_model_sorter(img_tensor) # type: ignore
            
            features_np = features.squeeze().cpu().numpy() # type: ignore
            
            # Ensure consistent feature size (pad or truncate)
            current_size = features_np.size
            target_size = self.main_app.ai_feature_size_sorter
            if current_size != target_size:
                # logging.warning(f"AI feature size mismatch for {os.path.basename(image_path)}. Expected {target_size}, got {current_size}. Adjusting.")
                if current_size < target_size:
                    padded = np.zeros(target_size)
                    padded[:current_size] = features_np
                    features_np = padded
                else:
                    features_np = features_np[:target_size]
            return features_np
        except Exception as e:
            logging.error(f"Error extracting AI features from {os.path.basename(image_path)}: {e}", exc_info=False) # Less verbose for individual image errors
            return None # Return None, will be replaced by zeros later if needed

    def _extract_color_features_sorter(self, image_path: str) -> np.ndarray:
        try:
            img = Image.open(image_path)
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            img_array = np.array(img)

            if len(img_array.shape) == 2: # Grayscale
                img_array = np.stack((img_array,) * 3, axis=-1)
            if img_array.shape[2] == 4: # RGBA
                img_array = img_array[:, :, :3] # Drop alpha
            
            if img_array.shape[2] != 3:
                logging.warning(f"Unexpected image format for color features in {os.path.basename(image_path)}")
                return np.zeros(3) # Default for unexpected format
            return np.mean(img_array, axis=(0, 1))
        except Exception as e:
            logging.error(f"Error extracting color features from {os.path.basename(image_path)}: {e}")
            return np.zeros(3) # Default on error

    def _identify_character_show_sorter(self, character_name: str) -> str:
        """Identifies show for a character using global DB, then filename patterns."""
        # 1. Check global DB (case-sensitive, then case-insensitive)
        if character_name in self.main_app.character_show_db:
            return self.main_app.character_show_db[character_name]
        
        char_lower = character_name.lower()
        for db_name, show in self.main_app.character_show_db.items():
            if char_lower == db_name.lower():
                # Optionally update DB with original casing if preferred, but usually scraper handles this.
                # self.main_app.character_show_db[character_name] = show 
                return show

        # 2. Pattern from filename: "Show Name_Character Name" (common from scraper)
        # This assumes character_name is derived from filename (without extension)
        if '_' in character_name:
            parts = character_name.split('_', 1)
            if len(parts) == 2:
                potential_show_name = parts[0].strip()
                # Heuristic: if potential_show_name is in known shows, it's likely correct
                # Or if it's not "unknown_show" and seems like a plausible show name (e.g., more than one word or capitalized)
                # This is a fallback, DB from scraper is primary.
                # For now, let's assume if DB doesn't have it, and it's "Show_Char", then "Show" is the show.
                # This is simpler than the original sorter's "Show Name - Character Name" from filename.
                if potential_show_name and potential_show_name != UNKNOWN_SHOW_SORTER:
                    # Don't automatically add this to the DB from sorter side to avoid conflicts with scraper.
                    # Sorter uses this for grouping if DB lookup fails.
                    return potential_show_name

        # 3. Original sorter's pattern: "Show Name - Character Name" (if char_name itself contains this)
        if ' - ' in character_name:
            parts = character_name.split(' - ', 1)
            show_name = parts[0].strip()
            if show_name:
                return show_name
        
        return UNKNOWN_SHOW_SORTER

    def _scan_and_extract_features_sorter(self, folder_path: str) -> Tuple[List[str], Dict[str, str], Dict[str, str], List[Optional[np.ndarray]], List[np.ndarray]]:
        """Scans folder, identifies shows, extracts features for Sorter."""
        self.main_app.status_var.set("Sorter: Scanning images...")
        if not self.main_app.root.winfo_exists(): return [], {}, {}, [], []
        self.main_app.root.update_idletasks()

        image_files = [f for f in os.listdir(folder_path)
                            if os.path.isfile(os.path.join(folder_path, f)) and
                            os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS_SORTER]

        if not image_files:
            messagebox.showerror("Error", "No valid image files found in the selected folder.", parent=self.main_app.root)
            self.main_app.status_var.set("Sorter: No images found.")
            return [], {}, {}, [], []

        prog_dialog = ProgressDialog(self.main_app.root, title="Sorter: Processing Images", total_items=len(image_files) * 2) # 2 passes
        current_progress = 0

        # Pass 1: Identify shows and store paths
        prog_dialog.set_phase("Phase 1/2: Identifying shows & indexing...")
        raw_character_names: List[str] = []
        character_to_show_map: Dict[str, str] = {} # {'char_name_from_file': 'show_name'}
        character_to_path_map: Dict[str, str] = {} # {'char_name_from_file': 'full_path'}

        for i, img_file in enumerate(image_files):
            if prog_dialog.cancelled: break
            # Character name from filename (strip extension, replace underscore with space for display/matching)
            char_name_from_file = os.path.splitext(img_file)[0] # Keep underscores for _identify_character_show_sorter logic
            
            raw_character_names.append(char_name_from_file)
            image_full_path = os.path.join(folder_path, img_file)
            character_to_path_map[char_name_from_file] = image_full_path
            
            current_progress +=1
            prog_dialog.update_progress(current_progress, f"Identifying show for: {char_name_from_file.replace('_', ' ')}")
            
            # Use sorter's identification logic (relies on global DB first)
            show_name = self._identify_character_show_sorter(char_name_from_file)
            character_to_show_map[char_name_from_file] = show_name
        if prog_dialog.cancelled: prog_dialog.close_dialog(); return [], {}, {}, [], []


        # Pass 2: Extract features
        prog_dialog.set_phase("Phase 2/2: Extracting visual features...")
        ai_features_list: List[Optional[np.ndarray]] = [] # Can contain None initially
        color_features_list: List[np.ndarray] = []
        processed_character_names: List[str] = [] # Names for which features were processed

        first_char_input = self.first_character_var_sorter.get().strip()
        first_character_found_in_files = False
        actual_first_character_name_from_file: Optional[str] = None

        for i, char_name in enumerate(raw_character_names): # Use original list of names from files
            if prog_dialog.cancelled: break
            current_progress +=1
            prog_dialog.update_progress(current_progress, f"Extracting features for: {char_name.replace('_', ' ')}")
            
            img_path = character_to_path_map[char_name]
            
            ai_feat = self._extract_ai_features_sorter(img_path) # Returns Optional[np.ndarray]
            color_feat = self._extract_color_features_sorter(img_path) # Returns np.ndarray (zeros on error)

            ai_features_list.append(ai_feat) # Keep None if AI fails for this image
            color_features_list.append(color_feat)
            processed_character_names.append(char_name)
            
            # Store in Sorter's local character_data for this session
            self.character_data_sorter[char_name] = {
                'image_path': img_path,
                'ai_features': ai_feat, # Store None if failed
                'color_features': color_feat
            }

            # Match first character (case-insensitive, comparing input with name from file)
            # The name from file might have underscores, input usually has spaces.
            if first_char_input and \
                (char_name.lower() == first_char_input.lower() or \
                char_name.replace('_', ' ').lower() == first_char_input.lower()) and \
                not first_character_found_in_files:
                first_character_found_in_files = True
                actual_first_character_name_from_file = char_name # Store the exact name from file
        
        if prog_dialog.cancelled: prog_dialog.close_dialog(); return [], {}, {}, [], []
        prog_dialog.close_dialog()

        if not processed_character_names:
            messagebox.showerror("Error", "No features could be extracted from any images.", parent=self.main_app.root)
            self.main_app.status_var.set("Sorter: Feature extraction failed.")
            return [], {}, {}, [], []
        
        if first_char_input and not first_character_found_in_files:
            messagebox.showwarning("Warning", f"The specified first character '{first_char_input}' was not found. Sorting will proceed without a specific start.", parent=self.main_app.root)
            self.first_character_var_sorter.set("") # Clear if not found
        elif first_character_found_in_files and actual_first_character_name_from_file:
            # Update var to the exact name from file (which might have underscores)
            # This ensures consistency when used later in _calculate_sorted_character_list
            self.first_character_var_sorter.set(actual_first_character_name_from_file)

        self.main_app.status_var.set(f"Sorter: Processed {len(processed_character_names)} characters.")
        # Note: Sorter does not save the DB; it primarily reads. Scraper saves it.
        return processed_character_names, character_to_show_map, character_to_path_map, ai_features_list, color_features_list

    def _calculate_sorted_character_list_sorter(
        self,
        character_names: List[str], # These are names from files, may include '_'
        character_to_show_map: Dict[str, str],
        ai_features: List[Optional[np.ndarray]], # List can contain None
        color_features: List[np.ndarray]
    ) -> List[str]:
        if not character_names: return []
        num_chars = len(character_names)
        
        # --- Determine feature vectors ---
        # Check if any AI features were successfully extracted
        # And if PyTorch is available and model loaded
        valid_ai_features_exist = any(f is not None for f in ai_features)
        use_ai_features = PYTORCH_AVAILABLE and self.main_app.ai_model_sorter is not None and valid_ai_features_exist

        feature_vectors_list = []
        if use_ai_features:
            self.main_app.status_var.set("Sorter: Using AI features for similarity.")
            for i in range(num_chars):
                if ai_features[i] is not None:
                    feature_vectors_list.append(ai_features[i])
                else: # AI feature failed for this image, use its color feature as fallback
                    # logging.warning(f"AI feature missing for {character_names[i]}, using its color feature as fallback in AI mode.")
                    # This is problematic. Better to use zeros for missing AI features if in AI mode.
                    feature_vectors_list.append(np.zeros(self.main_app.ai_feature_size_sorter))
            feature_vectors_np = np.array(feature_vectors_list)
        else: # Fallback to color features
            self.main_app.status_var.set("Sorter: Using color features for similarity (AI not available/failed).")
            if not any(f is not None and f.size > 0 for f in color_features): # Check if any color features are valid
                    messagebox.showerror("Error", "No valid color features could be obtained for sorting.", parent=self.main_app.root)
                    return []
            feature_vectors_np = np.array([cf if cf is not None else np.zeros(3) for cf in color_features])

        # Normalize features (mean centering and scaling)
        if feature_vectors_np.shape[0] > 1:
            mean = np.mean(feature_vectors_np, axis=0)
            std_dev = np.std(feature_vectors_np, axis=0)
            feature_vectors_np = (feature_vectors_np - mean) / (std_dev + 1e-9) # Epsilon for stability
        
        # Calculate pairwise distances
        distances_matrix = np.zeros((num_chars, num_chars))
        if SCIPY_AVAILABLE and feature_vectors_np.shape[0] > 0 :
            try:
                # Use cosine for high-dimensional AI features, euclidean for low-dim color
                metric = 'cosine' if use_ai_features and feature_vectors_np.shape[1] > 3 else 'euclidean'
                distances_matrix = squareform(pdist(feature_vectors_np, metric))
            except ValueError as e: # Can happen if all vectors are identical
                logging.warning(f"Could not calculate feature distances: {e}. Sorting may be affected.")
                # Keep distances_matrix as zeros, greedy sort will pick first available.
        elif not SCIPY_AVAILABLE:
                logging.warning("SciPy not available, distance calculation skipped. Sorting will be less accurate.")


        # --- Group characters by show ---
        show_groups: Dict[str, List[int]] = defaultdict(list) # {show_name: [char_index_in_original_list]}
        for i, char_name_from_file in enumerate(character_names):
            show = character_to_show_map.get(char_name_from_file, UNKNOWN_SHOW_SORTER)
            show_groups[show].append(i)

        # --- Sort show groups ---
        sorted_show_names = list(show_groups.keys())
        
        # Get the first character name (potentially with underscores, as stored from file scan)
        first_char_name_opt_from_file = self.first_character_var_sorter.get() 
        first_char_show: Optional[str] = None
        if first_char_name_opt_from_file and first_char_name_opt_from_file in character_to_show_map:
            first_char_show = character_to_show_map[first_char_name_opt_from_file]
        
        def sort_key_shows(show_name_key: str):
            is_first_show = (show_name_key == first_char_show)
            group_size = len(show_groups[show_name_key])
            return (not is_first_show, -group_size if self.sort_groups_by_size_var_sorter.get() else 0, show_name_key)
        sorted_show_names.sort(key=sort_key_shows)
        
        # --- Process each show group ---
        final_sorted_indices: List[int] = []
        for show_name in sorted_show_names:
            group_indices = show_groups[show_name] # Indices relative to original character_names list
            if not group_indices: continue

            if len(group_indices) == 1 or not self.sort_within_groups_var_sorter.get():
                final_sorted_indices.extend(group_indices) # Add in original relative order
            else: # Sort within this group by visual similarity
                current_group_start_idx: Optional[int] = None
                if first_char_name_opt_from_file and show_name == first_char_show:
                    try:
                        # Find the index of the first character *within the original character_names list*
                        first_char_global_idx = character_names.index(first_char_name_opt_from_file)
                        if first_char_global_idx in group_indices:
                            current_group_start_idx = first_char_global_idx
                    except ValueError: pass # Should not happen if validated
                
                if current_group_start_idx is None: # Default to the first char in this group (original order)
                    current_group_start_idx = group_indices[0]

                # Greedy nearest neighbor sort within the group
                group_sorted_indices_for_show = [current_group_start_idx]
                remaining_in_group = set(group_indices)
                remaining_in_group.remove(current_group_start_idx)
                
                current_global_idx_in_loop = current_group_start_idx
                while remaining_in_group:
                    min_dist_val = float('inf')
                    next_global_idx = -1
                    for r_idx in remaining_in_group:
                        dist_val = distances_matrix[current_global_idx_in_loop, r_idx]
                        if dist_val < min_dist_val:
                            min_dist_val = dist_val
                            next_global_idx = r_idx
                    
                    if next_global_idx != -1:
                        group_sorted_indices_for_show.append(next_global_idx)
                        remaining_in_group.remove(next_global_idx)
                        current_global_idx_in_loop = next_global_idx
                    else: break # No more findable items or error
                final_sorted_indices.extend(group_sorted_indices_for_show)
        
        # Return list of character names (from file, with underscores) in sorted order
        return [character_names[i] for i in final_sorted_indices]

    def _generate_mudae_commands_sorter(self, sorted_character_list: List[str]) -> List[str]:
        """Generates Mudae $sm or $smpos commands. Input names may have underscores."""
        if not sorted_character_list: return []

        commands: List[str] = []
        current_batch_names_for_cmd: List[str] = [] # Names for current command, will be space-separated
        is_first_batch = True 
        last_char_of_previous_batch_for_smpos: Optional[str] = None 

        cmd_prefix_type = self.cmd_type_var_sorter.get() # "sm" or "smpos"

        for char_name_from_file in sorted_character_list:
            # Mudae expects names with spaces, not underscores.
            char_name_for_mudae = char_name_from_file.replace('_', ' ')
            
            # Mudae uses '$' as a separator. If char name has '$', it needs careful handling.
            # For now, assume names are clean or Mudae handles it.
            # If names have spaces, Mudae sometimes needs quotes, but $sm/$smpos usually work with '$' sep.
            
            current_batch_names_for_cmd.append(char_name_for_mudae)
            
            # Tentatively build command to check length
            temp_command_parts = []
            current_cmd_base = "$sm" # Default base
            if cmd_prefix_type == "smpos" and not is_first_batch and last_char_of_previous_batch_for_smpos:
                current_cmd_base = f"$smpos \"{last_char_of_previous_batch_for_smpos}\"" # Quote if it has spaces
            
            temp_command_parts.append(current_cmd_base)
            temp_command_parts.extend([f"\"{name}\"" for name in current_batch_names_for_cmd]) # Quote all names
            tentative_command = " $ ".join(temp_command_parts)

            if len(tentative_command) > DISCORD_COMMAND_CHAR_LIMIT_SORTER:
                char_causing_overflow = current_batch_names_for_cmd.pop() # Remove last added
                
                if current_batch_names_for_cmd: # If batch still has chars
                    final_cmd_parts = []
                    final_cmd_base = "$sm"
                    if cmd_prefix_type == "smpos" and not is_first_batch and last_char_of_previous_batch_for_smpos:
                            final_cmd_base = f"$smpos \"{last_char_of_previous_batch_for_smpos}\""
                    
                    final_cmd_parts.append(final_cmd_base)
                    final_cmd_parts.extend([f"\"{name}\"" for name in current_batch_names_for_cmd])
                    commands.append(" $ ".join(final_cmd_parts))
                    
                    last_char_of_previous_batch_for_smpos = current_batch_names_for_cmd[-1]
                    is_first_batch = False
                    current_batch_names_for_cmd = [char_causing_overflow] # Start new batch
                else: # Single char name itself is too long (with prefix)
                    # This case is tricky. Just form a command with it.
                    final_cmd_parts = []
                    final_cmd_base = "$sm" # Start new chain with $sm
                    if cmd_prefix_type == "smpos" and not is_first_batch and last_char_of_previous_batch_for_smpos:
                        # This implies the $smpos prefix + this one char is too long.
                        # Fallback to $sm for this one problematic char to break the chain.
                        # Or, just let it be too long and user has to manually fix.
                        # For now, let's assume it's a new $sm command.
                        pass # Stick to $sm as base

                    final_cmd_parts.append(final_cmd_base)
                    final_cmd_parts.append(f"\"{char_causing_overflow}\"")
                    commands.append(" $ ".join(final_cmd_parts))

                    last_char_of_previous_batch_for_smpos = char_causing_overflow
                    is_first_batch = False # This was a batch of one
                    current_batch_names_for_cmd = [] # Batch is now empty
        
        # Add any remaining characters in the last batch
        if current_batch_names_for_cmd:
            final_cmd_parts = []
            final_cmd_base = "$sm"
            if cmd_prefix_type == "smpos" and not is_first_batch and last_char_of_previous_batch_for_smpos:
                    final_cmd_base = f"$smpos \"{last_char_of_previous_batch_for_smpos}\""
            
            final_cmd_parts.append(final_cmd_base)
            final_cmd_parts.extend([f"\"{name}\"" for name in current_batch_names_for_cmd])
            commands.append(" $ ".join(final_cmd_parts))
            
        return commands

    def _display_commands_in_ui_sorter(self, commands: List[str]):
        self.result_text_area_sorter.delete(1.0, tk.END)
        if not commands:
            self.result_text_area_sorter.insert(tk.END, "No commands generated. Check input and settings.")
            return

        instructions = (
            "# Mudae Sort Commands (Generated by Mudae Power Tools - Sorter)\n\n"
            "1. Copy each command block below (one at a time).\n"
            "2. Paste it into Discord where the Mudae bot is active.\n"
            "3. Wait for Mudae to confirm completion before pasting the next command.\n\n"
            "----------------------------------------------------------------\n"
        )
        self.result_text_area_sorter.insert(tk.END, instructions)

        for i, command_str in enumerate(commands):
            self.result_text_area_sorter.insert(tk.END, f"## Command {i + 1} of {len(commands)}:\n")
            self.result_text_area_sorter.insert(tk.END, command_str + "\n")
            self.result_text_area_sorter.insert(tk.END, "----------------------------------------------------------------\n\n")
        
        notes = (
            "# Notes:\n"
            f"- Commands are based on the '{self.cmd_type_var_sorter.get()}' type.\n"
            f"- Names are quoted. If issues, try removing quotes or check for special chars in names.\n"
            "- Show grouping and visual similarity sorting are best-effort.\n"
        )
        self.result_text_area_sorter.insert(tk.END, notes)
        self.main_app.status_var.set(f"Sorter: Generated {len(commands)} sort commands.")
        if self.main_app.root.winfo_exists():
            messagebox.showinfo("Success", f"{len(commands)} sort command(s) generated.\nCopy them from the text area.", parent=self.main_app.root)

    def _load_and_analyze_action_sorter(self):
        folder = self.sorter_image_folder_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid image folder first.", parent=self.main_app.root)
            return
        
        # Ensure AI model is loaded or loading initiated if PyTorch is available
        if PYTORCH_AVAILABLE and not self.main_app.ai_model_sorter:
            if not (self.main_app.model_loading_thread_sorter and self.main_app.model_loading_thread_sorter.is_alive()):
                self._setup_ai_model_sorter() # Start loading
                messagebox.showinfo("AI Model Loading", 
                                    "The AI model for visual sorting is now loading in the background. "
                                    "This might take a moment (especially on first run).\n\n"
                                    "Please try analyzing again once it's ready (status bar will update).", 
                                    parent=self.main_app.root)
                return # User needs to wait and click again
            else: # Model is currently loading
                messagebox.showinfo("Busy", "Sorter AI Model is still loading. Please wait and try again.", parent=self.main_app.root)
                return
        
        # Reload global character DB in case Scraper updated it
        self.main_app._load_character_db_global()

        self.character_data_sorter.clear() # Clear previous sorter run data
        
        # Scan images, identify shows, extract features
        names, shows, paths, ai_f, color_f = self._scan_and_extract_features_sorter(folder)
        
        if not names: # Error handled in scan function
            self._update_sorter_ui_elements()
            return

        # Store these for later use by preview and command generation
        self.processed_char_names_sorter_session = names
        self.char_to_show_map_sorter_session = shows
        self.char_to_path_map_sorter_session = paths 
        self.ai_features_list_sorter_session = ai_f
        self.color_features_list_sorter_session = color_f
        # self.character_data_sorter is already populated by _scan_and_extract_features_sorter

        self.main_app.status_var.set(f"Sorter: Analysis complete. {len(self.character_data_sorter)} characters processed.")
        self._update_sorter_ui_elements() # Enable other buttons
        if self.main_app.root.winfo_exists():
                messagebox.showinfo("Analysis Complete", f"{len(self.character_data_sorter)} characters analyzed and features extracted.", parent=self.main_app.root)


    def _generate_commands_action_sorter(self):
        if not self.character_data_sorter:
            messagebox.showerror("Error", "No character data loaded. Please 'Load & Analyze Images' first.", parent=self.main_app.root)
            return
        
        if not hasattr(self, 'processed_char_names_sorter_session') or \
            not self.processed_char_names_sorter_session: # Check if list is empty too
            messagebox.showerror("Error", "Analysis data is incomplete or empty. Please 'Load & Analyze Images' again.", parent=self.main_app.root)
            return

        sorted_chars_from_file = self._calculate_sorted_character_list_sorter(
            self.processed_char_names_sorter_session, 
            self.char_to_show_map_sorter_session, 
            self.ai_features_list_sorter_session, 
            self.color_features_list_sorter_session 
        )
        if not sorted_chars_from_file:
            self.result_text_area_sorter.delete(1.0, tk.END)
            self.result_text_area_sorter.insert(tk.END, "Could not determine sorted order. Check logs or image features.")
            return
            
        commands = self._generate_mudae_commands_sorter(sorted_chars_from_file)
        self._display_commands_in_ui_sorter(commands)

    def _preview_sort_action_sorter(self):
        if not self.character_data_sorter:
            messagebox.showerror("Error", "No character data loaded. Please 'Load & Analyze Images' first.", parent=self.main_app.root)
            return
        
        if not hasattr(self, 'processed_char_names_sorter_session') or \
            not self.processed_char_names_sorter_session:
            messagebox.showerror("Error", "Analysis data is incomplete or empty for preview. Please 'Load & Analyze Images' again.", parent=self.main_app.root)
            return

        sorted_chars_for_preview = self._calculate_sorted_character_list_sorter(
            self.processed_char_names_sorter_session, 
            self.char_to_show_map_sorter_session, 
            self.ai_features_list_sorter_session, 
            self.color_features_list_sorter_session 
        )
        if not sorted_chars_for_preview:
            messagebox.showerror("Preview Error", "Could not determine sorted order for preview.", parent=self.main_app.root)
            return

        self._show_preview_window_sorter(
            sorted_chars_for_preview, 
            self.char_to_show_map_sorter_session, 
            self.char_to_path_map_sorter_session 
        )

    def _show_preview_window_sorter(self, sorted_characters: List[str], 
                                    character_shows: Dict[str, str], 
                                    image_paths: Dict[str, str]):
        """Creates and displays the preview window with thumbnails. Names may have underscores."""
        preview_win = tk.Toplevel(self.main_app.root)
        preview_win.title("Sort Preview")
        preview_win.geometry("800x650")
        preview_win.minsize(600, 400)
        preview_win.grab_set()

        controls_frame = ttk.Frame(preview_win, padding=(10,10,10,0))
        controls_frame.pack(fill=tk.X)

        search_var = tk.StringVar()
        ttk.Label(controls_frame, text="Search:").pack(side=tk.LEFT, padx=(0,5))
        search_entry = ttk.Entry(controls_frame, textvariable=search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(0,10))

        view_var = tk.StringVar(value="grid") # "grid" or "list"
        ttk.Label(controls_frame, text="View:").pack(side=tk.LEFT, padx=(10,5))
        ttk.Radiobutton(controls_frame, text="Grid", variable=view_var, value="grid").pack(side=tk.LEFT)
        ttk.Radiobutton(controls_frame, text="List", variable=view_var, value="list").pack(side=tk.LEFT, padx=5)
        
        canvas_container = ttk.Frame(preview_win, padding=(10,5,10,0))
        canvas_container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(canvas_container, borderwidth=0, background="#ffffff")
        v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        content_frame = ttk.Frame(canvas) 
        canvas.create_window((0, 0), window=content_frame, anchor=tk.NW) 

        def on_frame_configure(event=None): canvas.configure(scrollregion=canvas.bbox("all"))
        content_frame.bind("<Configure>", on_frame_configure)

        preview_win.photo_references = [] 

        def populate_thumbnails(current_view_type: str, search_term: str = ""):
            for widget in content_frame.winfo_children(): widget.destroy()
            preview_win.photo_references.clear()

            thumbnail_size = (100, 100) if current_view_type == "grid" else (60, 60)
            cols = 5 if current_view_type == "grid" else 1
            row_idx, col_idx = 0, 0
            last_show_displayed = None

            for char_name_from_file in sorted_characters: 
                display_char_name = char_name_from_file.replace('_', ' ')
                show_name = character_shows.get(char_name_from_file, UNKNOWN_SHOW_SORTER)
                
                if search_term and not (search_term in display_char_name.lower() or search_term in show_name.lower()):
                    continue

                if show_name != last_show_displayed:
                    if last_show_displayed is not None and current_view_type == "grid" and col_idx != 0:
                        row_idx += 1; col_idx = 0
                    
                    show_header_frame = ttk.Frame(content_frame, padding=(0, (10 if last_show_displayed else 2), 0, 5))
                    show_header_frame.grid(row=row_idx, column=0, columnspan=cols, sticky=tk.EW, pady=(5,0))
                    ttk.Label(show_header_frame, text=f"— {show_name} —", font=("Arial", 11, "bold")).pack(fill=tk.X)
                    row_idx += 1; col_idx = 0
                    last_show_displayed = show_name

                item_frame = ttk.Frame(content_frame, padding=3, relief=tk.RIDGE, borderwidth=1)
                item_frame.grid(row=row_idx, column=col_idx, padx=5, pady=5, sticky=tk.NSEW if current_view_type == "list" else tk.N)

                img_path = image_paths.get(char_name_from_file)
                try:
                    if img_path and os.path.exists(img_path):
                        img = Image.open(img_path)
                        img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        preview_win.photo_references.append(photo)
                        ttk.Label(item_frame, image=photo).pack(pady=(0,3))
                    else: ttk.Label(item_frame, text="[No Img]", foreground="red").pack(pady=(0,3))
                except Exception as e_img:
                    logging.warning(f"Error loading thumbnail for {display_char_name}: {e_img}")
                    ttk.Label(item_frame, text="[Load Err]", foreground="red").pack(pady=(0,3))
                
                name_wraplength = 100 if current_view_type == "grid" else 500
                ttk.Label(item_frame, text=display_char_name, wraplength=name_wraplength, anchor=tk.CENTER).pack()

                if current_view_type == "grid":
                    col_idx += 1
                    if col_idx >= cols: col_idx = 0; row_idx += 1
                else: row_idx += 1
            
            content_frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox("all"))
            canvas.yview_moveto(0); canvas.xview_moveto(0)

        def on_view_change(*args): populate_thumbnails(view_var.get(), search_var.get().lower())
        view_var.trace_add("write", on_view_change)
        def on_search_change(*args): populate_thumbnails(view_var.get(), search_var.get().lower())
        search_var.trace_add("write", on_search_change)
        
        accept_button_frame = ttk.Frame(preview_win, padding=(10,5,10,10))
        accept_button_frame.pack(fill=tk.X)
        
        def accept_and_generate():
            preview_win.destroy()
            commands = self._generate_mudae_commands_sorter(sorted_characters)
            self._display_commands_in_ui_sorter(commands)

        ttk.Button(accept_button_frame, text="Accept This Sort Order & Generate Commands", command=accept_and_generate).pack(pady=5)
        populate_thumbnails(view_var.get()) 
        
        if view_var.get() == "grid":
            for i in range(5): content_frame.columnconfigure(i, weight=1)


    def update_default_image_folder(self, new_folder_path: str):
        current_sorter_folder = self.sorter_image_folder_var.get()
        last_known_default = getattr(self, '_last_known_scraper_output_folder', self.main_app.output_folder)

        if not current_sorter_folder or current_sorter_folder == last_known_default:
            self.sorter_image_folder_var.set(new_folder_path)
        
        setattr(self, '_last_known_scraper_output_folder', new_folder_path)


# --- Main Execution ---
def main():
    # Dependency Checks
    required_packages = {
        "Pillow": "Pillow (PIL)",
        "numpy": "numpy",
        "opencv-python": "opencv-python (cv2)",
        "pyautogui": "pyautogui",
        "pytesseract": "pytesseract",
    }
    missing_packages_critical = []
    for import_name, pkg_name in required_packages.items():
        try:
            if import_name == "Pillow": __import__("PIL")
            elif import_name == "opencv-python": __import__("cv2")
            else: __import__(import_name)
        except ImportError:
            missing_packages_critical.append(pkg_name)

    if missing_packages_critical:
        error_msg = "Critical Python libraries missing:\n" + "\n".join(f" - {lib}" for lib in missing_packages_critical)
        error_msg += "\n\nPlease install them (e.g., using pip)."
        if "pytesseract" in error_msg:
                error_msg += "\nFor Pytesseract, also install Tesseract OCR engine from UB Mannheim and add to PATH or configure in app."
        try: 
            root_err = tk.Tk(); root_err.withdraw()
            messagebox.showerror("Missing Libraries", error_msg)
            root_err.destroy()
        except tk.TclError: print(f"CRITICAL ERROR (Tkinter not available): {error_msg}")
        except Exception as e_gui: print(f"CRITICAL ERROR (GUI error dialog failed: {e_gui}): {error_msg}")
        return
    
    try:
        root = tk.Tk()
        app = MudaePowerToolsApp(root)
        root.mainloop()
    except Exception as e:
        logging.critical(f"Unhandled exception in main application: {e}", exc_info=True)
        try:
            err_root = tk.Tk(); err_root.withdraw()
            messagebox.showerror("Critical Application Error", f"A critical error occurred:\n{e}\n\nPlease check logs or console output.")
            err_root.destroy()
        except Exception as final_e: 
            print(f"CRITICAL APPLICATION ERROR (GUI for error failed): {e}\nSecondary error showing dialog: {final_e}")

if __name__ == "__main__":
    main()
