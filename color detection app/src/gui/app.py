"""
ColorPulse Vision: A Real-Time Spatial Color Intelligence Platform.
Features interactive static image inspection, K-Means palette clustering,
integrated real-time video tracking canvas, real-time color masking & HSV filter studio,
and dataset analytics.
"""

import csv
import json
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

from config.settings import GUI_CONFIG, get_colors_csv_path, get_sample_image_path
from src.core.color_classifier import ColorClassifier, ColorMatchResult
from src.core.color_extractor import DominantColorExtractor, PaletteExtractionResult
from src.core.object_tracker import ColorObjectTracker, TrackingMode
from src.core.color_masker import ColorMaskEngine, MaskFilterType, MaskResult
from src.core.video_stream import VideoStream
from src.utils.image_processing import resize_with_aspect_ratio, rgb_to_hex
from src.utils.logger import app_logger


class ColorVisionApp(ctk.CTk):
    """
    Main Application Window integrating Computer Vision, Color Masking, and Machine Learning services.
    """

    def __init__(self):
        super().__init__()

        # Appearance setup
        ctk.set_appearance_mode(GUI_CONFIG.THEME_APPEARANCE)
        ctk.set_default_color_theme(GUI_CONFIG.COLOR_THEME)

        self.title(GUI_CONFIG.APP_TITLE)
        self.geometry(GUI_CONFIG.WINDOW_SIZE)
        self.minsize(1100, 750)
        
        # Maximize window on launch for full-screen immersive view
        try:
            self.state("zoomed")
        except Exception:
            pass

        # Core Engines
        self.classifier = ColorClassifier()
        self.extractor = DominantColorExtractor(self.classifier)
        self.tracker = ColorObjectTracker(self.classifier)
        self.mask_engine = ColorMaskEngine()
        self.video_stream: Optional[VideoStream] = None

        # State Variables
        self.current_image_path: Path = get_sample_image_path()
        self.current_cv_image: Optional[np.ndarray] = None
        self.displayed_cv_image: Optional[np.ndarray] = None
        self.image_scale_factor: float = 1.0
        
        # Camera states
        self.is_camera_running: bool = False
        self.tracking_mode = TrackingMode.ALL_COLORS
        self.mask_filter_type = MaskFilterType.MULTI_GRID
        self.mask_view_mode: str = "isolated"  # "isolated", "binary", or "grid"
        self.detected_logs: List[dict] = []

        # Build UI layout
        self._create_header()
        self._create_main_tabs()

        # Load default image on startup
        self.after(100, self._load_initial_image)

        # Protocol for graceful window exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_header(self) -> None:
        """Builds modern application header bar."""
        header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray85", "gray14"))
        header_frame.pack(fill="x", padx=0, pady=0)

        title_label = ctk.CTkLabel(
            header_frame,
            text="ColorPulse Vision",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=12)

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="A Real-Time Spatial Color Intelligence Platform",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle_label.pack(side="left", padx=10, pady=12)

        # Dataset status badge
        dataset_count = len(self.classifier.df)
        badge = ctk.CTkLabel(
            header_frame,
            text=f"Indexed Colors: {dataset_count}",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1f538d",
            corner_radius=6,
            padx=10,
            pady=4
        )
        badge.pack(side="right", padx=20, pady=10)

    def _create_main_tabs(self) -> None:
        """Creates main tabbed interface."""
        self.tabview = ctk.CTkTabview(self, corner_radius=8)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=(10, 15))

        self.tab_image = self.tabview.add("Image Inspector & K-Means Palette")
        self.tab_tracker = self.tabview.add("Live Object Tracker")
        self.tab_masker = self.tabview.add("Color Mask & HSV Filter Studio")
        self.tab_dataset = self.tabview.add("Dataset & Telemetry Analytics")

        self._build_image_inspector_tab()
        self._build_video_tracker_tab()
        self._build_color_masker_tab()
        self._build_dataset_tab()

    # =========================================================================
    # TAB 1: STATIC IMAGE INSPECTOR & K-MEANS PALETTE
    # =========================================================================
    def _build_image_inspector_tab(self) -> None:
        """Constructs static image analysis, click-to-identify, and K-Means clustering view."""
        toolbar = ctk.CTkFrame(self.tab_image, fg_color="transparent")
        toolbar.pack(fill="x", padx=10, pady=5)

        btn_load = ctk.CTkButton(
            toolbar, text="Load Custom Image", command=self._on_select_image, width=160
        )
        btn_load.pack(side="left", padx=5)

        btn_sample = ctk.CTkButton(
            toolbar, text="Reset Sample Image", command=self._on_reset_sample_image, width=160
        )
        btn_sample.pack(side="left", padx=5)

        self.lbl_image_info = ctk.CTkLabel(
            toolbar, text="Click anywhere on the image to identify color.", text_color="gray"
        )
        self.lbl_image_info.pack(side="left", padx=15)

        content_frame = ctk.CTkFrame(self.tab_image, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas_frame = ctk.CTkFrame(content_frame)
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Pixel-perfect canvas for zero-jitter, 1-to-1 mouse coordinate calibration
        import tkinter as tk
        self.image_canvas = tk.Canvas(
            self.canvas_frame,
            bg="#181818",
            highlightthickness=0,
            cursor="crosshair"
        )
        self.image_canvas.pack(anchor="center", expand=True)
        self.image_canvas.bind("<Button-1>", self._on_image_clicked)
        self.image_canvas.bind("<Motion>", self._on_image_hover)
        self.image_canvas.bind("<Leave>", self._on_image_leave)
        self.canvas_frame.bind("<Configure>", self._on_canvas_frame_resize)
        self.tk_image_ref = None
        self._last_rendered_size = (0, 0)

        right_pane = ctk.CTkFrame(content_frame, width=380)
        right_pane.pack(side="right", fill="both", padx=(0, 0))

        card_inspect = ctk.CTkFrame(right_pane)
        card_inspect.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            card_inspect, text="Selected Pixel Color", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self.swatch_box = ctk.CTkFrame(card_inspect, height=45, corner_radius=6, fg_color="#333333")
        self.swatch_box.pack(fill="x", padx=10, pady=5)

        self.lbl_color_name = ctk.CTkLabel(
            card_inspect, text="Color Name: None", font=ctk.CTkFont(size=14, weight="bold")
        )
        self.lbl_color_name.pack(anchor="w", padx=10, pady=2)

        self.lbl_color_values = ctk.CTkLabel(
            card_inspect, text="RGB: - | Hex: - | dE: -", text_color="gray"
        )
        self.lbl_color_values.pack(anchor="w", padx=10, pady=(2, 10))

        card_kmeans = ctk.CTkFrame(right_pane)
        card_kmeans.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        ctk.CTkLabel(
            card_kmeans, text="K-Means Dominant Color Clustering", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        slider_frame = ctk.CTkFrame(card_kmeans, fg_color="transparent")
        slider_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_k_val = ctk.CTkLabel(slider_frame, text="Clusters (K): 5")
        self.lbl_k_val.pack(side="left", padx=5)

        self.slider_k = ctk.CTkSlider(
            slider_frame, from_=2, to=8, number_of_steps=6, command=self._on_k_slider_change
        )
        self.slider_k.set(5)
        self.slider_k.pack(side="right", fill="x", expand=True, padx=5)

        btn_extract = ctk.CTkButton(
            card_kmeans, text="Extract Dominant Palette", command=self._on_extract_palette
        )
        btn_extract.pack(fill="x", padx=10, pady=10)

        self.lbl_swatch_strip = ctk.CTkLabel(card_kmeans, text="", height=40)
        self.lbl_swatch_strip.pack(fill="x", padx=10, pady=5)

        self.palette_scroll = ctk.CTkScrollableFrame(card_kmeans, height=180)
        self.palette_scroll.pack(fill="both", expand=True, padx=10, pady=(5, 10))

    def _load_initial_image(self) -> None:
        sample_path = get_sample_image_path()
        if sample_path.exists():
            self._load_and_display_image(sample_path)
            self._on_extract_palette()

    def _on_canvas_frame_resize(self, event) -> None:
        """Dynamically rescales image to fill the available canvas space on window resize."""
        if event.widget != self.canvas_frame or self.current_cv_image is None:
            return
            
        cur_w, cur_h = event.width, event.height
        if cur_w < 100 or cur_h < 100:
            return

        # Avoid unnecessary re-renders if size change is minimal
        last_w, last_h = self._last_rendered_size
        if abs(cur_w - last_w) > 20 or abs(cur_h - last_h) > 20:
            self._render_current_image()

    def _load_and_display_image(self, path: Path) -> None:
        cv_img = cv2.imread(str(path))
        if cv_img is None:
            app_logger.error(f"Unable to read image at {path}")
            return

        self.current_image_path = path
        self.current_cv_image = cv_img
        self.pinned_coord = None
        self._render_current_image()
        
        self.lbl_image_info.configure(
            text=f"Loaded: {path.name} ({cv_img.shape[1]}x{cv_img.shape[0]}px) - Click anywhere on image to detect color"
        )

    def _render_current_image(self) -> None:
        """Renders the loaded image scaled to fully fill the available canvas container."""
        if self.current_cv_image is None:
            return

        try:
            self.update_idletasks()
        except Exception:
            pass

        # Get actual realized geometry of the container
        frame_w = self.canvas_frame.winfo_width()
        frame_h = self.canvas_frame.winfo_height()

        # If window hasn't drawn yet or frame is small, use screen size
        if frame_w < 300 or frame_h < 300:
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            frame_w = max(900, screen_w - 440)
            frame_h = max(650, screen_h - 220)

        target_max_w = max(frame_w - 20, 900)
        target_max_h = max(frame_h - 20, 650)

        resized_cv, scale = resize_with_aspect_ratio(
            self.current_cv_image,
            max_width=target_max_w,
            max_height=target_max_h,
            allow_upscale=True
        )
        self.displayed_cv_image = resized_cv
        self.image_scale_factor = scale
        self._last_rendered_size = (self.canvas_frame.winfo_width(), self.canvas_frame.winfo_height())

        rgb_img = cv2.cvtColor(resized_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Configure canvas to exact image dimensions for 1:1 hardware coordinate mapping
        self.image_canvas.config(width=pil_img.width, height=pil_img.height)
        self.tk_image_ref = ImageTk.PhotoImage(pil_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, image=self.tk_image_ref, anchor="nw")

    def _on_select_image(self) -> None:
        from customtkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self._load_and_display_image(Path(file_path))
            self._on_extract_palette()

    def _on_reset_sample_image(self) -> None:
        self._load_and_display_image(get_sample_image_path())
        self._on_extract_palette()

    def _on_k_slider_change(self, value: float) -> None:
        self.lbl_k_val.configure(text=f"Clusters (K): {int(value)}")

    def _on_image_hover(self, event) -> None:
        """Draws pointer cursor reticle on hover without changing color detection card."""
        if self.displayed_cv_image is None:
            return

        h, w = self.displayed_cv_image.shape[:2]
        x, y = event.x, event.y

        if 0 <= x < w and 0 <= y < h:
            self._render_pointer_overlay(x, y, (255, 255, 255), is_hover=True)

    def _on_image_clicked(self, event) -> None:
        """Detects and displays color strictly on click."""
        if self.displayed_cv_image is None:
            return

        h, w = self.displayed_cv_image.shape[:2]
        x, y = event.x, event.y

        if 0 <= x < w and 0 <= y < h:
            # Query source or displayed pixel
            if self.current_cv_image is not None and self.image_scale_factor > 0:
                orig_x = min(self.current_cv_image.shape[1] - 1, max(0, int(x / self.image_scale_factor)))
                orig_y = min(self.current_cv_image.shape[0] - 1, max(0, int(y / self.image_scale_factor)))
                b, g, r = self.current_cv_image[orig_y, orig_x]
            else:
                b, g, r = self.displayed_cv_image[y, x]

            result = self.classifier.classify_rgb(int(r), int(g), int(b))
            
            # Show detected color details only on click
            self.swatch_box.configure(fg_color=result.hex_code)
            self.lbl_color_name.configure(text=f"Color: {result.name}")
            self.lbl_color_values.configure(
                text=f"RGB: ({result.rgb[0]}, {result.rgb[1]}, {result.rgb[2]})\n"
                     f"Hex: {result.hex_code} | dE: {result.delta_e:.2f} (Conf: {result.confidence_score*100:.1f}%)"
            )
            self.pinned_coord = (x, y, (int(b), int(g), int(r)))
            self._render_pointer_overlay(x, y, (int(b), int(g), int(r)), is_hover=False)

    def _on_image_leave(self, event) -> None:
        """Restores pinned click reticle or clean image when mouse leaves canvas."""
        if hasattr(self, "pinned_coord") and self.pinned_coord:
            px, py, pbgr = self.pinned_coord
            self._render_pointer_overlay(px, py, pbgr, is_hover=False)
        elif self.displayed_cv_image is not None:
            rgb_img = cv2.cvtColor(self.displayed_cv_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            self.tk_image_ref = ImageTk.PhotoImage(pil_img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, image=self.tk_image_ref, anchor="nw")

    def _render_pointer_overlay(self, x: int, y: int, bgr_color: tuple, is_hover: bool = True) -> None:
        """Draws custom precision crosshairs and reticle pointer onto display image."""
        if self.displayed_cv_image is None:
            return

        canvas_copy = self.displayed_cv_image.copy()
        h, w = canvas_copy.shape[:2]

        # Draw crosshair axes exactly centered at (x, y)
        line_color = (255, 255, 255) if is_hover else (0, 255, 255)
        cv2.line(canvas_copy, (max(0, x - 16), y), (min(w - 1, x + 16), y), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(canvas_copy, (x, max(0, y - 16)), (x, min(h - 1, y + 16)), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(canvas_copy, (max(0, x - 16), y), (min(w - 1, x + 16), y), line_color, 1, cv2.LINE_AA)
        cv2.line(canvas_copy, (x, max(0, y - 16)), (x, min(h - 1, y + 16)), line_color, 1, cv2.LINE_AA)

        # Concentric precision rings
        cv2.circle(canvas_copy, (x, y), 8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(canvas_copy, (x, y), 8, line_color, 1, cv2.LINE_AA)
        if not is_hover:
            cv2.circle(canvas_copy, (x, y), 4, bgr_color, -1, cv2.LINE_AA)

        rgb_img = cv2.cvtColor(canvas_copy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.tk_image_ref = ImageTk.PhotoImage(pil_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, image=self.tk_image_ref, anchor="nw")

    def _on_extract_palette(self) -> None:
        if self.current_cv_image is None:
            return

        k = int(self.slider_k.get())
        result = self.extractor.extract_palette(self.current_cv_image, k=k)

        swatch_rgb = cv2.cvtColor(result.palette_image, cv2.COLOR_BGR2RGB)
        swatch_pil = Image.fromarray(swatch_rgb)
        ctk_swatch = ctk.CTkImage(light_image=swatch_pil, dark_image=swatch_pil, size=(320, 35))
        self.lbl_swatch_strip.configure(image=ctk_swatch)
        self.lbl_swatch_strip.image = ctk_swatch

        for widget in self.palette_scroll.winfo_children():
            widget.destroy()

        for color in result.dominant_colors:
            item_row = ctk.CTkFrame(self.palette_scroll, fg_color=("gray90", "gray18"), height=35)
            item_row.pack(fill="x", padx=2, pady=3)

            swatch_mini = ctk.CTkFrame(item_row, width=24, height=24, fg_color=color.hex_code, corner_radius=4)
            swatch_mini.pack(side="left", padx=8, pady=5)

            lbl_info = ctk.CTkLabel(
                item_row,
                text=f"{color.color_name} ({color.hex_code})",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            lbl_info.pack(side="left", padx=5)

            lbl_percent = ctk.CTkLabel(
                item_row,
                text=f"{color.percentage}%",
                font=ctk.CTkFont(size=12),
                text_color="#3B8ED0"
            )
            lbl_percent.pack(side="right", padx=10)

    # =========================================================================
    # TAB 2: REAL-TIME VIDEO TRACKER
    # =========================================================================
    def _build_video_tracker_tab(self) -> None:
        top_bar = ctk.CTkFrame(self.tab_tracker, fg_color="transparent")
        top_bar.pack(fill="x", padx=10, pady=5)

        self.btn_camera_tracker = ctk.CTkButton(
            top_bar, text="Start Webcam", command=self._toggle_camera, width=140, fg_color="#2FA572", hover_color="#20724F"
        )
        self.btn_camera_tracker.pack(side="left", padx=5)

        ctk.CTkLabel(top_bar, text="Tracking Mode:").pack(side="left", padx=(15, 5))
        self.combo_tracker_mode = ctk.CTkComboBox(
            top_bar,
            values=[
                "All Colors (CIELAB KD-Tree)",
                "Multi-Simultaneous (RGB)",
                "Red",
                "Green",
                "Blue",
                "Yellow",
                "Orange",
                "Purple",
                "Off (Raw Feed)"
            ],
            command=self._on_tracker_mode_selected,
            width=220
        )
        self.combo_tracker_mode.set("All Colors (CIELAB KD-Tree)")
        self.combo_tracker_mode.pack(side="left", padx=5)

        btn_snap = ctk.CTkButton(
            top_bar, text="Snapshot", command=self._take_snapshot, width=110
        )
        btn_snap.pack(side="left", padx=10)

        video_content = ctk.CTkFrame(self.tab_tracker, fg_color="transparent")
        video_content.pack(fill="both", expand=True, padx=10, pady=5)

        self.video_canvas_frame = ctk.CTkFrame(video_content)
        self.video_canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.lbl_video_display = ctk.CTkLabel(
            self.video_canvas_frame,
            text="Camera stream is inactive.\nClick 'Start Webcam' to begin live tracking.",
            font=ctk.CTkFont(size=14)
        )
        self.lbl_video_display.pack(fill="both", expand=True, padx=10, pady=10)

        stats_pane = ctk.CTkFrame(video_content, width=320)
        stats_pane.pack(side="right", fill="both")

        ctk.CTkLabel(
            stats_pane, text="Live Telemetry & Metrics", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=10, pady=10)

        metric_frame = ctk.CTkFrame(stats_pane)
        metric_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_fps = ctk.CTkLabel(
            metric_frame, text="FPS: --", font=ctk.CTkFont(size=14, weight="bold"), text_color="#2FA572"
        )
        self.lbl_fps.pack(anchor="w", padx=10, pady=5)

        self.lbl_obj_count = ctk.CTkLabel(
            metric_frame, text="Objects Tracked: 0", font=ctk.CTkFont(size=13)
        )
        self.lbl_obj_count.pack(anchor="w", padx=10, pady=5)

        ctk.CTkLabel(
            stats_pane, text="Detected Targets:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=10, pady=(15, 5))

        self.tracker_scroll = ctk.CTkScrollableFrame(stats_pane, height=260)
        self.tracker_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _on_tracker_mode_selected(self, choice: str) -> None:
        mode_map = {
            "All Colors (CIELAB KD-Tree)": TrackingMode.ALL_COLORS,
            "Multi-Simultaneous (RGB)": TrackingMode.MULTI_SIMULTANEOUS,
            "Red": TrackingMode.RED,
            "Green": TrackingMode.GREEN,
            "Blue": TrackingMode.BLUE,
            "Yellow": TrackingMode.YELLOW,
            "Orange": TrackingMode.ORANGE,
            "Purple": TrackingMode.PURPLE,
            "Off (Raw Feed)": TrackingMode.OFF
        }
        self.tracking_mode = mode_map.get(choice, TrackingMode.ALL_COLORS)

    # =========================================================================
    # TAB 3: COLOR MASK & HSV FILTER STUDIO
    # =========================================================================
    def _build_color_masker_tab(self) -> None:
        """Constructs dedicated color isolation, bitwise AND filtering, and HSV tuning studio."""
        top_bar = ctk.CTkFrame(self.tab_masker, fg_color="transparent")
        top_bar.pack(fill="x", padx=10, pady=5)

        self.btn_camera_masker = ctk.CTkButton(
            top_bar, text="Start Webcam", command=self._toggle_camera, width=140, fg_color="#2FA572", hover_color="#20724F"
        )
        self.btn_camera_masker.pack(side="left", padx=5)

        ctk.CTkLabel(top_bar, text="Mask Filter:").pack(side="left", padx=(15, 5))
        self.combo_mask_filter = ctk.CTkComboBox(
            top_bar,
            values=[
                "Multi-Channel Grid (RGB)",
                "Blue",
                "Red",
                "Green",
                "Yellow",
                "Orange",
                "Purple",
                "Custom HSV Calibration"
            ],
            command=self._on_mask_filter_selected,
            width=220
        )
        self.combo_mask_filter.set("Multi-Channel Grid (RGB)")
        self.combo_mask_filter.pack(side="left", padx=5)

        ctk.CTkLabel(top_bar, text="View Output:").pack(side="left", padx=(15, 5))
        self.combo_mask_view = ctk.CTkComboBox(
            top_bar,
            values=["Isolated Color Feed", "Binary Black/White Mask"],
            command=self._on_mask_view_selected,
            width=200
        )
        self.combo_mask_view.set("Isolated Color Feed")
        self.combo_mask_view.pack(side="left", padx=5)

        content = ctk.CTkFrame(self.tab_masker, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=10, pady=5)

        # Video Frame
        self.mask_canvas_frame = ctk.CTkFrame(content)
        self.mask_canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.lbl_mask_display = ctk.CTkLabel(
            self.mask_canvas_frame,
            text="Camera stream is inactive.\nClick 'Start Webcam' to begin live color masking.",
            font=ctk.CTkFont(size=14)
        )
        self.lbl_mask_display.pack(fill="both", expand=True, padx=10, pady=10)

        # HSV Calibration & Telemetry Sidebar
        hsv_pane = ctk.CTkScrollableFrame(content, width=340)
        hsv_pane.pack(side="right", fill="both")

        ctk.CTkLabel(
            hsv_pane, text="HSV Tuning & Calibration", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        # Sliders for Custom HSV
        # Lower HSV
        ctk.CTkLabel(hsv_pane, text="Lower HSV Limits:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=10, pady=(8, 2))
        
        self.lbl_lh = ctk.CTkLabel(hsv_pane, text="Lower Hue (H): 0")
        self.lbl_lh.pack(anchor="w", padx=10)
        self.slider_lh = ctk.CTkSlider(hsv_pane, from_=0, to=180, number_of_steps=180, command=lambda v: self.lbl_lh.configure(text=f"Lower Hue (H): {int(v)}"))
        self.slider_lh.set(0)
        self.slider_lh.pack(fill="x", padx=10, pady=2)

        self.lbl_ls = ctk.CTkLabel(hsv_pane, text="Lower Saturation (S): 50")
        self.lbl_ls.pack(anchor="w", padx=10)
        self.slider_ls = ctk.CTkSlider(hsv_pane, from_=0, to=255, number_of_steps=255, command=lambda v: self.lbl_ls.configure(text=f"Lower Saturation (S): {int(v)}"))
        self.slider_ls.set(50)
        self.slider_ls.pack(fill="x", padx=10, pady=2)

        self.lbl_lv = ctk.CTkLabel(hsv_pane, text="Lower Value (V): 50")
        self.lbl_lv.pack(anchor="w", padx=10)
        self.slider_lv = ctk.CTkSlider(hsv_pane, from_=0, to=255, number_of_steps=255, command=lambda v: self.lbl_lv.configure(text=f"Lower Value (V): {int(v)}"))
        self.slider_lv.set(50)
        self.slider_lv.pack(fill="x", padx=10, pady=2)

        # Upper HSV
        ctk.CTkLabel(hsv_pane, text="Upper HSV Limits:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=10, pady=(15, 2))
        
        self.lbl_uh = ctk.CTkLabel(hsv_pane, text="Upper Hue (H): 180")
        self.lbl_uh.pack(anchor="w", padx=10)
        self.slider_uh = ctk.CTkSlider(hsv_pane, from_=0, to=180, number_of_steps=180, command=lambda v: self.lbl_uh.configure(text=f"Upper Hue (H): {int(v)}"))
        self.slider_uh.set(180)
        self.slider_uh.pack(fill="x", padx=10, pady=2)

        self.lbl_us = ctk.CTkLabel(hsv_pane, text="Upper Saturation (S): 255")
        self.lbl_us.pack(anchor="w", padx=10)
        self.slider_us = ctk.CTkSlider(hsv_pane, from_=0, to=255, number_of_steps=255, command=lambda v: self.lbl_us.configure(text=f"Upper Saturation (S): {int(v)}"))
        self.slider_us.set(255)
        self.slider_us.pack(fill="x", padx=10, pady=2)

        self.lbl_uv = ctk.CTkLabel(hsv_pane, text="Upper Value (V): 255")
        self.lbl_uv.pack(anchor="w", padx=10)
        self.slider_uv = ctk.CTkSlider(hsv_pane, from_=0, to=255, number_of_steps=255, command=lambda v: self.lbl_uv.configure(text=f"Upper Value (V): {int(v)}"))
        self.slider_uv.set(255)
        self.slider_uv.pack(fill="x", padx=10, pady=2)

        # Mask Telemetry Box
        telemetry_box = ctk.CTkFrame(hsv_pane)
        telemetry_box.pack(fill="x", padx=10, pady=15)

        self.lbl_mask_coverage = ctk.CTkLabel(
            telemetry_box, text="Mask Coverage: 0.0%", font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_mask_coverage.pack(anchor="w", padx=10, pady=5)

        self.lbl_mask_pixels = ctk.CTkLabel(
            telemetry_box, text="Matched Pixels: 0", font=ctk.CTkFont(size=12)
        )
        self.lbl_mask_pixels.pack(anchor="w", padx=10, pady=5)

    def _on_mask_filter_selected(self, choice: str) -> None:
        filter_map = {
            "Multi-Channel Grid (RGB)": MaskFilterType.MULTI_GRID,
            "Blue": MaskFilterType.BLUE,
            "Red": MaskFilterType.RED,
            "Green": MaskFilterType.GREEN,
            "Yellow": MaskFilterType.YELLOW,
            "Orange": MaskFilterType.ORANGE,
            "Purple": MaskFilterType.PURPLE,
            "Custom HSV Calibration": MaskFilterType.CUSTOM_HSV
        }
        self.mask_filter_type = filter_map.get(choice, MaskFilterType.MULTI_GRID)

    def _on_mask_view_selected(self, choice: str) -> None:
        self.mask_view_mode = "binary" if "Binary" in choice else "isolated"

    # =========================================================================
    # CAMERA CONTROLS & RENDER LOOP
    # =========================================================================
    def _toggle_camera(self) -> None:
        """Starts or stops live video processing thread."""
        if not self.is_camera_running:
            self.video_stream = VideoStream().start()
            if self.video_stream.stopped:
                self.lbl_video_display.configure(text="Error: Could not connect to camera.")
                self.lbl_mask_display.configure(text="Error: Could not connect to camera.")
                return

            self.is_camera_running = True
            btn_text = "Stop Webcam"
            self.btn_camera_tracker.configure(text=btn_text, fg_color="#C0392B", hover_color="#962D22")
            self.btn_camera_masker.configure(text=btn_text, fg_color="#C0392B", hover_color="#962D22")
            self._update_video_loop()
        else:
            self.is_camera_running = False
            if self.video_stream:
                self.video_stream.stop()
                self.video_stream = None
            btn_text = "Start Webcam"
            self.btn_camera_tracker.configure(text=btn_text, fg_color="#2FA572", hover_color="#20724F")
            self.btn_camera_masker.configure(text=btn_text, fg_color="#2FA572", hover_color="#20724F")
            self.lbl_video_display.configure(image=None, text="Camera stream stopped.")
            self.lbl_mask_display.configure(image=None, text="Camera stream stopped.")
            self.lbl_fps.configure(text="FPS: --")

    def _update_video_loop(self) -> None:
        """Unified rendering loop servicing both Object Tracker and Color Masking Studio."""
        if not self.is_camera_running or not self.video_stream:
            return

        grabbed, frame = self.video_stream.read()
        if grabbed and frame is not None:
            frame = cv2.flip(frame, 1)
            active_tab = self.tabview.get()

            # Process according to which video tab is currently active
            if "Tracker" in active_tab:
                # TAB 2: Object Tracker
                annotated_frame, tracked_objects, _ = self.tracker.process_frame(
                    frame, mode=self.tracking_mode, draw_annotations=True
                )
                fps = self.video_stream.get_fps()
                self.lbl_fps.configure(text=f"FPS: {fps}")
                self.lbl_obj_count.configure(text=f"Objects Tracked: {len(tracked_objects)}")
                self._update_telemetry_list(tracked_objects)

                resized_frame, _ = resize_with_aspect_ratio(annotated_frame, max_width=640, max_height=480)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(pil_img.width, pil_img.height))

                self.lbl_video_display.configure(image=ctk_img, text="")
                self.lbl_video_display.image = ctk_img

            elif "Mask" in active_tab:
                # TAB 3: Color Mask Studio
                if self.mask_filter_type == MaskFilterType.MULTI_GRID:
                    rendered_frame = self.mask_engine.generate_multi_grid_view(frame)
                    self.lbl_mask_coverage.configure(text="Mask Coverage: Multi-Grid")
                    self.lbl_mask_pixels.configure(text="View: Split RGB Channels")
                else:
                    custom_lower = np.array([int(self.slider_lh.get()), int(self.slider_ls.get()), int(self.slider_lv.get())], dtype=np.uint8)
                    custom_upper = np.array([int(self.slider_uh.get()), int(self.slider_us.get()), int(self.slider_uv.get())], dtype=np.uint8)
                    
                    mask_res = self.mask_engine.apply_filter(
                        frame,
                        filter_type=self.mask_filter_type,
                        custom_lower=custom_lower,
                        custom_upper=custom_upper
                    )
                    
                    if self.mask_view_mode == "binary":
                        rendered_frame = cv2.cvtColor(mask_res.binary_mask, cv2.COLOR_GRAY2BGR)
                    else:
                        rendered_frame = mask_res.isolated_frame

                    self.lbl_mask_coverage.configure(text=f"Mask Coverage: {mask_res.coverage_percentage}%")
                    self.lbl_mask_pixels.configure(text=f"Matched Pixels: {mask_res.pixel_count:,}")

                resized_mask, _ = resize_with_aspect_ratio(rendered_frame, max_width=640, max_height=480)
                rgb_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2RGB)
                pil_mask = Image.fromarray(rgb_mask)
                ctk_mask_img = ctk.CTkImage(light_image=pil_mask, dark_image=pil_mask, size=(pil_mask.width, pil_mask.height))

                self.lbl_mask_display.configure(image=ctk_mask_img, text="")
                self.lbl_mask_display.image = ctk_mask_img

        if self.is_camera_running:
            self.after(15, self._update_video_loop)

    def _update_telemetry_list(self, tracked_objects: List) -> None:
        for widget in self.tracker_scroll.winfo_children():
            widget.destroy()

        if not tracked_objects:
            lbl = ctk.CTkLabel(self.tracker_scroll, text="No targets in view", text_color="gray")
            lbl.pack(pady=10)
            return

        for obj in tracked_objects[:6]:
            hex_color = rgb_to_hex(obj.rgb_color[0], obj.rgb_color[1], obj.rgb_color[2])
            item = ctk.CTkFrame(self.tracker_scroll, fg_color=("gray90", "gray18"))
            item.pack(fill="x", padx=2, pady=2)

            swatch = ctk.CTkFrame(item, width=16, height=16, fg_color=hex_color, corner_radius=3)
            swatch.pack(side="left", padx=6, pady=4)

            lbl_name = ctk.CTkLabel(item, text=f"{obj.label}", font=ctk.CTkFont(size=12, weight="bold"))
            lbl_name.pack(side="left", padx=4)

            lbl_pos = ctk.CTkLabel(item, text=f"({obj.centroid[0]}, {obj.centroid[1]})", text_color="gray", font=ctk.CTkFont(size=10))
            lbl_pos.pack(side="right", padx=6)

            self.detected_logs.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "label": obj.label,
                "rgb": obj.rgb_color,
                "hex": hex_color,
                "centroid": obj.centroid,
                "area": obj.area
            })

    def _take_snapshot(self) -> None:
        if not self.video_stream:
            return
        grabbed, frame = self.video_stream.read()
        if grabbed and frame is not None:
            save_dir = Path("data/snapshots")
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), frame)
            app_logger.info(f"Snapshot saved to {filename}")

    # =========================================================================
    # TAB 4: DATASET & ANALYTICS
    # =========================================================================
    def _build_dataset_tab(self) -> None:
        bar = ctk.CTkFrame(self.tab_dataset, fg_color="transparent")
        bar.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(bar, text="Search Color:").pack(side="left", padx=5)
        self.entry_search = ctk.CTkEntry(bar, placeholder_text="Type color name or hex...", width=200)
        self.entry_search.pack(side="left", padx=5)
        self.entry_search.bind("<KeyRelease>", self._on_dataset_search)

        btn_export_csv = ctk.CTkButton(
            bar, text="Export Logs (CSV)", command=self._export_logs_csv, width=140
        )
        btn_export_csv.pack(side="right", padx=5)

        btn_export_json = ctk.CTkButton(
            bar, text="Export Logs (JSON)", command=self._export_logs_json, width=140
        )
        btn_export_json.pack(side="right", padx=5)

        self.dataset_scroll = ctk.CTkScrollableFrame(self.tab_dataset)
        self.dataset_scroll.pack(fill="both", expand=True, padx=10, pady=5)

        self._populate_dataset_view()

    def _populate_dataset_view(self, query: str = "") -> None:
        for widget in self.dataset_scroll.winfo_children():
            widget.destroy()

        df = self.classifier.df
        if query:
            df = df[df["color_name"].str.contains(query, case=False) | df["hex"].str.contains(query, case=False)]

        for _, row in df.head(50).iterrows():
            hex_str = str(row["hex"])
            if not hex_str.startswith("#"):
                hex_str = f"#{hex_str}"

            row_frame = ctk.CTkFrame(self.dataset_scroll, fg_color=("gray90", "gray18"), height=35)
            row_frame.pack(fill="x", padx=5, pady=2)

            swatch = ctk.CTkFrame(row_frame, width=25, height=25, fg_color=hex_str, corner_radius=4)
            swatch.pack(side="left", padx=10, pady=5)

            name_lbl = ctk.CTkLabel(row_frame, text=f"{row['color_name']}", font=ctk.CTkFont(size=12, weight="bold"))
            name_lbl.pack(side="left", padx=10)

            hex_lbl = ctk.CTkLabel(row_frame, text=f"{hex_str}", text_color="gray")
            hex_lbl.pack(side="left", padx=20)

            rgb_lbl = ctk.CTkLabel(row_frame, text=f"RGB: ({row['R']}, {row['G']}, {row['B']})", text_color="#3B8ED0")
            rgb_lbl.pack(side="right", padx=15)

    def _on_dataset_search(self, event) -> None:
        query = self.entry_search.get().strip()
        self._populate_dataset_view(query)

    def _export_logs_csv(self) -> None:
        if not self.detected_logs:
            return
        out_path = Path("data/detection_logs.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "rgb", "hex", "centroid", "area"])
            writer.writeheader()
            writer.writerows(self.detected_logs)
        app_logger.info(f"Exported {len(self.detected_logs)} logs to {out_path}")

    def _export_logs_json(self) -> None:
        if not self.detected_logs:
            return
        out_path = Path("data/detection_logs.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.detected_logs, f, indent=2)
        app_logger.info(f"Exported {len(self.detected_logs)} logs to {out_path}")

    def on_closing(self) -> None:
        app_logger.info("Shutting down ColorVisionApp...")
        self.is_camera_running = False
        if self.video_stream:
            self.video_stream.stop()
        self.destroy()


def run_gui() -> None:
    app = ColorVisionApp()
    app.mainloop()
