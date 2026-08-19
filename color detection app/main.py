"""
ColorPulse Vision: A Real-Time Spatial Color Intelligence Platform - Main Application Entrypoint.
Provides unified access to GUI, CLI interactive computer vision modes,
K-Means dominant color extraction, real-time color masking studio, and latency benchmarking.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Configure console encoding for Windows compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config.settings import get_sample_image_path, ML_CONFIG, VISION_CONFIG
from src.core.color_classifier import ColorClassifier
from src.core.color_extractor import DominantColorExtractor
from src.core.object_tracker import ColorObjectTracker, TrackingMode
from src.core.color_masker import ColorMaskEngine, MaskFilterType
from src.core.video_stream import VideoStream
from src.utils.logger import app_logger


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ColorPulse Vision: A Real-Time Spatial Color Intelligence Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Launch the modern CustomTkinter graphical dashboard (default if no args provided)"
    )
    parser.add_argument(
        "--cli", action="store_true", default=False,
        help="Run in standalone high-performance OpenCV native window mode"
    )
    parser.add_argument(
        "-i", "--image", type=str, default=None,
        help="Path to an image for static color inspection or dominant palette extraction"
    )
    parser.add_argument(
        "-k", "--kmeans", type=int, default=None,
        help="Extract top K dominant colors using unsupervised K-Means clustering (e.g. -k 5)"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="all",
        choices=["all", "red", "green", "blue", "yellow", "orange", "purple", "multi", "grid", "mask", "off"],
        help="Processing mode: 'all' (tracking), 'grid' (2x2 RGB mask split), 'mask' (color isolation), or color names"
    )
    parser.add_argument(
        "--benchmark", action="store_true", default=False,
        help="Run high-throughput CIELAB KD-Tree benchmark against standard RGB distance"
    )
    return parser.parse_args()


def run_image_inspection(image_path: Path, k_clusters: int | None = None) -> None:
    """Performs static image color picking or K-Means clustering in CLI mode."""
    if not image_path.exists():
        app_logger.error(f"Image not found at {image_path}")
        return

    cv_img = cv2.imread(str(image_path))
    if cv_img is None:
        app_logger.error(f"Failed to load image at {image_path}")
        return

    classifier = ColorClassifier()

    if k_clusters is not None:
        app_logger.info(f"Running Unsupervised K-Means Dominant Color Clustering (K={k_clusters})...")
        extractor = DominantColorExtractor(classifier)
        result = extractor.extract_palette(cv_img, k=k_clusters)
        
        print("\n" + "=" * 60)
        print(f" DOMINANT COLOR PALETTE REPORT ({image_path.name})")
        print("=" * 60)
        print(f"{'Rank':<6}{'Color Name':<28}{'Hex Code':<12}{'RGB':<18}{'Percentage':<10}")
        print("-" * 60)
        for idx, color in enumerate(result.dominant_colors, 1):
            rgb_str = f"({color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]})"
            print(f"{idx:<6}{color.color_name:<28}{color.hex_code:<12}{rgb_str:<18}{color.percentage}%")
        print("=" * 60)
        return

    app_logger.info("Opening interactive image inspector. Click anywhere on the image to detect and pin color, press ESC to exit.")
    
    # Display image at original resolution
    base_img = cv_img.copy()
    h, w = base_img.shape[:2]
    
    # State tracking
    state = {
        "x": w // 2,
        "y": h // 2,
        "pinned_text": None,
        "pinned_bgr": None,
        "clicked": False
    }

    def render_view():
        frame = base_img.copy()
        cur_x = max(0, min(w - 1, state["x"]))
        cur_y = max(0, min(h - 1, state["y"]))
        
        # 1. Render persistent pinned top banner on click
        if state["pinned_text"] and state["pinned_bgr"]:
            b, g, r = state["pinned_bgr"]
            cv2.rectangle(frame, (20, 20), (min(w - 20, 750), 65), (b, g, r), -1)
            cv2.rectangle(frame, (20, 20), (min(w - 20, 750), 65), (255, 255, 255), 1)
            text_color = (0, 0, 0) if (b * 0.114 + g * 0.587 + r * 0.299) > 140 else (255, 255, 255)
            cv2.putText(frame, state["pinned_text"], (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2, cv2.LINE_AA)

        # 2. Draw crosshair pointer cursor at cur_x, cur_y
        reticle_color = (0, 255, 255) if state["clicked"] else (255, 255, 255)
        
        # Outer shadow
        cv2.line(frame, (max(0, cur_x - 16), cur_y), (min(w - 1, cur_x + 16), cur_y), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(frame, (cur_x, max(0, cur_y - 16)), (cur_x, min(h - 1, cur_y + 18)), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(frame, (cur_x, cur_y), 9, (0, 0, 0), 3, cv2.LINE_AA)

        # Inner crosshairs
        cv2.line(frame, (max(0, cur_x - 16), cur_y), (min(w - 1, cur_x + 16), cur_y), reticle_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cur_x, max(0, cur_y - 16)), (cur_x, min(h - 1, cur_y + 18)), reticle_color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cur_x, cur_y), 9, reticle_color, 1, cv2.LINE_AA)
        
        if state["clicked"] and state["pinned_bgr"]:
            cv2.circle(frame, (cur_x, cur_y), 4, state["pinned_bgr"], -1, cv2.LINE_AA)

        cv2.imshow("Interactive Color Inspector", frame)

    def on_mouse_event(event, x, y, flags, param):
        clamped_x = max(0, min(w - 1, x))
        clamped_y = max(0, min(h - 1, y))
        state["x"] = clamped_x
        state["y"] = clamped_y

        # Detect and pin color strictly on click
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK]:
            b, g, r = [int(v) for v in base_img[clamped_y, clamped_x]]
            res = classifier.classify_rgb(r, g, b)
            state["pinned_text"] = f"{res.name} | RGB=({r},{g},{b}) | Hex={res.hex_code} | dE={res.delta_e:.1f}"
            state["pinned_bgr"] = (b, g, r)
            state["clicked"] = True

        render_view()

    cv2.namedWindow("Interactive Color Inspector", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Interactive Color Inspector", on_mouse_event)
    render_view()
    
    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break
            
    cv2.destroyAllWindows()


def run_cli_color_masker(filter_str: str = "grid") -> None:
    """Executes high-throughput OpenCV native color masking and filter engine."""
    mask_engine = ColorMaskEngine()
    
    filter_map = {
        "grid": MaskFilterType.MULTI_GRID,
        "mask": MaskFilterType.MULTI_GRID,
        "blue": MaskFilterType.BLUE,
        "red": MaskFilterType.RED,
        "green": MaskFilterType.GREEN,
        "yellow": MaskFilterType.YELLOW,
        "orange": MaskFilterType.ORANGE,
        "purple": MaskFilterType.PURPLE
    }
    selected_filter = filter_map.get(filter_str, MaskFilterType.MULTI_GRID)

    app_logger.info(f"Launching Color Masking Studio in mode: {selected_filter.value}")
    app_logger.info("Press ESC in window to exit.")

    stream = VideoStream().start()
    if stream.stopped:
        app_logger.error("Could not access camera.")
        return

    try:
        while True:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            if selected_filter == MaskFilterType.MULTI_GRID:
                rendered = mask_engine.generate_multi_grid_view(frame)
            else:
                res = mask_engine.apply_filter(frame, filter_type=selected_filter)
                rendered = res.isolated_frame
                cv2.putText(
                    rendered, f"Coverage: {res.coverage_percentage}%", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                )

            fps = stream.get_fps()
            cv2.putText(
                rendered, f"FPS: {fps}", (rendered.shape[1] - 140, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )

            cv2.imshow("Color Masking & HSV Studio", rendered)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def run_cli_video_tracker(mode_str: str) -> None:
    """Executes high-throughput OpenCV native video tracker."""
    if mode_str in ["grid", "mask"]:
        run_cli_color_masker(mode_str)
        return

    mode_dict = {
        "all": TrackingMode.ALL_COLORS,
        "red": TrackingMode.RED,
        "green": TrackingMode.GREEN,
        "blue": TrackingMode.BLUE,
        "yellow": TrackingMode.YELLOW,
        "orange": TrackingMode.ORANGE,
        "purple": TrackingMode.PURPLE,
        "multi": TrackingMode.MULTI_SIMULTANEOUS,
        "off": TrackingMode.OFF
    }
    mode = mode_dict.get(mode_str, TrackingMode.ALL_COLORS)
    
    classifier = ColorClassifier()
    tracker = ColorObjectTracker(classifier)
    
    app_logger.info(f"Launching high-performance video tracker in mode: {mode.value}")
    app_logger.info("Press ESC in window to exit.")

    stream = VideoStream().start()
    if stream.stopped:
        app_logger.error("Could not access camera.")
        return

    try:
        while True:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            annotated_frame, tracked_objs, _ = tracker.process_frame(frame, mode=mode, draw_annotations=True)

            fps = stream.get_fps()
            cv2.putText(
                annotated_frame, f"FPS: {fps}", (frame.shape[1] - 140, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )

            cv2.imshow("AI Real-Time Color Tracker", annotated_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def run_benchmark() -> None:
    """Runs latency and accuracy benchmark of CIELAB KDTree vs naive Euclidean matching."""
    print("\n" + "=" * 65)
    print(" BENCHMARK: CIELAB KD-TREE vs NAIVE LINEAR SEARCH")
    print("=" * 65)

    classifier_lab = ColorClassifier(use_lab=True)
    
    n_queries = 10000
    np.random.seed(42)
    random_rgb = np.random.randint(0, 256, size=(n_queries, 3), dtype=np.uint8)

    print(f"Running {n_queries:,} nearest-color queries over {len(classifier_lab.df)} reference colors...")

    t0 = time.perf_counter()
    lab_results = classifier_lab.batch_classify_rgb(random_rgb)
    t_lab = time.perf_counter() - t0
    avg_lab_us = (t_lab / n_queries) * 1_000_000

    print(f"CIELAB KD-Tree: Total = {t_lab:.4f}s | Avg Latency = {avg_lab_us:.2f} us/query ({int(n_queries/t_lab):,} queries/sec)")

    t0 = time.perf_counter()
    df = classifier_lab.df
    for i in range(min(500, n_queries)):
        r, g, b = random_rgb[i]
        d = np.abs(df['R'] - r) + np.abs(df['G'] - g) + np.abs(df['B'] - b)
        _ = df.loc[d.idxmin(), 'color_name']
    t_naive_partial = time.perf_counter() - t0
    avg_naive_us = (t_naive_partial / min(500, n_queries)) * 1_000_000

    print(f"Naive Linear Search: Avg Latency = {avg_naive_us:.2f} us/query")
    speedup = avg_naive_us / avg_lab_us
    print(f"KD-Tree Acceleration: {speedup:.1f}x Faster with CIELAB Perceptual Accuracy!")
    print("=" * 65 + "\n")


def main():
    args = parse_arguments()

    if args.benchmark:
        run_benchmark()
        return

    if args.kmeans is not None or (args.image is not None and not args.gui):
        target_path = Path(args.image) if args.image else get_sample_image_path()
        run_image_inspection(target_path, k_clusters=args.kmeans)
        return

    if args.cli:
        if args.mode in ["grid", "mask"]:
            run_cli_color_masker(args.mode)
        else:
            run_cli_video_tracker(args.mode)
        return

    from src.gui.app import run_gui
    run_gui()


if __name__ == "__main__":
    main()
