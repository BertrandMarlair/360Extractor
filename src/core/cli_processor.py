"""
CLI Processor - Adapter for processing videos without Qt signals.
Provides synchronous processing with terminal-friendly progress output.
"""

import sys
import cv2
import numpy as np
import os
import time
from collections import deque

from core.geometry import GeometryProcessor
from core.ai_model import AIService
from utils.file_manager import FileManager
from utils.image_utils import ImageUtils


class CLIProcessor:
    """
    Process videos from CLI without Qt dependencies.
    Provides terminal output instead of GUI signals.
    """
    
    def __init__(self, jobs):
        """
        Initialize CLI processor.
        
        Args:
            jobs: List of Job objects to process
        """
        self.jobs = jobs
        self.ai_service = None
        
        # Initialize AI Service if needed
        needs_ai = any(job.settings.get('ai_mode', 'None') != 'None' for job in self.jobs)
        
        if needs_ai:
            print("Loading AI model (YOLOv8)...")
            self.ai_service = AIService('yolov8n-seg.pt')
            print("AI model loaded.\n")
    
    def process_all(self):
        """
        Process all jobs sequentially.
        
        Returns:
            bool: True if all jobs succeeded, False otherwise
        """
        total_jobs = len(self.jobs)
        success_count = 0
        
        for i, job in enumerate(self.jobs):
            print(f"\n[{i+1}/{total_jobs}] Processing: {job.filename}")
            print("-" * 60)
            
            try:
                self.process_video(job, i, total_jobs)
                job.status = "Done"
                success_count += 1
                print(f"✓ Completed: {job.filename}")
            except Exception as e:
                job.status = "Error"
                print(f"✗ Error processing {job.filename}: {str(e)}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        
        return success_count == total_jobs
    
    def process_video(self, job, job_index, total_jobs):
        """
        Process a single video job.
        This is adapted from ProcessingWorker.process_video() but without Qt signals.
        """
        file_path = job.file_path
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]

        # Determine Output Directory
        custom_dir = job.output_dir
        if custom_dir:
            # Custom output directory specified
            # Convert to absolute path if relative
            if not os.path.isabs(custom_dir):
                custom_dir = os.path.abspath(custom_dir)
            base_output_dir = custom_dir
        else:
            # Default: use video's directory
            base_output_dir = os.path.dirname(file_path)

        # Create a specific subfolder for this video
        output_dir = os.path.join(base_output_dir, f"{name_no_ext}_processed")
        
        try:
            FileManager.ensure_directory(output_dir)
            print(f"Output directory: {output_dir}")
        except OSError as e:
            raise IOError(f"Cannot create output directory {output_dir}: {e}")

        # Determine Output Format & Params
        fmt = job.output_format.lower()
        if fmt not in ['jpg', 'png', 'tiff']:
            fmt = 'jpg'
        
        ext = f".{fmt}"
        if fmt == 'tiff':
            ext = '.tif'
        
        save_params = []
        if fmt == 'jpg':
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif fmt == 'png':
            save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        elif fmt == 'tiff':
            save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {file_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_video <= 0: 
            total_frames_video = 1
        
        print(f"Video info: {int(fps)} fps, {total_frames_video} frames")
        
        # Calculate extraction interval
        interval_value = float(job.settings.get('interval_value', 1.0))
        interval_unit = job.settings.get('interval_unit', 'Seconds')
        
        if interval_unit == 'Frames':
            interval = int(max(1, interval_value))
        else:
            interval = int(max(1, fps * interval_value))
        
        print(f"Extracting every {interval_value} {interval_unit.lower()} (frame interval: {interval})")
        
        # Geometry Settings
        out_res = job.settings.get('resolution', 1024)
        fov = job.settings.get('fov', 90)
        camera_count = job.settings.get('camera_count', 6)
        pitch_offset = job.settings.get('pitch_offset', 0)
        
        # AI Mode
        ai_mode_ui = job.settings.get('ai_mode', 'None')
        ai_mode_internal = 'none'
        if ai_mode_ui == 'Skip Frame':
            ai_mode_internal = 'skip_frame'
        elif ai_mode_ui == 'Generate Mask':
            ai_mode_internal = 'generate_mask'

        # Blur Filter Settings
        blur_enabled = job.settings.get('blur_filter_enabled', False)
        smart_blur_enabled = job.settings.get('smart_blur_enabled', False)
        blur_threshold = job.settings.get('blur_threshold', 100.0)
        skipped_blur_count = 0
        
        # Adaptive Blur State
        blur_history = deque(maxlen=10)
        consecutive_blur_skips = 0

        # Sharpening Settings
        sharpen_enabled = job.settings.get('sharpening_enabled', False)
        sharpen_strength = job.settings.get('sharpening_strength', 0.5)
        
        # Video Mode
        video_mode = job.settings.get('video_mode', '360')
        
        # Get video dimensions
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Generate views and maps for 360 mode only
        views = []
        maps = {}
        
        if video_mode == '360':
            print(f"360° mode: Generating {camera_count} camera views (FOV: {fov}°, pitch: {pitch_offset}°)")
            views = GeometryProcessor.generate_views(camera_count, pitch_offset=pitch_offset)
            
            for name, y, p, r in views:
                maps[name] = GeometryProcessor.create_rectilinear_map(
                    src_h, src_w, out_res, out_res, fov, y, p, r
                )
            print(f"Reprojection maps generated.")
        else:
            print(f"FLAT mode: Extracting full frames ({src_w}x{src_h})")

        frame_idx = 0
        saved_frame_count = 0
        job_start_time = time.time()
        last_print_time = time.time()
        
        print("\nProcessing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % interval == 0:
                # Progress output (print every 2 seconds to avoid spam)
                current_time = time.time()
                if current_time - last_print_time >= 2.0 or frame_idx == 0:
                    progress = int((frame_idx / total_frames_video) * 100)
                    elapsed = current_time - job_start_time
                    
                    if frame_idx > 0 and elapsed > 0:
                        rate = frame_idx / elapsed
                        remaining_frames = total_frames_video - frame_idx
                        eta_seconds = remaining_frames / rate
                        eta_min = int(eta_seconds // 60)
                        eta_sec = int(eta_seconds % 60)
                        print(f"  Frame {frame_idx}/{total_frames_video} ({progress}%) - ETA: {eta_min}m {eta_sec}s", flush=True)
                    else:
                        print(f"  Frame {frame_idx}/{total_frames_video} ({progress}%)", flush=True)
                    
                    last_print_time = current_time

                if video_mode == '360':
                    # 360° MODE: Process each reprojected view
                    for name, _, _, _ in views:
                        map_x, map_y = maps[name]
                        rect_img = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                        
                        # Blur Detection
                        if blur_enabled:
                            score = ImageUtils.calculate_blur_score(rect_img)
                            is_blurry = self._check_blur(score, blur_threshold, smart_blur_enabled, 
                                                        blur_history, consecutive_blur_skips)
                            
                            if is_blurry:
                                skipped_blur_count += 1
                                continue

                        # Sharpening
                        if sharpen_enabled:
                            gaussian = cv2.GaussianBlur(rect_img, (0, 0), 2.0)
                            rect_img = cv2.addWeighted(rect_img, 1.0 + sharpen_strength, gaussian, -sharpen_strength, 0)

                        # AI Processing
                        final_img = rect_img
                        mask_or_skip = None
                        
                        if self.ai_service and ai_mode_internal != 'none':
                            final_img, result_extra = self.ai_service.process_image(rect_img, mode=ai_mode_internal)
                            
                            if ai_mode_internal == 'skip_frame' and result_extra is True:
                                continue
                            elif ai_mode_internal == 'generate_mask':
                                mask_or_skip = result_extra
                        
                        # Save
                        if final_img is not None:
                            save_name = f"{name_no_ext}_frame{frame_idx:06d}_{name}{ext}"
                            FileManager.save_image(os.path.join(output_dir, save_name), final_img, save_params)
                            saved_frame_count += 1
                            
                            if mask_or_skip is not None and isinstance(mask_or_skip, np.ndarray):
                                # Create mask subfolder
                                mask_dir = os.path.join(output_dir, "mask")
                                FileManager.ensure_directory(mask_dir)
                                # COLMAP convention: image.jpg.png (keep original extension + .png)
                                mask_name = f"{save_name}.png"
                                FileManager.save_mask(os.path.join(mask_dir, mask_name), mask_or_skip)
                
                else:
                    # FLAT MODE: Process the full frame
                    final_img = frame.copy()
                    
                    # Blur Detection
                    if blur_enabled:
                        score = ImageUtils.calculate_blur_score(final_img)
                        is_blurry = self._check_blur(score, blur_threshold, smart_blur_enabled, 
                                                    blur_history, consecutive_blur_skips)
                        
                        if is_blurry:
                            skipped_blur_count += 1
                            frame_idx += 1
                            continue
                    
                    # Sharpening
                    if sharpen_enabled:
                        gaussian = cv2.GaussianBlur(final_img, (0, 0), 2.0)
                        final_img = cv2.addWeighted(final_img, 1.0 + sharpen_strength, gaussian, -sharpen_strength, 0)
                    
                    # AI Processing
                    mask_or_skip = None
                    
                    if self.ai_service and ai_mode_internal != 'none':
                        final_img, result_extra = self.ai_service.process_image(final_img, mode=ai_mode_internal)
                        
                        if ai_mode_internal == 'skip_frame' and result_extra is True:
                            frame_idx += 1
                            continue
                        elif ai_mode_internal == 'generate_mask':
                            mask_or_skip = result_extra
                    
                    # Save
                    if final_img is not None:
                        save_name = f"{name_no_ext}_frame{frame_idx:06d}{ext}"
                        FileManager.save_image(os.path.join(output_dir, save_name), final_img, save_params)
                        saved_frame_count += 1
                        
                        if mask_or_skip is not None and isinstance(mask_or_skip, np.ndarray):
                            # Create mask subfolder
                            mask_dir = os.path.join(output_dir, "mask")
                            FileManager.ensure_directory(mask_dir)
                            # COLMAP convention: image.jpg.png (keep original extension + .png)
                            mask_name = f"{save_name}.png"
                            FileManager.save_mask(os.path.join(mask_dir, mask_name), mask_or_skip)
            
            frame_idx += 1
            
        cap.release()
        
        print(f"\nSaved {saved_frame_count} images")
        if skipped_blur_count > 0:
            print(f"Skipped {skipped_blur_count} blurry frames")
    
    def _check_blur(self, score, threshold, smart_mode, history, consecutive_skips):
        """Helper method to check if a frame is blurry."""
        is_blurry = False
        
        if smart_mode:
            # Minimum floor check
            if score < threshold:
                is_blurry = True
            # Adaptive check
            elif len(history) > 0:
                avg_score = sum(history) / len(history)
                if score < avg_score * 0.6:
                    is_blurry = True
            
            # Safety override
            if is_blurry:
                consecutive_skips += 1
                if consecutive_skips > 5:
                    is_blurry = False
                    consecutive_skips = 0
            
            # Update history
            if not is_blurry:
                consecutive_skips = 0
                history.append(score)
        else:
            # Standard mode
            if score < threshold:
                is_blurry = True
        
        return is_blurry
