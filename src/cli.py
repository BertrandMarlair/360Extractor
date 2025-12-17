#!/usr/bin/env python3
"""
360 Extractor CLI - Command Line Interface
Process 360° and flat videos from the terminal without GUI.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.job import Job
from core.cli_processor import CLIProcessor


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='360 Extractor - Process 360° and flat videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process 360 video with defaults
  %(prog)s video360.mp4

  # Process flat video with blur filter
  %(prog)s --mode flat --blur-filter --blur-threshold 150 video.mp4

  # Process multiple videos with AI masking
  %(prog)s --mode flat --ai-mode mask video1.mp4 video2.mp4

  # Custom 360 processing
  %(prog)s --mode 360 --cameras 8 --fov 100 --interval 2 video360.mp4
        '''
    )
    
    # Positional arguments
    parser.add_argument(
        'videos',
        nargs='+',
        help='One or more video files to process'
    )
    
    # Video mode
    parser.add_argument(
        '--mode',
        choices=['360', 'flat'],
        default='360',
        help='Video processing mode (default: 360)'
    )
    
    # Extraction settings
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Extraction interval value (default: 1.0)'
    )
    
    parser.add_argument(
        '--interval-unit',
        choices=['seconds', 'frames'],
        default='seconds',
        help='Interval unit (default: seconds)'
    )
    
    # 360 mode settings
    parser.add_argument(
        '--resolution',
        type=int,
        default=1024,
        help='Output resolution for 360 mode in pixels (default: 1024)'
    )
    
    parser.add_argument(
        '--fov',
        type=int,
        default=90,
        help='Field of view for 360 mode in degrees (default: 90)'
    )
    
    parser.add_argument(
        '--cameras',
        type=int,
        default=6,
        help='Number of cameras for 360 mode (default: 6)'
    )
    
    parser.add_argument(
        '--pitch',
        type=int,
        default=0,
        help='Camera pitch offset in degrees (default: 0)'
    )
    
    # Output settings
    parser.add_argument(
        '--format',
        choices=['jpg', 'png', 'tiff'],
        default='jpg',
        help='Output image format (default: jpg)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='',
        help='Custom output directory (default: same as input video)'
    )
    
    # AI settings
    parser.add_argument(
        '--ai-mode',
        choices=['none', 'skip', 'mask'],
        default='none',
        help='AI operator removal mode (default: none, skip=skip frames with people, mask=generate masks)'
    )
    
    # Blur filter settings
    parser.add_argument(
        '--blur-filter',
        action='store_true',
        help='Enable blur filter to skip blurry frames'
    )
    
    parser.add_argument(
        '--blur-threshold',
        type=float,
        default=100.0,
        help='Blur detection threshold (default: 100.0, higher=stricter)'
    )
    
    parser.add_argument(
        '--smart-blur',
        action='store_true',
        help='Enable adaptive blur filtering (experimental)'
    )
    
    # Sharpening settings
    parser.add_argument(
        '--sharpen',
        action='store_true',
        help='Enable sharpening post-processing'
    )
    
    parser.add_argument(
        '--sharpen-strength',
        type=float,
        default=0.5,
        help='Sharpening strength (default: 0.5)'
    )
    
    return parser.parse_args()


def validate_files(video_paths):
    """Validate that all video files exist and are valid."""
    valid_extensions = ['.mp4', '.mov', '.mkv', '.avi']
    valid_files = []
    
    for path in video_paths:
        if not os.path.isfile(path):
            print(f"Error: File not found: {path}", file=sys.stderr)
            continue
        
        ext = os.path.splitext(path)[1].lower()
        if ext not in valid_extensions:
            print(f"Error: Invalid file type: {path} (expected {', '.join(valid_extensions)})", file=sys.stderr)
            continue
        
        valid_files.append(path)
    
    return valid_files


def create_jobs_from_args(args):
    """Create Job objects from parsed arguments."""
    # Map CLI arguments to settings dictionary
    ai_mode_mapping = {
        'none': 'None',
        'skip': 'Skip Frame',
        'mask': 'Generate Mask'
    }
    
    settings = {
        'video_mode': args.mode.upper() if args.mode == 'flat' else '360',
        'interval_value': args.interval,
        'interval_unit': args.interval_unit.capitalize(),
        'resolution': args.resolution,
        'fov': args.fov,
        'camera_count': args.cameras,
        'pitch_offset': args.pitch,
        'output_format': args.format,
        'custom_output_dir': args.output_dir,
        'ai_mode': ai_mode_mapping[args.ai_mode],
        'blur_filter_enabled': args.blur_filter,
        'blur_threshold': args.blur_threshold,
        'smart_blur_enabled': args.smart_blur,
        'sharpening_enabled': args.sharpen,
        'sharpening_strength': args.sharpen_strength
    }
    
    # Create jobs
    jobs = []
    for video_path in args.videos:
        job = Job(file_path=video_path, settings=settings.copy())
        jobs.append(job)
    
    return jobs


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    # Validate video files
    valid_videos = validate_files(args.videos)
    
    if not valid_videos:
        print("Error: No valid video files to process.", file=sys.stderr)
        return 1
    
    # Update args with only valid videos
    args.videos = valid_videos
    
    # Create jobs
    jobs = create_jobs_from_args(args)
    
    print(f"360 Extractor CLI")
    print(f"================")
    print(f"Mode: {args.mode.upper()}")
    print(f"Processing {len(jobs)} video(s)...")
    print()
    
    # Process jobs
    processor = CLIProcessor(jobs)
    success = processor.process_all()
    
    if success:
        print("\n✓ All videos processed successfully!")
        return 0
    else:
        print("\n✗ Processing completed with errors.", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
