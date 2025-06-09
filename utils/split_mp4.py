import cv2
import os
import argparse

def extract_frames_from_video(video_path, output_folder, prefix=""):
    cap = cv2.VideoCapture(video_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        
        if success:
            frame_filename = os.path.join(output_folder, f"{prefix}_frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

    cap.release()
    
    return frame_count

def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video file.')
    
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('-o', '--output', default='output_frames',
                        help='Output folder for extracted frames (default: output_frames)')
    parser.add_argument('-p', '--prefix', default='',
                        help='Prefix for output frame filenames (default: no prefix)')
    
    args = parser.parse_args()
    
    total_frames = extract_frames_from_video(args.video_path, args.output, args.prefix)
    
    print(f"Successfully extracted {total_frames} frames to {args.output}")

if __name__ == "__main__":
    main()
