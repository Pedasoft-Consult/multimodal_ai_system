# src/preprocessing/video/preprocessor.py
import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
from ..image.preprocessor import ImagePreprocessor
from ..audio.preprocessor import AudioPreprocessor


class VideoPreprocessor:
    def __init__(self, frame_rate=1, image_size=(224, 224)):
        self.frame_rate = frame_rate  # Extract n frames per second
        self.image_size = image_size
        self.image_processor = ImagePreprocessor(image_size=image_size)
        self.audio_processor = AudioPreprocessor()

    def extract_frames(self, video_path, output_dir=None, max_frames=100):
        """Extract frames from video at specified frame rate"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame interval based on desired frame rate
        frame_interval = int(fps / self.frame_rate)

        frames = []
        frame_paths = []

        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Extract frames
        count = 0
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            if i % frame_interval == 0:
                # Process frame
                frame = cv2.resize(frame, self.image_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame)

                # Save frame if output_dir is provided
                if output_dir:
                    frame_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frame_paths.append(frame_path)

                count += 1
                if count >= max_frames:
                    break

        cap.release()
        return frames, frame_paths

    def extract_optical_flow(self, video_path):
        """Extract optical flow from video"""
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Parameters for optical flow
        flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flows = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, **flow_params
            )

            flows.append(flow)
            prev_gray = gray

        cap.release()
        return flows

    def extract_audio_from_video(self, video_path, output_path=None):
        """Extract audio from video file"""
        # If output path is not provided, create a temporary one
        if output_path is None:
            output_path = video_path.replace(".mp4", ".wav")

        # Use ffmpeg to extract audio
        os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {output_path} -y")

        # Process the audio
        audio_features = self.audio_processor.process_audio(output_path)

        return audio_features

    def process_video(self, video_path, temp_dir=None):
        """Process video: extract frames, optical flow, and audio features"""
        # Create temporary directory for frames if needed
        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(video_path), "temp_frames")

        # Extract frames
        frames, frame_paths = self.extract_frames(video_path, temp_dir)

        # Process frames with image processor
        frame_features = []
        for path in frame_paths:
            features = self.image_processor.extract_features(path)
            frame_features.append(features)

        # Extract optical flow
        flows = self.extract_optical_flow(video_path)

        # Extract and process audio
        audio_path = os.path.join(temp_dir, "audio.wav")
        audio_features = self.extract_audio_from_video(video_path, audio_path)

        return {
            'frames': frames,
            'frame_features': np.array(frame_features),
            'optical_flow': flows,
            'audio_features': audio_features
        }