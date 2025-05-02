# src/models/summarization/video_summarizer.py
import numpy as np
import cv2
import torch
from ..fusion.multimodal_transformer import MultimodalTransformer


class VideoSummarizer:
    def __init__(self, frame_features_dim=2048, audio_features_dim=128):
        # Load fusion model for importance scoring
        self.fusion_model = MultimodalTransformer(
            image_dim=frame_features_dim,
            audio_dim=audio_features_dim,
            embed_dim=512
        )

        # Importance scoring layer
        self.importance_scorer = torch.nn.Linear(512, 1)

    def compute_frame_importance(self, frame_features, audio_features=None):
        """Compute importance score for each frame"""
        # Convert to tensors
        frame_features_tensor = torch.tensor(frame_features, dtype=torch.float32)

        if audio_features is not None:
            audio_features_tensor = torch.tensor(audio_features, dtype=torch.float32)
        else:
            audio_features_tensor = None

        # Get multimodal embeddings
        with torch.no_grad():
            embeddings = self.fusion_model(
                image_features=frame_features_tensor,
                audio_features=audio_features_tensor
            )

            # Score importance
            scores = self.importance_scorer(embeddings)
            scores = torch.sigmoid(scores).squeeze().numpy()

        return scores

    def generate_keyframes(self, frame_features, frames, audio_features=None, num_keyframes=5):
        """Generate key frames from video"""
        # Compute importance scores
        scores = self.compute_frame_importance(frame_features, audio_features)

        # Get indices of top scoring frames
        top_indices = np.argsort(scores)[-num_keyframes:]

        # Return keyframes
        keyframes = [frames[i] for i in sorted(top_indices)]

        return keyframes, top_indices

    def create_video_summary(self, video_path, keyframe_indices, output_path,
                             segment_duration=3):
        """Create a video summary from keyframes"""
        # Open input video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Calculate segment frame count
        segment_frames = int(segment_duration * fps)

        # Extract segments around keyframes
        for idx in keyframe_indices:
            # Set position to start of segment
            segment_start = max(0, idx - segment_frames // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start)

            # Write segment frames
            for _ in range(segment_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)

        cap.release()
        out.release()

        return output_path

    def generate_highlights(self, video_path, frame_features, frames,
                            audio_features=None, num_segments=3,
                            output_path="highlights.mp4"):
        """Generate video highlights"""
        # Get keyframes
        _, keyframe_indices = self.generate_keyframes(
            frame_features, frames, audio_features, num_segments
        )

        # Create highlight video
        summary_path = self.create_video_summary(
            video_path, keyframe_indices, output_path
        )

        return summary_path