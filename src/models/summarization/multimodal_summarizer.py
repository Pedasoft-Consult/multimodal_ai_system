# src/models/summarization/multimodal_summarizer.py
from .text_summarizer import TextSummarizer
from .video_summarizer import VideoSummarizer
import os


class MultimodalSummarizer:
    def __init__(self):
        self.text_summarizer = TextSummarizer()
        self.video_summarizer = VideoSummarizer()

    def generate_multimodal_summary(self, text=None, video_path=None,
                                    frame_features=None, frames=None,
                                    audio_features=None, output_dir="summaries"):
        """Generate a comprehensive multimodal summary"""
        summary = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate text summary if available
        if text:
            abstractive_summary = self.text_summarizer.abstractive_summarize(text)
            extractive_summary = self.text_summarizer.extractive_summarize(text)

            summary["text"] = {
                "abstractive": abstractive_summary,
                "extractive": extractive_summary
            }

            # Save text summaries
            with open(os.path.join(output_dir, "abstractive_summary.txt"), "w") as f:
                f.write(abstractive_summary)

            with open(os.path.join(output_dir, "extractive_summary.txt"), "w") as f:
                f.write(extractive_summary)

        # Generate video summary if available
        if video_path and frame_features is not None and frames is not None:
            # Generate keyframes
            keyframes, keyframe_indices = self.video_summarizer.generate_keyframes(
                frame_features, frames, audio_features
            )

            # Create highlight video
            highlight_path = os.path.join(output_dir, "highlights.mp4")
            self.video_summarizer.create_video_summary(
                video_path, keyframe_indices, highlight_path
            )

            summary["video"] = {
                "keyframe_indices": keyframe_indices.tolist(),
                "highlight_video": highlight_path
            }

            # Save keyframes as images
            for i, frame in enumerate(keyframes):
                cv2.imwrite(os.path.join(output_dir, f"keyframe_{i}.jpg"), frame)

        return summary