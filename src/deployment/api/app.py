# src/deployment/api/app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
import shutil
import uuid
from typing import Optional, List

# Import processing modules
from src.preprocessing.text.preprocessor import TextPreprocessor
from src.preprocessing.image.preprocessor import ImagePreprocessor
from src.preprocessing.audio.preprocessor import AudioPreprocessor
from src.preprocessing.video.preprocessor import VideoPreprocessor
from src.models.classification.multimodal_classifier import MultimodalClassifier
from src.knowledge_graph.entity_extraction.text_entity_extractor import TextEntityExtractor
from src.knowledge_graph.entity_extraction.image_entity_extractor import ImageEntityExtractor
from src.knowledge_graph.neo4j_interface import Neo4jInterface
from src.models.summarization.multimodal_summarizer import MultimodalSummarizer

app = FastAPI(title="Multimodal AI System API")

# Initialize processors and models
text_processor = TextPreprocessor()
image_processor = ImagePreprocessor()
audio_processor = AudioPreprocessor()
video_processor = VideoPreprocessor()
classifier = MultimodalClassifier(num_classes=20)  # Replace with actual number of classes
text_entity_extractor = TextEntityExtractor()
image_entity_extractor = ImageEntityExtractor()
neo4j_interface = Neo4jInterface(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password"
)
summarizer = MultimodalSummarizer()


# Define request models
class TextRequest(BaseModel):
    text: str


class ClassificationResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class SummaryResult(BaseModel):
    text_summary: Optional[str] = None
    video_highlights: Optional[str] = None
    keyframes: Optional[List[str]] = None


# Define endpoints
@app.post("/process/text", response_model=dict)
async def process_text(request: TextRequest):
    """Process text input"""
    # Preprocess text
    text_features = text_processor.process_document(request.text)

    # Classify text
    class_id = classifier.predict(text_features=text_features["embeddings"])[0].item()

    # Extract entities and relations
    extraction_result = text_entity_extractor.process_document(request.text)

    # Generate summary
    summary = summarizer.text_summarizer.abstractive_summarize(request.text)

    return {
        "classification": {
            "class_id": class_id,
            "class_name": f"Class {class_id}",  # Replace with actual class names
            "confidence": 0.85  # Placeholder
        },
        "entities": extraction_result["entities"],
        "relations": extraction_result["relations"],
        "summary": summary
    }


@app.post("/process/image", response_model=dict)
async def process_image(file: UploadFile = File(...)):
    """Process image input"""
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process image
        image_features = image_processor.extract_features(temp_path)

        # Classify image
        class_id = classifier.predict(image_features=image_features)[0].item()

        # Extract image entities
        extraction_result = image_entity_extractor.process_image(temp_path)

        return {
            "classification": {
                "class_id": class_id,
                "class_name": f"Class {class_id}",  # Replace with actual class names
                "confidence": 0.85  # Placeholder
            },
            "objects": extraction_result["objects"],
            "knowledge_objects": extraction_result["knowledge_objects"]
        }

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


@app.post("/process/video", response_model=dict)
async def process_video(file: UploadFile = File(...)):
    """Process video input"""
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process video
        video_result = video_processor.process_video(temp_path, temp_dir)

        # Generate highlights
        highlights_path = os.path.join(temp_dir, "highlights.mp4")
        summarizer.video_summarizer.generate_highlights(
            temp_path,
            video_result["frame_features"],
            video_result["frames"],
            video_result["audio_features"]["embeddings"],
            output_path=highlights_path
        )

        # Copy highlights to a persistent location
        output_dir = "static/highlights"
        os.makedirs(output_dir, exist_ok=True)

        unique_id = str(uuid.uuid4())
        output_path = os.path.join(output_dir, f"highlight_{unique_id}.mp4")
        shutil.copy(highlights_path, output_path)

        return {
            "summary": {
                "video_highlights": f"/static/highlights/highlight_{unique_id}.mp4"
            }
        }

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


@app.post("/process/multimodal", response_model=dict)
async def process_multimodal(
        text: Optional[str] = Form(None),
        image: Optional[UploadFile] = File(None),
        video: Optional[UploadFile] = File(None),
        audio: Optional[UploadFile] = File(None)
):
    """Process multimodal inputs"""
    temp_dir = tempfile.mkdtemp()

    try:
        results = {}
        features = {}

        # Process text if provided
        if text:
            text_result = text_processor.process_document(text)
            features["text_features"] = text_result["embeddings"]
            results["text"] = {
                "entities": text_entity_extractor.process_document(text)["entities"]
            }

        # Process image if provided
        if image:
            image_path = os.path.join(temp_dir, image.filename)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            image_features = image_processor.extract_features(image_path)
            features["image_features"] = image_features
            results["image"] = {
                "objects": image_entity_extractor.process_image(image_path)["objects"]
            }

        # Process audio if provided
        if audio:
            audio_path = os.path.join(temp_dir, audio.filename)
            with open(audio_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)

            audio_result = audio_processor.process_audio(audio_path)
            features["audio_features"] = audio_result["embeddings"]

        # Process video if provided
        if video:
            video_path = os.path.join(temp_dir, video.filename)
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)

            video_result = video_processor.process_video(video_path, temp_dir)
            features["video_features"] = video_result["frame_features"].mean(axis=0)  # Simple averaging

            # Generate highlights
            if "multimodal_summary" not in results:
                results["multimodal_summary"] = {}

            highlights_path = os.path.join(temp_dir, "highlights.mp4")
            summarizer.video_summarizer.generate_highlights(
                video_path,
                video_result["frame_features"],
                video_result["frames"],
                video_result.get("audio_features", {}).get("embeddings"),
                output_path=highlights_path
            )

            # Copy highlights to a persistent location
            output_dir = "static/highlights"
            os.makedirs(output_dir, exist_ok=True)

            unique_id = str(uuid.uuid4())
            output_path = os.path.join(output_dir, f"highlight_{unique_id}.mp4")
            shutil.copy(highlights_path, output_path)

            results["multimodal_summary"]["video_highlights"] = f"/static/highlights/highlight_{unique_id}.mp4"

        # Classification based on available features
        if features:
            class_id = classifier.predict(**features)[0].item()
            results["classification"] = {
                "class_id": class_id,
                "class_name": f"Class {class_id}",  # Replace with actual class names
                "confidence": 0.85  # Placeholder
            }

        return results

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


@app.get("/knowledge-graph/query", response_model=dict)
async def query_knowledge_graph(query: str):
    """Execute a query on the knowledge graph"""
    try:
        results = neo4j_interface.query_knowledge_graph(query)
        return {"results": [dict(record) for record in results]}
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/healthcheck")
async def healthcheck():
    """API health check endpoint"""
    return {"status": "healthy"}


# Start the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)