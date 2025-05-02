# src/knowledge_graph/entity_extraction/image_entity_extractor.py
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests


class ImageEntityExtractor:
    def __init__(self, model_name="facebook/detr-resnet-50"):
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)

    def extract_objects(self, image_path, threshold=0.9):
        """Extract objects from an image"""
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        # Extract detected objects
        objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Convert box coordinates
            box = [round(i, 2) for i in box.tolist()]

            # Get object class label
            class_label = self.model.config.id2label[label.item()]

            # Add object to list
            objects.append({
                "label": class_label,
                "score": round(score.item(), 3),
                "box": box
            })

        return objects

    def extract_scene(self, image_path):
        """Extract scene information from an image"""
        # Load scene classification model (e.g., ResNet)
        # For simplicity, we'll return a placeholder
        return {"scene": "indoor"}

    def map_to_knowledge_graph(self, objects):
        """Map detected objects to knowledge graph concepts"""
        # Define object hierarchies and relationships
        object_mappings = {
            "person": {"type": "Person", "relations": ["uses", "wears"]},
            "cat": {"type": "Animal", "relations": ["lives in", "pet of"]},
            "dog": {"type": "Animal", "relations": ["lives in", "pet of"]},
            "chair": {"type": "Furniture", "relations": ["part of", "used by"]},
            "car": {"type": "Vehicle", "relations": ["owned by", "driven by"]},
            # Add more mappings as needed
        }

        # Process each object
        knowledge_objects = []
        for obj in objects:
            label = obj["label"]

            # Get mapping if available
            mapping = object_mappings.get(label, {"type": "Object", "relations": []})

            # Create knowledge graph node
            knowledge_obj = {
                "label": label,
                "type": mapping["type"],
                "possible_relations": mapping["relations"],
                "confidence": obj["score"],
                "position": obj["box"]
            }

            knowledge_objects.append(knowledge_obj)

        return knowledge_objects

    def process_image(self, image_path):
        """Process an image to extract knowledge graph entities"""
        # Extract objects
        objects = self.extract_objects(image_path)

        # Extract scene info
        scene = self.extract_scene(image_path)

        # Map to knowledge graph concepts
        knowledge_objects = self.map_to_knowledge_graph(objects)

        return {
            "objects": objects,
            "scene": scene,
            "knowledge_objects": knowledge_objects
        }