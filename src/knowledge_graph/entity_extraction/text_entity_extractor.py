# src/knowledge_graph/entity_extraction/text_entity_extractor.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import spacy
from spacy.matcher import Matcher


class TextEntityExtractor:
    def __init__(self, model_name="dslim/bert-base-NER"):
        # BERT-based NER model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Spacy for relation extraction
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)

        # Define patterns for relation extraction
        self.setup_relation_patterns()

    def setup_relation_patterns(self):
        """Set up patterns for relation extraction"""
        # Pattern for "works for" relation
        works_for = [
            [{"POS": "PROPN"}, {"LOWER": "works"}, {"LOWER": "for"}, {"POS": "PROPN"}],
            [{"POS": "PROPN"}, {"LOWER": "is"}, {"LOWER": "employed"}, {"LOWER": "by"}, {"POS": "PROPN"}]
        ]

        # Pattern for "located in" relation
        located_in = [
            [{"POS": "PROPN"}, {"LOWER": "is"}, {"LOWER": "located"}, {"LOWER": "in"}, {"POS": "PROPN"}],
            [{"POS": "PROPN"}, {"LOWER": "in"}, {"POS": "PROPN"}]
        ]

        # Add patterns to matcher
        self.matcher.add("WORKS_FOR", works_for)
        self.matcher.add("LOCATED_IN", located_in)

    def extract_entities(self, text):
        """Extract named entities from text using BERT NER"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Get entity labels
        entities = []
        current_entity = {}

        # Convert token predictions to entities
        input_ids = inputs["input_ids"][0].tolist()
        for idx, (token_id, pred_id) in enumerate(zip(input_ids, predictions[0].tolist())):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            tag = self.model.config.id2label[pred_id]

            if tag.startswith("B-"):
                # Beginning of a new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"text": token, "type": tag[2:], "start": idx, "end": idx}
            elif tag.startswith("I-") and current_entity:
                # Inside an entity
                current_entity["text"] += " " + token
                current_entity["end"] = idx
            elif tag == "O":
                # Outside any entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = {}

        # Add the last entity if there's one
        if current_entity:
            entities.append(current_entity)

        return entities

    def extract_relations(self, text):
        """Extract relations between entities using spaCy patterns"""
        doc = self.nlp(text)
        relations = []

        # Find matches for relation patterns
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            # Get the matched span
            span = doc[start:end]

            # Get the relation type
            relation_type = self.nlp.vocab.strings[match_id]

            # Find the entities in the matched span
            entities = [(ent.text, ent.label_) for ent in span.ents]

            if len(entities) >= 2:
                # Create a relation between entities
                relation = {
                    "source": entities[0][0],
                    "source_type": entities[0][1],
                    "relation": relation_type,
                    "target": entities[-1][0],
                    "target_type": entities[-1][1],
                    "sentence": span.text
                }
                relations.append(relation)

        return relations

    def process_document(self, text):
        """Process a document to extract entities and relations"""
        entities = self.extract_entities(text)
        relations = self.extract_relations(text)

        return {
            "entities": entities,
            "relations": relations
        }