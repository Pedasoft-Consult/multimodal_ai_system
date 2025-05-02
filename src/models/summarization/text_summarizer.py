# src/models/summarization/text_summarizer.py
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import networkx as nx
from nltk.tokenize import sent_tokenize
import numpy as np


class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        # Load BART for abstractive summarization
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def abstractive_summarize(self, text, max_length=150, min_length=40):
        """Generate an abstractive summary using BART"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    def extractive_summarize(self, text, num_sentences=3):
        """Generate an extractive summary using TextRank algorithm"""
        # Split text into sentences
        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Create a graph
        similarity_matrix = self._build_similarity_matrix(sentences)
        nx_graph = nx.from_numpy_array(similarity_matrix)

        # Apply PageRank
        scores = nx.pagerank(nx_graph)

        # Rank sentences by score
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)

        # Get top sentences
        top_sentences = [s for _, i, s in ranked_sentences[:num_sentences]]

        # Reorder sentences according to their original position
        ordered_sentences = sorted([(i, s) for _, i, s in ranked_sentences[:num_sentences]])

        # Return the summary
        summary = " ".join([s for _, s in ordered_sentences])

        return summary

    def _build_similarity_matrix(self, sentences):
        """Build a similarity matrix for sentences"""
        # Simple cosine similarity matrix (can be improved with embeddings)
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])

        return similarity_matrix

    def _sentence_similarity(self, sent1, sent2):
        """Calculate similarity between two sentences (simple word overlap)"""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        # Calculate Jaccard similarity
        if len(words1) == 0 or len(words2) == 0:
            return 0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union