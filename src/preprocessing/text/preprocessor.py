# src/preprocessing/text/preprocessor.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from transformers import BertTokenizer, BertModel
import torch


class TextPreprocessor:
    def __init__(self, model_name='bert-base-uncased'):
        # Download required NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def clean_text(self, text):
        """Basic text cleaning: lowercase, remove special chars, etc."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return tokens

    def get_bert_embeddings(self, text, max_length=512):
        """Generate BERT embeddings for a text document"""
        # Tokenize and prepare for BERT
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length,
                                truncation=True, padding="max_length")

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use [CLS] token embedding as document representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings

    def process_document(self, document):
        """Full preprocessing pipeline for a text document"""
        cleaned_text = self.clean_text(document)
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        embeddings = self.get_bert_embeddings(cleaned_text)

        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'embeddings': embeddings
        }