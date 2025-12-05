import difflib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class OCRScorer:
    """
    Advanced Scoring Engine for OCR comparisons.
    Metrics:
    1. Levenshtein Similarity: Measures structural and character precision.
    2. Cosine Similarity: Measures word/content presence (ignoring order).
    """

    def normalize(self, text):
        """
        Standardizes text: lowercase, single spaces, removed jagged edges.
        """
        if not text: return ""
        text = text.lower()
        # Replace newlines/tabs with spaces
        text = re.sub(r'[\n\t\r]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def levenshtein_score(self, s1, s2):
        """
        Returns similarity 0-100 based on edit distance.
        Good for checking if Layout and Spelling are perfect.
        """
        # difflib.SequenceMatcher.ratio() is a robust pythonic equivalent 
        # to (1 - distance/length).
        matcher = difflib.SequenceMatcher(None, s1, s2)
        return matcher.ratio() * 100

    def cosine_score(self, s1, s2):
        """
        Returns similarity 0-100 based on Word Frequency vectors (TF-IDF).
        Good for checking if the CONTENT is there, even if jumbled.
        """
        # Handle empty cases
        if not s1.strip() or not s2.strip():
            return 0.0 if s1 != s2 else 100.0

        try:
            vectorizer = TfidfVectorizer(
                analyzer='word',
                token_pattern=r"(?u)\b\w\w+\b" # Standard word tokenizer
            )
            
            # Create TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform([s1, s2])
            
            # Calculate Cosine Similarity
            # Returns a matrix [[1, score], [score, 1]]
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return score * 100
            
        except ValueError:
            # Usually happens if no valid words are found (e.g. only symbols)
            return 0.0

    def evaluate(self, prediction, truth):
        """
        Returns a dictionary containing both metrics.
        """
        clean_pred = self.normalize(prediction)
        clean_truth = self.normalize(truth)

        return {
            "levenshtein": self.levenshtein_score(clean_pred, clean_truth),
            "cosine": self.cosine_score(clean_pred, clean_truth),
        }