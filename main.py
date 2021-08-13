import os
from typing import Dict, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


student_filenames = [filename for filename in os.listdir("./Student's works") if filename.endswith('.txt')]
student_notes = [open(filename).read() for filename in student_filenames]

corpus = {k: v for k, v in zip(student_filenames, student_notes)}

class PlagiarismChecker:
    def __init__(self):
        self._vectorizer = TfidfVectorizer()

    def vectorize(self, texts: List):
        return self._vectorizer.fit_transform(texts).toarray()

    @staticmethod
    def similarity(self, text1, text2):
        return cosine_similarity(text1, text2)