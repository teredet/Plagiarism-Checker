from operator import index
import os
from typing import Dict, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


student_filenames = [filename for filename in os.listdir("./Student's works") if filename.endswith('.txt')]
student_notes = [open(f"./Student's works/{filename}").read() for filename in student_filenames]

corpus = {k: v for k, v in zip(student_filenames, student_notes)}

class PlagiarismChecker:
    def __init__(self):
        self._vectorizer = TfidfVectorizer()

    def vectorize(self, texts: List):
        return self._vectorizer.fit_transform(texts).toarray()

    @staticmethod
    def similarity(text1, text2):
        return cosine_similarity(text1, text2)

    def check(self, corpus: Dict):
        vectorized_texts = self.vectorize(corpus.values())
        filenames = list(corpus.keys())
        results = pd.DataFrame(columns=filenames, index=filenames)
        similarity_results = self.similarity(vectorized_texts, vectorized_texts)
        for filename, row in zip(filenames, similarity_results):
            results[filename] = row
        return results

print(PlagiarismChecker().check(corpus))
