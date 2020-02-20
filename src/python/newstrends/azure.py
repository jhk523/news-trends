"""
pip install azure-ai-textanalytics 로 패키지 설치 후 사용
"""
import json
import os

import numpy as np
from azure.ai.textanalytics import TextAnalyticsClient, TextAnalyticsApiKeyCredential

_ROOT_DIR = os.path.abspath(__file__ + '/../../../../')
_CLIENT = None


def load_client():
    global _CLIENT
    if _CLIENT is None:
        path = os.path.join(_ROOT_DIR, 'data/azure_info.json')
        azure_info = json.loads(open(path).read())
        key = azure_info['key']
        endpoint = azure_info['endpoint']
        _CLIENT = TextAnalyticsClient(endpoint, TextAnalyticsApiKeyCredential(key))
    return _CLIENT


def compute_scores(documents):
    max_queries = 1000
    scores = []
    while len(scores) < len(documents):
        index = len(scores)
        docs = documents[index:index + max_queries]
        response = load_client().analyze_sentiment(docs, language='kor')
        for doc in response:
            if doc.is_error:
                raise ValueError(doc)
            score = [doc.sentiment_scores.positive,
                     doc.sentiment_scores.neutral,
                     doc.sentiment_scores.negative]
            scores.append(score)
    return np.array(scores)


def print_scores(documents, scores):
    sentiments = ['positive', 'neutral', 'negative']
    for t, s in zip(documents, scores):
        scores_str = ', '.join(f'{e:.3f}' for e in s)
        print(f'Document: {t}')
        print(f'Sentiment: {sentiments[np.argmax(s)]}')
        print(f'Scores: ({scores_str})')
        print()
