"""
pip install azure-ai-textanalytics 로 패키지 설치 후 사용
"""
import json
import os

import numpy as np

_ROOT_DIR = os.path.abspath(__file__ + '/../../../../')
_CLIENT = None


def load_client():
    from azure.ai.textanalytics import TextAnalyticsClient, TextAnalyticsApiKeyCredential
    global _CLIENT
    if _CLIENT is None:
        path = os.path.join(_ROOT_DIR, 'data/azure_info.json')
        azure_info = json.loads(open(path).read())
        key = azure_info['key']
        endpoint = azure_info['endpoint']
        _CLIENT = TextAnalyticsClient(endpoint, TextAnalyticsApiKeyCredential(key))
    return _CLIENT


def send_query(documents):
    scores = []
    response = load_client().analyze_sentiment(documents, language='kor')
    for doc in response:
        if doc.is_error:
            raise ValueError(doc)
        score = [doc.sentiment_scores.positive,
                 doc.sentiment_scores.neutral,
                 doc.sentiment_scores.negative]
        scores.append(score)
    return scores


def compute_scores(documents, max_records=1000):
    documents_split = []
    for i in range(0, len(documents), max_records):
        documents_split.append(documents[i:i + max_records])
    scores_list = [send_query(d) for d in documents_split]
    # scores_list = multiprocessing.Pool().map(send_query, documents_split)
    return np.array([e for s in scores_list for e in s])
