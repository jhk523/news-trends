"""
pip install azure-ai-textanalytics 로 패키지 설치 후 사용
"""

import sys

sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages')

# noinspection PyPackageRequirements
from azure.ai.textanalytics import TextAnalyticsClient, TextAnalyticsApiKeyCredential

key = 'dbeb8ec261d0450298de3f7e9386494a'
endpoint = 'https://koreacentral.api.cognitive.microsoft.com/'

text_analytics_client = TextAnalyticsClient(endpoint, TextAnalyticsApiKeyCredential(key))


def main():
    documents = [
        "이 영화 참 재밌다.",
        "너무 잠온다.",
        "I hate you."
    ]

    response = text_analytics_client.analyze_sentiment(documents, language="kor")  # Korean
    # response = text_analytics_client.analyze_sentiment(documents, language="en")  # English
    result = [doc for doc in response if not doc.is_error]

    for doc in result:
        print("Overall sentiment: {}".format(doc.sentiment))
        print("Scores: positive={0:.3f}; neutral={1:.3f}; negative={2:.3f} \n".format(
            doc.sentiment_scores.positive,
            doc.sentiment_scores.neutral,
            doc.sentiment_scores.negative,
        ))


if __name__ == '__main__':
    main()
