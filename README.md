# 언론사 편향성을 고려한 뉴스 트렌드 분석 시스템

본 코드 저장소는 KPMG Ideathon에 참가한 씨그램(Seagram) 팀의 작업 공간입니다.

## 개요

뉴스 트렌드를 파악하는 것은 모든 기업 및 기관에서 중요한 문제입니다. 그러나, 현재까지 개발된 뉴스 트렌드
분석 서비스는 한국의 언론사 편향성을 고려하지 않는다는 문제점을 가지고 있습니다. 예를 들어 대표적인 국내
언론사인 조선일보와 한겨레는 동일한 사건에 대해 전혀 다른 방향으로 보도하는 경우가 많습니다. 다음 예제는
2020년 2월 21일에 네이버 메인에 올라온, 코로나 19 바이러스와 문재인 대통령에 대한 두 언론사의 기사를
가져 온 것입니다. 이러한 편항성을 고려하지 않는다면 대중과 언론의 생각을 잘못 파악할 가능성이 높습니다.
- **조선일보:** 대통령 부부, 이시국에 '파안대소'… 일부 네티즌 비판
- **한겨레:** 문 대통령 “상황 엄중, 빠르고 강력한 대책 시행”
 
우리는 본 프로젝트에서 언론사 편향성을 고려하는 새로운 뉴스 트렌드 분석 시스템을 제안하고, 이를 홈페이지
형태로 구현한 프로토타입을 제작합니다. 본 시스템이 지원하는 기능은 다음과 같습니다.
- 입력한 키워드에 대해 각 언론사의 긍정/부정 반응도 계산하기
- 입력한 키워드에 대한 각 언론사의 향후 트렌드 예측하기
- 입력한 (임의의) 문장에 대한 언론사 편향성 계산하기  

## 기능 설명

본 프로토타입은 뉴스 트렌드를 분석하기 위한 여러 가지 기능을 포함하고 있습니다. 각 기능은 다음과 같은
기술적 요소로 이루어져 있습니다.

### 핫 키워드 탐색

최근 이슈가 되는 핫 키워드를 탐색하는 기능은 크게 두 가지 요소로 이루어집니다. 먼저, 데이터베이스 내의
모든 기사 제목을 받아 와 전처리를 실행합니다. 키워드 탐색은 주로 명사를 타겟으로 하고, 기사 제목에서는
대부분의 명사가 띄어쓰기로 구분되어 있기 때문에 복잡한 형태소 분석기를 사용하지 않고 간단한 휴리스틱 기반의
전처리 함수를 구현하여 사용합니다. 그런 다음, 키워드별로 최근 N일 동안의 등장 횟수를 센 다음 횟수가 가장
큰 순서로 나열합니다. 우리는 실험을 통해 이런 단순한 기법이 복잡한 형태소 분석기를 사용했을 때보다 더욱
직관적이고, 정성적으로 더 나은 결과로 이어진다는 것을 확인하였습니다.

### 기사의 논조 분석

우리는 특정 키워드에 대한 언론사 편향성을 계산하기 위해 [Microsoft Azure Text Analytics](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/)
서비스를 활용하였습니다. 특정 키워드가 주어졌을 때 먼저 해당 키워드를 포함하는 모든 기사를 검색합니다.
그런 다음, MS Azure API를 통해 각 기사 제목의 논조를 세 가지 감정(긍정, 중립, 부정)에 대한 점수로
계산합니다. 일반적으로 기사의 제목은 중립적이라고 생각하기 쉬우나, 우리는 실험을 통해 많은 수의 기사가
부정적인 감정을 내포하고 있으며 특정 대상을 비판하는 방식으로 쓰인다는 것을 확인하였습니다. 이렇게 수집된
기사와 논조를 먼저 언론사에 대해 분류한 뒤, 각 언론사가 키워드를 주로 다루는 방식을 출력합니다.

## 데이터 정보

본 프로젝트에서 사용하는 데이터는 RSS 서비스를 이용해 각 언론사로부터 직접 수집한 것입니다. 개인 서버에서
일 분에 한 번 전체 기사의 언론사, 제목, 그리고 요약문을 가져와 MySQL 데이터베이스에 저장합니다. 
테스트를 마친 후 2월 12일부터 정상적인 스크래핑을 시작하였고 2020년 2월 21일 기준으로 데이터베이스에
저장된 기사 수는 약 3만 6천 개입니다. 각 기사는 다음과 같은 형식으로 저장됩니다.
- `2020-02-12 13:44:24`, `조선일보`, `“봉준호 감독 미국서 유명해요?”...봉준호 아들이 들은말`, ...

## 코드 정보

본 코드 저장소는 크게 두 가지 모듈을 담고 있습니다.
- `src/python`: 데이터 스크래핑, 전처리, 기계 학습 모델 개발 등 전반적인 코드를 저장하고 있습니다.
- `src/website`: 웹사이트 프로토타입에 대한 코드를 저장하고 있습니다.

Python 3.6 언어를 권장하나 Python 3.7에서도 정상적으로 구동될 수 있습니다. 소스 코드 실행을 위해
필요한 패키지는 `requirements.txt` 파일에 정리되어 있습니다. MySQL 접속 정보 등 일부 데이터는 본
저장소에 포함되어 있지 않기 때문에 이러한 데이터를 `data` 폴더에 따로 저장해야 합니다. 
 
## 팀 멤버

본 코드 저장소는 씨그램(Seagram) 팀의 다음 네 명의 멤버가 함께 작성하였습니다.

- 유재민 (leader, jaeminyoo@snu.ac.kr)
- 김정훈 (joseph.junghoon.kim@gmail.com)
- 조민용 (chominyong@gmail.com)
- 이성민 (ligi214@snu.ac.kr)
