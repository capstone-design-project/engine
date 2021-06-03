# engine

유튜브 영상의 `video ID` 입력시 그 영상 속 대화의 난이도를 반환한다.

## 구현 수준
- `word_analyzer.py` : 텍스트 입력시 해당 텍스트 내의 단어들을 통해 전체 문장의 CEFR 점수와 가독성 점수를 반환
- `script_analyzer.py` : video ID 입력시 캡션 수집 후 대화의 속도 반환
- `core.py` : 위 두가지 분석기를 통해 얻은 분석 결과를 기계 학습 가능한 형태로 반환
