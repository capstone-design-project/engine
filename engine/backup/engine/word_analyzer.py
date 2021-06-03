# 다양한 단어 메타 데이터 자료를 활용하여 유의미한 정보를 추출하기 위한 도구들
import nltk
import json
from collections import defaultdict
from nltk.tag.brill import Word
from numpy import False_
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import wordnet as wn
lemma = WordNetLemmatizer()
stem = PorterStemmer()
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

pos_map = defaultdict(lambda: wn.NOUN)
pos_map['J'] = wn.ADJ
pos_map['V'] = wn.VERB
pos_map['R'] = wn.ADV


class WordAnalyzer:
    Dale_Chall_List = pd.read_csv(
        "/Users/hwangbyeonghyeon/dev/capstone_engine/engine/engine_server/engine/word_data/Dale_Chall_List.csv", index_col=0).values.tolist()
    CEFR2score = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6, 'N': 6}
    oxford_CEFR = pd.read_csv(
        '/Users/hwangbyeonghyeon/dev/capstone_engine/engine/engine_server/engine/word_data/oxford_5000_CEFR.csv', index_col=0).values.tolist()
    japanese_CEFR = pd.read_csv(
        '/Users/hwangbyeonghyeon/dev/capstone_engine/engine/engine_server/engine/word_data/cefr_japanese.csv', index_col=0).values.tolist()
    pg_freq_10000 = pd.read_csv(
        '/Users/hwangbyeonghyeon/dev/capstone_engine/engine/engine_server/engine/word_data/project_gutenberg_10000.csv', index_col=0).values.tolist()
    simpson_freq_5000 = pd.read_csv(
        '/Users/hwangbyeonghyeon/dev/capstone_engine/engine/engine_server/engine/word_data/simpson_freq_5000.csv', index_col=0).values.tolist()
    tv_freq_10000 = pd.read_csv(
        '/Users/hwangbyeonghyeon/dev/capstone_engine/engine/engine_server/engine/word_data/tv_movie_script_10000.csv', index_col=0).values.tolist()

    ###### 전처리부 #########
    # 텍스트 전처리- 영어 이외에 전부 공백처리, 전부 소문자화
    def preprocessing(self, text):
        clean_text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
        return clean_text
    # 텍스트를 단어 리스트로 변환(stopword 적용)

    def text2list(self, text):
        return [word for word in text.lower().split() if not word in set(stopwords.words('english'))]
    # stemming , 어간 추출을 통해 단어를 원형으로 복원한다.

    def stemming(self, text2list):
        tag_list = pos_tag(text2list)
        # return list(map(lambda x : stem.stem(x),text2list))
        return list(map(lambda x: lemma.lemmatize(x[0], pos_map[x[1][0]]), tag_list))

    ###### DCL 부 ############
    # Dale Chall List 에 기반한 uncommon word 확인(리스트에 없으면 uncommon)
    def checkDCL(self, word):
        if word in map(lambda x: x[0], self.Dale_Chall_List):
            return False  # common
        else:
            return True  # uncommon
    # uncommon 단어 추출

    def extractUncommon(self, text2list):
        uncommon_list = []
        for word in text2list:
            if self.checkDCL(word) == True:
                uncommon_list.append(word)
        return uncommon_list

    ######## CEFR 부 ###########
    # CEFR 종류에 따른 cefr 점수 반환(없으면 N 반환)
    def checkCEFR(self, word, CEFR=oxford_CEFR):
        for item in CEFR:
            if item[0] == word:
                return item[1]
        return 'N'
    # 난이도별 단어 추출

    def extractCEFR(self, text2list, CEFR=oxford_CEFR):
        CEFR_words = {'A1': [], 'A2': [], 'B1': [],
                      'B2': [], 'C1': [], 'C2': [], 'N': []}
        for word in text2list:
            CEFR_words[self.checkCEFR(word, CEFR)].append(word)
        return CEFR_words

    ########### freq 부 ###########
    # freq 부는 stemming을 하지 않은 텍스트로 적용해야함
    # 빈도수 랭킹을 반환 (없으면 -1)
    def checkFreq(self, word, Freq=tv_freq_10000):
        for item in Freq:
            if item[1] == word:
                return item[0]
        return -1
    # 빈도수 랭킹을 cefr 점수로 변환

    def extractFreq(self, text2list, Freq=tv_freq_10000):
        CEFR_words = {'A1': [], 'A2': [], 'B1': [],
                      'B2': [], 'C1': [], 'C2': [], 'N': []}
        for word in text2list:
            if self.checkFreq(word, Freq) == -1:
                CEFR_words['N'].append(word)
            elif self.checkFreq(word, Freq) <= 600:
                CEFR_words['A1'].append(word)
            elif self.checkFreq(word, Freq) <= 1200:
                CEFR_words['A2'].append(word)
            elif self.checkFreq(word, Freq) <= 2500:
                CEFR_words['B1'].append(word)
            elif self.checkFreq(word, Freq) <= 5000:
                CEFR_words['B2'].append(word)
            elif self.checkFreq(word, Freq) <= 10000:
                CEFR_words['C1'].append(word)
            elif self.checkFreq(word, Freq) <= 20000:
                CEFR_words['C2'].append(word)
        return CEFR_words

    ####### 최종 아웃풋 ########
    # cefr 점수 평균 계산

    def calAvgCEFR(self, classified_words, total_words):
        score_sum = 0
        CEFR_level = classified_words.keys()
        for level in CEFR_level:
            score_sum += len(classified_words[level])*self.CEFR2score[level]
        return score_sum/total_words
    # Dale Chall 가독성 점수 계산

    def calReadability(self, uncommon_ratio, total_words, total_sentences):
        return 0.1579*uncommon_ratio + 0.0496*(total_words/total_sentences)

    def analyzeText(self, text):
        output_json = {
            'Input_text': '',
            'Total_words': 0,
            'Total_sentences': 0,
            'Total_avg_CEFR': 0,
            'DC_Readability': 0,
            'DCL': {
                'uncommon_ratio': 0,
                'uncommon_words': [],
            },
            'CEFR': {
                'Oxford': {
                    'avg_CEFR': 0,
                    'classified_words': [],
                },
                'Japanese': {
                    'avg_CEFR': 0,
                    'classified_words': [],
                }
            },
            'Freq': {
                'Tv': {
                    'avg_CEFR': 0,
                    'classified_words': [],
                },
                'Simpson': {
                    'avg_CEFR': 0,
                    'classified_words': [],
                },
                'Gutenberg': {
                    'avg_CEFR': 0,
                    'classified_words': []
                }
            }
        }
        output_json['Input_text'] = text
        clean_text = self.preprocessing(text)
        word_list = self.text2list(clean_text)
        word_list_stem = list(set(self.stemming(word_list)))
        total_words = len(word_list)
        total_unique_words = len(word_list_stem)
        output_json['Total_words'] = total_words
        output_json['Total_unique_words'] = total_unique_words
        output_json['Total_sentences'] = len(sent_tokenize(text))

        output_json['DCL']['uncommon_words'] = self.extractUncommon(
            word_list_stem)
        output_json['DCL']['uncommon_ratio'] = len(
            output_json['DCL']['uncommon_words'])/total_unique_words*100
        output_json['DC_Readability'] = self.calReadability(
            output_json['DCL']['uncommon_ratio'], total_words, output_json['Total_sentences'])

        output_json['CEFR']['Oxford']['classified_words'] = self.extractCEFR(
            word_list_stem, self.oxford_CEFR)
        output_json['CEFR']['Oxford']['avg_CEFR'] = self.calAvgCEFR(
            output_json['CEFR']['Oxford']['classified_words'], total_unique_words)
        output_json['CEFR']['Japanese']['classified_words'] = self.extractCEFR(
            word_list_stem, self.japanese_CEFR)
        output_json['CEFR']['Japanese']['avg_CEFR'] = self.calAvgCEFR(
            output_json['CEFR']['Japanese']['classified_words'], total_unique_words)

        output_json['Freq']['Tv']['classified_words'] = self.extractFreq(
            word_list_stem, self.tv_freq_10000)
        output_json['Freq']['Tv']['avg_CEFR'] = self.calAvgCEFR(
            output_json['Freq']['Tv']['classified_words'], total_unique_words)
        output_json['Freq']['Simpson']['classified_words'] = self.extractFreq(
            word_list_stem, self.simpson_freq_5000)
        output_json['Freq']['Simpson']['avg_CEFR'] = self.calAvgCEFR(
            output_json['Freq']['Simpson']['classified_words'], total_unique_words)
        output_json['Freq']['Gutenberg']['classified_words'] = self.extractFreq(
            word_list_stem, self.pg_freq_10000)
        output_json['Freq']['Gutenberg']['avg_CEFR'] = self.calAvgCEFR(
            output_json['Freq']['Gutenberg']['classified_words'], total_unique_words)

        output_json['Total_avg_CEFR'] = (output_json['CEFR']['Oxford']['avg_CEFR']+output_json['CEFR']['Japanese']['avg_CEFR']+output_json['Freq']
                                         ['Tv']['avg_CEFR']+output_json['Freq']['Simpson']['avg_CEFR']+output_json['Freq']['Gutenberg']['avg_CEFR']) / 5
        return json.dumps(output_json, indent=4)
