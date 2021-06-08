import json
import pickle
import numpy as np

from theano.tensor.basic import outer, sub
from word_analyzer import WordAnalyzer
from script_analyzer import ScriptAnalyzer
from youtube_crawler import youtubeCrawler
from punctuator import Punctuator


target_video_id = "TLnUJzueBOQ"
cbc_kid = 'SuSTBXGiOsw'
comedy_central = 'fKSiol1uczc'
bbc_news = 'hFAROEKiHl8'

WA = WordAnalyzer()
SA = ScriptAnalyzer()
# Place model files at '~/.punctuator' Download ULR : https://drive.google.com/drive/folders/0B7BsN5f2F1fZQnFsbzJ3TWxxMms
# model list : Demo-Europarl-EN.pcl INTERSPEECH-T-BRNN-pre.pcl INTERSPEECH-T-BRNN.pcl
P = Punctuator('Demo-Europarl-EN.pcl')


# analyze for making training set
def analyzeAll(videoId):
    sa_result = json.loads(SA.analyzeScript(videoId))
    print('script analyze ok')
    punc_script = P.punctuate(sa_result['script'])  # 문장부호 포함된 스크립트
    print('punctuator ok')
    wa_result = json.loads(WA.analyzeText(punc_script))
    print('word analyze ok')

    analyze_result = {}

    analyze_result['videoId'] = sa_result['videoId']
    analyze_result['script'] = punc_script
    analyze_result['totalWords'] = wa_result['Total_words']
    analyze_result['totalUniqueWords'] = wa_result['Total_unique_words']
    analyze_result['totalSentences'] = wa_result['Total_sentences']
    analyze_result['avgSyllPerSec'] = sa_result['avgSyllPerSec']
    analyze_result['avgCEFRScore'] = wa_result['Total_avg_CEFR']
    analyze_result['avgWordCEFR'] = wa_result['Word_avg_CEFR']
    analyze_result['avgFreqCEFR'] = wa_result['Freq_avg_CEFR']
    analyze_result['readability'] = wa_result['DC_Readability']
    analyze_result['avgSentenceLength'] = wa_result['avg_sentence_length']
    analyze_result['uncommonRatio'] = wa_result['DCL']['uncommon_ratio']

    def calCEFRRatio(targetList, subject):
        #targetList :  ['CEFR', 'Freq']
        #subject : ['Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg']
        cefr_sum = [0, 0, 0, 0, 0, 0, 0]
        for checker in targetList:
            for cefr in subject:
                if cefr in wa_result[checker]:
                    for idx, level in enumerate(['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'N']):
                        cefr_sum[idx] += len(wa_result[checker]
                                             [cefr]['classified_words'][level])
        cefr_ratio = list(
            map(lambda x: (x/len(subject))/analyze_result['totalUniqueWords']*100, cefr_sum))
        return cefr_ratio

    cefr_ratio = calCEFRRatio(
        ['CEFR', 'Freq'], ['Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'])
    analyze_result['totalEasyRatio'] = cefr_ratio[0]+cefr_ratio[1]
    analyze_result['totalMiddleRatio'] = cefr_ratio[2]+cefr_ratio[3]
    analyze_result['totalHardRatio'] = cefr_ratio[4] + \
        cefr_ratio[5]+cefr_ratio[6]
    cefr_ratio = calCEFRRatio(['CEFR'], ['Oxford', 'Japanese'])
    analyze_result['wordEasyRatio'] = cefr_ratio[0]+cefr_ratio[1]
    analyze_result['wordMiddleRatio'] = cefr_ratio[2]+cefr_ratio[3]
    analyze_result['wordHardRatio'] = cefr_ratio[4] + \
        cefr_ratio[5]+cefr_ratio[6]
    cefr_ratio = calCEFRRatio(['Freq'], ['Tv', 'Simpson', 'Gutenberg'])
    analyze_result['FreqEasyRatio'] = cefr_ratio[0]+cefr_ratio[1]
    analyze_result['FreqMiddleRatio'] = cefr_ratio[2]+cefr_ratio[3]
    analyze_result['FreqHardRatio'] = cefr_ratio[4] + \
        cefr_ratio[5]+cefr_ratio[6]

    print('analyze ok')

    return json.dumps(analyze_result, indent=4)


# 분석된 json으로 예측 결과 반환
# old feature list = ['avgSyllPerSec', 'avgCEFRScore', 'readability', 'uncommonRatio', 'A1ratio', 'A2ratio', 'B1ratio', 'B2ratio', 'C1ratio', 'C2ratio', 'Nratio']
# new feature list = ['avgSyllPerSec', 'avgCEFRScore','avgWordCEFR','avgFreqCEFR', 'readability','avgSentenceLength', 'uncommonRatio', 'totalEasyRatio','totalMiddleRatio','totalHardRatio','wordEasyRatio','wordMiddleRatio','wordHardRatio','FreqEasyRatio','FreqMiddleRatio','FreqHardRatio']
def predictDifficulty(analyzed_result, feature_list, model):
    video_data = []
    for feature in analyzed_result.keys():
        if feature in feature_list:
            video_data.append(analyzed_result[feature])

    with open('models/'+model, 'rb') as model_file:
        model = pickle.load(model_file)
    return model.predict([video_data]).tolist()[0]


# 분석 결과 와 예측 결과를 반환
def analyzeNpredict(videoId):
    sa_result = json.loads(SA.analyzeScript(videoId))
    print('script analyze ok')
    try:
        punc_script = P.punctuate(sa_result['script'])  # 문장부호 포함된 스크립트
        print('punctuator ok')
    except NotImplementedError:
        print('empty caption! Cant analyze')
        return json.dumps({}, indent=4)

    wa_result = json.loads(WA.analyzeText(punc_script))
    print('word analyze ok')

    analyze_result = {}

    analyze_result['videoId'] = sa_result['videoId']
    analyze_result['script'] = punc_script
    analyze_result['totalWords'] = wa_result['Total_words']
    analyze_result['totalUniqueWords'] = wa_result['Total_unique_words']
    analyze_result['totalSentences'] = wa_result['Total_sentences']
    analyze_result['avgSyllPerSec'] = sa_result['avgSyllPerSec']
    analyze_result['avgCEFRScore'] = wa_result['Total_avg_CEFR']
    analyze_result['readability'] = wa_result['DC_Readability']
    analyze_result['uncommonRatio'] = wa_result['DCL']['uncommon_ratio']

    cefr_sum = [0, 0, 0, 0, 0, 0, 0]
    for checker in ['CEFR', 'Freq']:
        for cefr in ['Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg']:
            if cefr in wa_result[checker]:
                for idx, level in enumerate(['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'N']):
                    cefr_sum[idx] += len(wa_result[checker]
                                         [cefr]['classified_words'][level])
    cefr_ratio = list(
        map(lambda x: (x/5)/analyze_result['totalUniqueWords']*100, cefr_sum))

    analyze_result['A1ratio'] = cefr_ratio[0]
    analyze_result['A2ratio'] = cefr_ratio[1]
    analyze_result['B1ratio'] = cefr_ratio[2]
    analyze_result['B2ratio'] = cefr_ratio[3]
    analyze_result['C1ratio'] = cefr_ratio[4]
    analyze_result['C2ratio'] = cefr_ratio[5]
    analyze_result['Nratio'] = cefr_ratio[6]

    # 단어 리스트 반환
    analyze_result['uniqueList'] = wa_result['unique_words']
    analyze_result['uncommonList'] = wa_result['DCL']['uncommon_words']

    def makeWordSet(targetList, subject, level):
        wordSet = set()
        for target in targetList:
            for sub in subject:
                if sub in wa_result[target]:
                    for lv in level:
                        wordSet.update(wa_result[target]
                                       [sub]['classified_words'][lv])
        return wordSet

    analyze_result['easyWordList'] = makeWordSet(['CEFR', 'Freq'], [
                                                 'Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'], ['A1', 'A2'])
    analyze_result['middleWordList'] = makeWordSet(
        ['CEFR', 'Freq'], ['Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'], ['B1', 'B2'])
    analyze_result['middleWordList'] -= analyze_result['easyWordList']
    analyze_result['hardWordList'] = makeWordSet(['CEFR', 'Freq'], [
                                                 'Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'], ['C1', 'C2'])
    analyze_result['hardWordList'] -= (analyze_result['easyWordList']
                                       | analyze_result['middleWordList'])
    analyze_result['unrankedWordList'] = makeWordSet(['CEFR', 'Freq'], [
        'Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'], ['N'])
    analyze_result['unrankedWordList'] -= (analyze_result['easyWordList']
                                           | analyze_result['middleWordList'] | analyze_result['hardWordList'])
    analyze_result['easyWordList'] = list(analyze_result['easyWordList'])
    analyze_result['middleWordList'] = list(analyze_result['middleWordList'])
    analyze_result['hardWordList'] = list(analyze_result['hardWordList'])
    analyze_result['unrankedWordList'] = list(
        analyze_result['unrankedWordList'])

    print('analyze ok')

    model = 'capstone_model_RF.pkl'
    predict_feature = ['avgSyllPerSec', 'avgCEFRScore', 'readability', 'uncommonRatio',
                       'A1ratio', 'A2ratio', 'B1ratio', 'B2ratio', 'C1ratio', 'C2ratio', 'Nratio']
    difficulty = predictDifficulty(analyze_result, predict_feature, model)
    analyze_result['difficulty'] = difficulty

    print('predict ok')

    return json.dumps(analyze_result, indent=4)

# 모든 정보 반환


def analyzeComplete(videoId, youtubeApiKey):
    # 분석용 데이터
    try:
        analyze_result = json.loads(analyzeAll(videoId))
    except NotImplementedError:
        print("empty caption! Cant analyze")
        return json.dumps({}, indent=4)

    # 단어 분석
    wa_result = json.loads(WA.analyzeText(analyze_result['script']))

    # 단어 리스트 반환
    analyze_result['uniqueList'] = wa_result['unique_words']
    analyze_result['uncommonList'] = wa_result['DCL']['uncommon_words']

    def makeWordSet(targetList, subject, level):
        wordSet = set()
        for target in targetList:
            for sub in subject:
                if sub in wa_result[target]:
                    for lv in level:
                        wordSet.update(wa_result[target]
                                       [sub]['classified_words'][lv])
        return wordSet

    analyze_result['easyWordList'] = makeWordSet(['CEFR', 'Freq'], [
                                                 'Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'], ['A1', 'A2'])
    analyze_result['middleWordList'] = makeWordSet(
        ['CEFR', 'Freq'], ['Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'], ['B1', 'B2'])
    analyze_result['middleWordList'] -= analyze_result['easyWordList']
    analyze_result['hardWordList'] = makeWordSet(['CEFR', 'Freq'], [
                                                 'Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'], ['C1', 'C2'])
    analyze_result['hardWordList'] -= (analyze_result['easyWordList']
                                       | analyze_result['middleWordList'])
    analyze_result['unrankedWordList'] = makeWordSet(['CEFR', 'Freq'], [
        'Oxford', 'Japanese', 'Tv', 'Simpson', 'Gutenberg'], ['N'])
    analyze_result['unrankedWordList'] -= (analyze_result['easyWordList']
                                           | analyze_result['middleWordList'] | analyze_result['hardWordList'])
    analyze_result['easyWordList'] = list(analyze_result['easyWordList'])
    analyze_result['middleWordList'] = list(analyze_result['middleWordList'])
    analyze_result['hardWordList'] = list(analyze_result['hardWordList'])
    analyze_result['unrankedWordList'] = list(
        analyze_result['unrankedWordList'])

    print('analyze ok')

    # model = 'capstone_model_RF_upgrade_2.pkl' 예전버전
    model = 'capstone_model_RF_new.pkl'
    predict_feature = ['avgSyllPerSec', 'avgCEFRScore', 'avgWordCEFR', 'avgFreqCEFR', 'readability', 'avgSentenceLength', 'uncommonRatio', 'totalEasyRatio',
                       'totalMiddleRatio', 'totalHardRatio', 'wordEasyRatio', 'wordMiddleRatio', 'wordHardRatio', 'FreqEasyRatio', 'FreqMiddleRatio', 'FreqHardRatio']
    difficulty = predictDifficulty(analyze_result, predict_feature, model)
    analyze_result['difficulty'] = difficulty

    print('predict ok')

    try:
        analyze_result['videoInfo'] = json.loads(
            getVideoInfo(youtubeApiKey, videoId))
        analyze_result['videoInfo']['channelImage'] = getChannelImage(
            youtubeApiKey, analyze_result['videoInfo']['channelId'])

    except Exception:
        print('youtube api problem')
        analyze_result['videoInfo'] = {}

    print('video info ok')

    return json.dumps(analyze_result, indent=4)


def getVideoInfo(APIKey, videoId):
    YC = youtubeCrawler(APIKey)
    video_info = YC.get_video_info(videoId)

    return json.dumps(video_info, indent=4)


def getChannelInfo(APIKey, channelId):
    YC = youtubeCrawler(APIKey)
    channel_info = YC.get_channel_info(channelId)

    return json.dumps(channel_info, indent=4)


def getChannelImage(APIKey, channelId):
    YC = youtubeCrawler(APIKey)
    channel_info = YC.get_channel_info(channelId)

    return channel_info['thumbnails']

# output = analyzeAll(comedy_central)
# with open('comedycentral_test_upgrade.json', 'wt', encoding='UTF-8') as f:
#     f.write(output)


# output = analyzeNpredict(cbc_kid)
# with open('api_result_test.json', 'wt', encoding='UTF-8') as f:
#     f.write(output)
# print(output)

# with open('APIKey.json', 'rt', encoding='UTF-8') as f:
#     key = json.load(f)
#     apikey = key['APIkey']

# output = analyzeComplete(cbc_kid, apikey)
# with open('rebuild_test.json', 'wt', encoding='UTF-8') as f:
#     f.write(output)
# print(output)

# with open('APIKey.json', 'rt', encoding='UTF-8') as f:
#     key = json.load(f)
#     apikey = key['APIkey']
# output = getVideoInfo(apikey, cbc_kid)
# print(output)
