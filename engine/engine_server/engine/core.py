import json
from .word_analyzer import WordAnalyzer
from .script_analyzer import ScriptAnalyzer
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

    print('analyze ok')

    return json.dumps(analyze_result, indent=4)


# output = analyzeAll(comedy_central)

# with open('comedycentral_test.json', 'wt', encoding='UTF-8') as f:
#     f.write(output)
