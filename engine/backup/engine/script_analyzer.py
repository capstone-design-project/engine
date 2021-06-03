import syllables
import nltk
from youtube_transcript_api import YouTubeTranscriptApi
import json
import sys
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# print(os.getcwd())
# nltk.download('punkt')


# example_videoId = "TLnUJzueBOQ"


class ScriptAnalyzer:
    def getCaptions(self, videoId):
        # Get captions
        try:  # using the generated one
            captions = YouTubeTranscriptApi.list_transcripts(
                videoId).find_generated_transcript(['en']).fetch()
        except:  # if there is no English caption including auto-generated one.
            captions = [{'text': '', 'start': -1, 'duration': -1}]
        return captions

    def getScript(self, videoId):
        captions = self.getCaptions(videoId)
        return ' '.join([caption['text']
                         for caption in captions])

    def analyzeScript(self, videoId):
        captions = self.getCaptions(videoId)
        # Caption info
        info = dict()
        info['videoId'] = videoId
        info['script'] = ' '.join([caption['text']
                                   for caption in captions])  # whole script text
        info['avgSyllPerSec'] = syllables.estimate(
            info['script']) / sum([caption['duration'] for caption in captions])

        for caption in captions:  # Get tokens, words per second, syllables per second for each captions
            caption['token'] = nltk.word_tokenize(caption['text'])
            caption['wordPerSec'] = len(caption['token']) / caption['duration']
            caption['syllPerSec'] = syllables.estimate(
                caption['text']) / caption['duration']
            info[caption['start']] = caption
            del caption['start']

        return json.dumps(info, indent=4)
