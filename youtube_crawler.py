from googleapiclient.discovery import build


class youtubeCrawler:
    def __init__(self, ApiKey):
        self.youtube = build('youtube', 'v3', developerKey=ApiKey)

    def get_recent_videos(self, channel_id):
        newpageToken = ''
        video_list = []
        stopsign = False
        while not stopsign:
            activites = self.youtube.activities().list(part='contentDetails', channelId=channel_id,
                                                       fields='pageInfo, nextPageToken, items(contentDetails)', maxResults=50, pageToken=newpageToken).execute()
            try:
                newpageToken = activites['nextPageToken']
            except KeyError:
                print('no more videos')
                stopsign = True

            for item in activites['items']:
                if 'upload' in item['contentDetails']:
                    video_list.append(
                        item['contentDetails']['upload']['videoId'])
                if len(video_list) == 100:
                    stopsign = True
                    print(f'{channel_id} vidoes OK')
                    break

        return video_list

    def get_video_info(self, video_id):
        video_category_id = {
            1: 'Film & Animation',
            2: 'Autos & Vehicles',
            10: 'Music',
            15: 'Pets & Animals',
            17: 'Sports',
            18: 'Short Movies',
            19: 'Travel & Events',
            20: 'Gaming',
            21: 'Videoblogging',
            22: 'People & Blogs',
            23: 'Comedy',
            24: 'Entertainment',
            25: 'News & Politics',
            26: 'Howto & Style',
            27: 'Education',
            28: 'Science & Technology',
            29: 'Nonprofits & Activism',
            30: 'Movies',
            31: 'Anime/Animation',
            32: 'Action/Adventure',
            33: 'Classics',
            34: 'Comedy',
            35: 'Documentary',
            36: 'Drama',
            37: 'Family',
            38: 'Foreign',
            39: 'Horror',
            40: 'Sci-Fi/Fantasy',
            41: 'Thriller',
            42: 'Shorts',
            43: 'Shows',
            44: 'Trailers',
        }
        videos = self.youtube.videos().list(
            part='id, snippet,  recordingDetails, statistics,  topicDetails', id=video_id).execute()
        video_info = {}
        video_info['videoId'] = video_id
        video_info['publishedDate'] = videos['items'][0]['snippet']['publishedAt']
        video_info['channelId'] = videos['items'][0]['snippet']['channelId']
        video_info['channelTitle'] = videos['items'][0]['snippet']['channelTitle']
        video_info['title'] = videos['items'][0]['snippet']['title']
        video_info['description'] = videos['items'][0]['snippet']['description']
        try:
            video_info['thumbnails'] = videos['items'][0]['snippet']['thumbnails']['standard']['url']
        except KeyError:
            video_info['thumbnails'] = videos['items'][0]['snippet']['thumbnails']['medium']['url']
        video_info['category'] = video_category_id[int(videos['items']
                                                   [0]['snippet']['categoryId'])]
        try:
            video_info['topic'] = list(map(lambda x: x.replace(
                'https://en.wikipedia.org/wiki/', ''), videos['items'][0]['topicDetails']['topicCategories']))
        except KeyError:
            video_info['topic'] = []

        return video_info

    def get_channel_info(self, channel_id):

        channels = self.youtube.channels().list(
            part='id, snippet,  statistics,  topicDetails', id=channel_id).execute()
        channel_info = {}
        channel_info['channelId'] = channel_id
        channel_info['title'] = channels['items'][0]['snippet']['title']
        try:
            channel_info['description'] = channels['items'][0]['snippet']['description']
        except KeyError:
            channel_info['description'] = ""
        try:
            channel_info['thumbnails'] = channels['items'][0]['snippet']['thumbnails']['medium']['url']
        except KeyError:
            channel_info['thumbnails'] = channels['items'][0]['snippet']['thumbnails']['default']['url']
        try:
            channel_info['topic'] = list(map(lambda x: x.replace(
                'https://en.wikipedia.org/wiki/', ''), channels['items'][0]['topicDetails']['topicCategories']))
        except KeyError:
            channel_info['topic'] = []
        try:
            channel_info['subscriber'] = channels['items'][0]['statistics']['subscriberCount']
        except KeyError:
            channel_info['subscriber'] = -1
        try:
            channel_info['country'] = channels['items'][0]['snippet']['country']
        except KeyError:
            channel_info['country'] = "unknown"

        return channel_info
