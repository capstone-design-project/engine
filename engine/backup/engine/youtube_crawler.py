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
