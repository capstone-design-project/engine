from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework import status
from django.views import View
from django.http import HttpResponse, JsonResponse
from .engine import core

import json


class Video(View):

    def post(self, request):

        s = request.body.decode('utf8').replace("'", '"')
        j = json.loads(s)
        videoId = j['videoId']

        print('videoId: ' + videoId)
        output = core.analyzeComplete(videoId,'AIzaSyDRSRROudV5mjr1VEREqw6FKwbCTXpeqLw')
        with open('./comedycentral_test.json', 'wt', encoding='UTF-8') as f:
            f.write(output)

        return JsonResponse({'status': 'ok', 'output' : output})