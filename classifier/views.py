import base64
import io
import os
import sys

from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView

from classifier import mnistclassifier


class ClassifierView(APIView):
    def get(self, request):
        hello = mnistclassifier.gethello()
        return Response({"response": hello})

    def post(self, request):
        # Convert base64 to image.
        raw_image = request.data['image'].replace('data:image/png;base64', '')
        base64img = base64.b64decode(raw_image)
        image = io.BytesIO(base64img)
        # Predict and return result.
        prediction = mnistclassifier.predict(image)
        return Response({"prediction": prediction})

class DefaultView(APIView):
    def get(self, request):
        with open(os.path.join(sys.path[0], 'classifier', "mainview.html"), "r") as f:
            html = f.readlines()
            f.close()
        return HttpResponse(html)