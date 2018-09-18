import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


class CloudVision:
    client = vision.ImageAnnotatorClient()

    def getLabels(self, image):
        image = types.Image(content=image)
        response = self.client.label_detection(image=image)
        return response.label_annotations


# ------------

with io.open('igel.png', 'rb') as image_file:
    content = image_file.read()

vision = CloudVision()
labels = vision.getLabels(content)

for label in labels:
    print(label.description)
