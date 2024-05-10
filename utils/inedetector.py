import os
from utils.model_inference import ModelInference
from utils.rect import Rect
from utils.ine import INE
from PIL import ImageDraw, Image


threshold = 0.5

class INEDetector:
    def __init__(self, modelPath = './libs/ineDetection.tflite') -> None:

        self.modelInference = ModelInference(modelPath, 320)

        self.labels = ["ine_back", "ine_front"]

    def detect(self, image):
        # Gray image
        img = image.convert('L')
        # RGB encoding Gray image just for TF
        img = img.convert('RGB')
        status, response = self.modelInference.runDetection(img, self.labels)
        
        # Filter detections by Threshold
        res = []
        if status==0:
            for ine in response:
                score = ine.getScore()
                if score>=threshold:
                    boundingBox = ine.getBoundingBox()
  
                    left = boundingBox.left
                    top = boundingBox.top * 0.75
                    right = boundingBox.right
                    bottom = boundingBox.bottom * 1.15
                    
                    # Append INE detections with no decoded data
                    res.append(INE(Rect(left, top, right, bottom), ine.getLabel()))
        return res
    
    def drawBoundingBox(self, image, boundingBox):
        # Coordenadas del bounding box (izquierda, superior, derecha, inferior)
        rectangle = (boundingBox.left, boundingBox.top, boundingBox.right, boundingBox.bottom)

        return self.drawRectangle(image, rectangle)


    # Coordenadas del rectangle (izquierda, superior, derecha, inferior)
    def drawRectangle(self, image, rectangle):
        # Crea un objeto de dibujo
        draw = ImageDraw.Draw(image)

        # Dibuja el bounding box en la imagen
        draw.rectangle(rectangle, outline='red', width=2)
        
        return image
    
    def cropBoundingBox(self, image:Image, boundingBox:Rect):
        rect = (boundingBox.left, boundingBox.top, boundingBox.right, boundingBox.bottom)
        return self.cropRectangle(image, rect)

    def cropRectangle(self, image:Image, rect:Rect):
        return image.crop(rect)