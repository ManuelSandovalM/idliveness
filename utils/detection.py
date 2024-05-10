from utils.rect import Rect

class Detection:
    def __init__(self, boundingBox:Rect, score, labelIndex, label=None):
        self.boundingBox = boundingBox
        self.score = score
        self.labelIndex = labelIndex
        self.label = label

    def getBoundingBox(self)-> Rect:
        return self.boundingBox
    
    def getScore(self):
        return self.score
    
    def getLabelIndex(self):
        return self.labelIndex
    
    def getLabel(self):
        return self.label