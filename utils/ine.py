from utils.rect import Rect

class INE:
    def __init__(self, boundingBox:Rect, side):
        self.boundingBox = boundingBox
        self.side = side

    def getINEArea(self):
        width = self.boundingBox.right - self.boundingBox.left
        height = self.boundingBox.bottom - self.boundingBox.top

        return width * height

    def getBoundingBox(self):
        return self.boundingBox

    def getINESide(self):
        return self.side
