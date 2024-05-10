class Rect:
    def __init__(self, left:float, top:float, right:float, bottom:float):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def width(self) -> float:
        return self.right - self.left

    def height(self) -> float:
        return self.bottom - self.top
    
    def getCenterX(self):
        return self.left + (self.width()/2)
    
    def getCenterY(self):
        return self.top + (self.height()/2)
    
    def getArea(self):
        return self.width()*self.height()
    
    def asArray(self):
        return [self.left, self.top, self.right, self.bottom]