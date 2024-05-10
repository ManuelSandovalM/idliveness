import tensorflow.lite as tflite
import numpy as np
from utils.detection import Detection
from utils.rect import Rect
from PIL import Image

class ModelInference:

    def __init__(self, modelPath, resizeTo = 224) -> None:
        # Carga el modelo TFLite
        self.interpreter = tflite.Interpreter(model_path=modelPath)
        self.interpreter.allocate_tensors()

        # Obtiene los detalles del modelo
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        #print(self.input_details)
        self.image = None
        self.resizeTo = resizeTo


    def runClassModel(self, image):
        self.image = image
        input_data = self.preProcessTF(self.image, self.resizeTo)

        try:
            self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(0, input_data)
            self.interpreter.invoke()
            output_index = 168
            scores = self.interpreter.get_tensor(output_index)[0]
            return 0, scores/255
        except Exception as e:
            return 1, f"Error: {e}"


    def runDetection(self, image, labels=None):
        #try:
        self.image = image
        input_data = self.preProcessTF(self.image, self.resizeTo)

        # Asigna los datos de entrada al intérprete
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Ejecuta la inferencia
        self.interpreter.invoke()

        output_index1 = 600 #scores
        output_index2 = 598 #bounding_boxes
        output_index3 = 601 #???
        output_index4 = 599 #Classes

        scores = self.interpreter.get_tensor(output_index1)[0]
        boxes = self.interpreter.get_tensor(output_index2)[0]
        #output_data3 = self.interpreter.get_tensor(output_index3)[0]
        classes = self.interpreter.get_tensor(output_index4)[0]
        
        
        confidence_threshold = 0.1
        
        valid_scores = [scores[i] for i in range(len(scores)) if scores[i] > confidence_threshold]
        valid_boxes = [boxes[i] for i in range(len(scores)) if scores[i] > confidence_threshold]
        valid_classes = [classes[i] for i in range(len(scores)) if scores[i] > confidence_threshold]

        #Escala los bounding Boxes
        for i in range(len(valid_boxes)):
            ymin, xmin, ymax, xmax = valid_boxes[i]
            width, height = self.image.size
            x_min = int(xmin * width)
            x_max = int(xmax * width)
            y_min = int(ymin * height)
            y_max = int(ymax * height)

            if x_min<0:
                x_min = 1
            
            if y_min<0:
                y_min = 1
            
            if x_max>width:
                x_max = width-2
            
            if y_max>height:
                y_max = height-2

            valid_boxes[i] = (x_min, y_min, x_max, y_max)

        #Normaliza los arreglos
        valid_scores = np.array(valid_scores)
        valid_boxes = np.array(valid_boxes)
        valid_classes = np.array(valid_classes)

        #Filtra los objetos con más de una detección duplicada
        selected_indices = self.non_max_suppression(valid_boxes, valid_scores, confidence_threshold)

        valid_scores = valid_scores[selected_indices]
        valid_boxes = valid_boxes[selected_indices]
        valid_classes = valid_classes[selected_indices]

        #Genera la lista de Detecciones final
        detections = []
        for i in range(len(valid_scores)):
            box = valid_boxes[i]
            rect_boxes = Rect(box[0], box[1], box[2], box[3])

            score = valid_scores[i]
            boundingBox = rect_boxes
            labelIndex = int(valid_classes[i])

            if labels:
                label = labels[labelIndex]
            else:
                label = None

            detections.append(Detection(boundingBox, score, labelIndex, label))
        
        return 0, detections
    

    def preProcessTF(self, image:Image, resize_size, float_mod:bool=False):
        image = image.resize((resize_size, resize_size))

        if not float_mod:
            np_image = np.array(image, dtype=np.uint8)
            return np.expand_dims(np_image, axis=0)

        np_image = np.array(image)

        if float_mod:
            norm_img_data = (np_image - np.min(np_image))/(np.max(np_image)-np.min(np_image))
            norm_img_data = np.array(norm_img_data, dtype=np.float32)
        else:
            norm_img_data = np.zeros(np_image.shape).astype('uint8')

        np_image = np.expand_dims(norm_img_data, axis=0)
        return np_image
    
    def non_max_suppression(self, boxes, scores, iou_threshold):

        if boxes.size == 0 or scores.size == 0:
            return np.array([])
        
        # Ordena las detecciones por puntajes de confianza en orden descendente
        sorted_indices = np.argsort(scores)[::-1]

        # Lista para almacenar los índices seleccionados después de NMS
        selected_indices = []

        if len(sorted_indices)==1:
            selected_indices.append(sorted_indices[0])
        else:
            while sorted_indices.size > 1:  # Verifica que hay al menos dos elementos para comparar
                # Obtiene el índice con el puntaje más alto
                best_index = sorted_indices[0]
                selected_indices.append(best_index)

                # Calcula la superposición (IoU) con todas las demás detecciones
                iou = np.linalg.norm(boxes[best_index] - boxes[sorted_indices[1:]], axis=1)

                # Filtra las detecciones que tienen una superposición significativa con la mejor detección
                filtrado = sorted_indices[1:][iou < iou_threshold]
                if not isinstance(filtrado, np.ndarray):
                    selected_indices.append(filtrado)

                # Actualiza los índices restantes para la siguiente iteración
                sorted_indices = sorted_indices[1:][iou >= iou_threshold]

        for i in reversed(range(0, len(selected_indices))):
            index = selected_indices[i]
            box = boxes[index]

            centerX = (box[0] + box[2]) // 2
            centerY = (box[1] + box[3]) // 2
            area = (box[2] - box[0]) * (box[3] - box[1])
                
            for ii in reversed(range(0, len(selected_indices))):
                if i==ii:
                    continue

                index2 = selected_indices[ii]
                box2 = boxes[index2]
                box_inside_box2 = self.is_center_inside_box(centerX, centerY, box2[0], box2[1], box2[2], box2[3]) or self.has_corner_inside_box(box[0], box[1], box[2], box[3], box2[0], box2[1], box2[2], box2[3])
                if box_inside_box2:
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    if area<area2:
                        selected_indices.pop(i)
                        break
        
        return selected_indices
    
    def has_corner_inside_box(self, box1_left, box1_top, box1_right, box1_bottom, box2_left, box2_top, box2_right, box2_bottom):
        left_top = (box2_left <= box1_left <= box2_right) and (box2_top <= box1_top <= box2_bottom)
        right_top = (box2_left <= box1_right <= box2_right) and (box2_top <= box1_top <= box2_bottom)
        left_bottom = (box2_left <= box1_left <= box2_right) and (box2_top <= box1_bottom <= box2_bottom)
        right_bottom = (box2_left <= box1_right <= box2_right) and (box2_top <= box1_bottom <= box2_bottom)

        return left_top or right_top or left_bottom or right_bottom
    
    
    def is_center_inside_box(self, center_x, center_y, box_left, box_top, box_right, box_bottom):
        return (box_left <= center_x <= box_right) and (box_top <= center_y <= box_bottom)
