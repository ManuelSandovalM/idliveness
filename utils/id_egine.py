from utils.inedetector import INEDetector
from utils.model_inference import ModelInference
from PIL import Image
from utils.rect import Rect

PATH_LIV_MODEL  = "./libs/INE_LIV_1epoch-v1.tflite"
PATH_SIDE_MODEL = "./libs/ANV_REV_1epochv1.tflite"
PATH_TYPE_MODEL = "./libs/INE_LIV_1epoch-v1.tflite"
PATH_ORI_MODEL  = "./libs/orientationINE_3epoch-v1.tflite"

ine_detector = INEDetector()

liv_inference  = ModelInference(PATH_LIV_MODEL)
side_inference = ModelInference(PATH_SIDE_MODEL)
type_inference = ModelInference(PATH_TYPE_MODEL)
ori_inference  = ModelInference(PATH_ORI_MODEL)

def get_id_side(input_image:Image)-> str:

    side_res = side_inference.runClassModel(input_image)

    status = side_res[0]

    if status != 0:
        return "Error: AnvRev Model status not zero"
    
    scores = side_res[1]

    if len(scores) != 2:
        return "Error: AnvRev Model scores input wrong"
    
    anverso = scores[0]
    reverso = scores[1]

    if anverso > reverso:
        return "ANVERSO"
    
    return "REVERSO"


def get_id_live(input_image:Image)-> float:

    live_res = liv_inference.runClassModel(input_image)

    status = live_res[0]

    if status != 0:
        return "Error: Liveness model status not zero"
    
    scores = live_res[1]

    if len(scores) != 2:
        return "Error: Liveness model scores input wrong"
    
    # TODO: Checar con Rafa si esto está bien, poner umbral o regresar valor?

    real = scores[0]
    fake = scores[1]

    return real


def get_id_type(input_image:Image)-> str:

    type_res = type_inference.runClassModel(input_image)

    status = type_res[0]

    if status != 0: 
        return "Error: Type model status not zero"
    
    scores = type_res[1]

    if len(scores) != 2:
        return "Error: Wrong scores input type model"
    
    gh_score = scores[0]
    ef_score = scores[1]

    if ef_score > gh_score: 
        return "EF"
    
    return "GH"


def get_id_ori(input_image:Image) -> str:

    ori_res = ori_inference.runClassModel(input_image)

    status = ori_res[0]

    if status != 0: 
        return "Error: ori model status not zero"
    
    scores = ori_res[1]

    if len(scores) != 4:
        return "Error: Wrong scores input model"
    
    index_max = max(range(len(scores)), key=scores.__getitem__)

    if index_max == 0: 
        return "DOWN"
    elif index_max == 1: 
        return "LEFT"
    elif index_max == 2:
        return "RIGHT"
    elif index_max == 3:
        return "UP"
    
    return "UNKOWN"


def get_id_detection(input_image:Image)-> Rect:

    id_detections = ine_detector.detect(input_image)

    if len(id_detections) == 1:
        bounding_box = id_detections[0].getBoundingBox()
    elif len(id_detections) > 1:

        areas = []

        for detection in id_detections:
            areas.append(detection.getINEArea())

        index_max = max(range(len(areas)), key=areas.__getitem__)

        bounding_box = id_detections[index_max].getBoundingBox()
    else: 

        bounding_box = None

    return bounding_box



def get_id_data(input_image:Image)-> dict:
    
    id_rect = get_id_detection(input_image)
    id_side = get_id_side(input_image)
    id_type = get_id_type(input_image)
    id_live = get_id_live(input_image)
    id_ori  = get_id_ori(input_image)

    return {
        "side": id_side,
        "type": id_type,
        "ori": id_ori,
        "liveness": id_live,
        "coord": {
            "top": int(id_rect.top),
            "right": int(id_rect.right),
            "bottom": int(id_rect.bottom),
            "left": int(id_rect.left)
        }
    }