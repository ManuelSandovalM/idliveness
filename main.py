from fastapi import FastAPI
from pydantic import BaseModel
from utils.id_egine import *

from PIL import Image
import base64
from io import BytesIO

app = FastAPI()

class RequestImage(BaseModel):
    idImgBase64:str


@app.get("/")
def index():
    return {
        "status": 0,
        "description": "INE Data-Liveness Service running",
        "endpoints": ["id-liveness"]
    }

@app.post("/id-liveness")
def id_liveness(data:RequestImage):

    try:
        input_image = Image.open(BytesIO(base64.b64decode(data.idImgBase64))).convert('RGB')
    except Exception as e:
        return {
            'status': 1,
            'description': f"Error on input image, {e}",
            'details': {}
        }

    response_details = get_id_data(input_image)

    return {
        "status": 0,
        "description": "OK",
        "details": response_details
    }