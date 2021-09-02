from bottle import route, run, request, response
import src.evaluation as evaluation
import src.model as model
import src.utils as utils
import logging
import numpy as np
import cv2
logging.basicConfig(level=logging.INFO)

model = evaluation.load_trained_model(checkpoint="nutrinet.pytorch",
                                 checkpoint_dir="./", device="cpu")


@route('/nutri-score', method='POST')
def index():
    image = request.files.get("image")

    if image is None:
        response.status = 400
        return {"error": "Please supply 'image' parameter"}
    try:
        raw = image.file.read()
        nparr = np.frombuffer(raw, np.uint8)
        im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        number, name, probs = evaluation.predict_nutrition_score(model, im)
        response.status = 200
        return {"result": "OK", "name": name, "probs": probs.tolist(), "label_number": number, "class_to_label": utils.CLASS_TO_LABEL}
    
    except Exception as ex:
        logging.error(ex)
        response.status = 400
        return {"result": "ERROR", "error": str(ex)}

    response.status = 500
    return {"result": "UNKNOWN"}

@route('/health', method='GET')
def healthCheck():
    response.status = 200
    return {"result": "OK"}


run(servery="cherrypy", host='0.0.0.0', port=8080)
