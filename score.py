import os
import logging
import json
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO)


def init():
    global model, vectorizer

    try:
        # AZUREML_MODEL_DIR is where model files are stored
        model_dir = os.getenv("AZUREML_MODEL_DIR")
        logging.info(f"Model directory: {model_dir}")

        if model_dir and os.path.exists(model_dir):
            logging.info(f"Files in model dir: {os.listdir(model_dir)}")

        model_path = os.path.join(model_dir, "bank_model.pkl")

        if not os.path.exists(model_path):
            model_path = "bank_model.pkl"
            logging.info("Trying current directory for model file")

        logging.info(f"Loading model from: {model_path}")

        with open(model_path, "rb") as f:
            vectorizer, model = joblib.load(f)

        logging.info("Model loaded successfully")
        logging.info(f"Vectorizer type: {type(vectorizer)}")
        logging.info(f"Model type: {type(model)}")

    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")


def run(raw_data):
    try:
        logging.info("Received prediction request")
        data = json.loads(raw_data)

        text = data.get("text", "")
        if not text:
            if isinstance(data, str):
                text = data
            else:
                return {"error": "No text provided in 'text' field"}

        logging.info(f"Processing text: {text}")

        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        probabilities = []
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X).tolist()[0]

        result = {
            "prediction": str(prediction),
            "probabilities": probabilities
        }

        logging.info(f"Prediction: {result}")
        return result

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logging.error(f"{error_msg}")
        return {"error": error_msg}
