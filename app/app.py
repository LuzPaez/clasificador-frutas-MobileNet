from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pathlib
import sys

# Añadir la carpeta src al path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from inference import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = pathlib.Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo"})

    file = request.files["file"]
    filename = secure_filename(file.filename)

    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    clase, confianza = predict_image(file_path)

    return jsonify({
        "clase": clase,
        "confianza": round(confianza, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
