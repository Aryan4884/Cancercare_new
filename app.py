from flask import Flask, request, render_template, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import google.generativeai as genai
import requests
import imghdr
from urllib.request import urlretrieve
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Load API Key from environment variables
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=API_KEY)

# Google Drive model file link (Replace with your actual file ID)
GOOGLE_DRIVE_FILE_ID = "https://drive.google.com/uc?export=download&id=1oexNe-WvK8QO5bBIozDl0uyuYANZIcX7"
MODEL_PATH = "my_model.h5"

def download_model_from_drive():
    """Download the model from Google Drive if it's not available locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
        urlretrieve(url, MODEL_PATH)
        print("Model downloaded successfully.")

# Ensure the model is available
download_model_from_drive()
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['Oral_Scc', 'Oral_Normal', 'Lymph_Mcl', 'Lymph_Fl', 'Lymph_Cll', 'Lung_Scc', 'Lung_Bnt', 'Lung_Aca',
                'Colon_Bnt', 'Colon_Aca', 'Kidney_Tumor', 'Kidney_Normal', 'Cervix_Sfi', 'Cervix_Pab', 'Cervix_Mep',
                'Cervix_Koc', 'Cervix_Dyk', 'Breast_Malignant', 'Breast_Benign', 'Brain_Tumor', 'Brain_Menin',
                'Brain_Glioma', 'ALL_Pro', 'ALL_Pre', 'ALL_Early', 'ALL_Benign']

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def preprocess_image(img_path):
    """Preprocess an image for model prediction."""
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/genome')
def genome_page():
    return render_template('genome.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and predict the class."""
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '' or imghdr.what(file) is None:
        return jsonify({"error": "Invalid image file"}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    os.remove(file_path)

    gemini_response = send_to_gemini(predicted_label)
    tcga_info = get_tcga_data(predicted_label)

    return render_template('result.html', label=predicted_label, gemini_response=gemini_response, tcga_info=tcga_info)

def send_to_gemini(predicted_label):
    """Get AI-generated response from Gemini API."""
    prompt = f"I have been diagnosed with {predicted_label} cancer. Can you tell me more about it?"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else "Sorry, no response from Gemini API."

def get_tcga_data(predicted_label):
    """Fetch TCGA data for a given cancer type."""
    tcga_api_url = f"https://api.gdc.cancer.gov/projects/{predicted_label}"
    response = requests.get(tcga_api_url)

    if response.status_code == 200:
        data = response.json()
        return {
            'project_id': data.get('project_id', 'N/A'),
            'name': data.get('name', 'N/A'),
            'primary_site': data.get('primary_site', 'N/A'),
            'disease_type': data.get('disease_type', 'N/A'),
            'program': data.get('program', {}).get('name', 'N/A')
        }
    return None

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests to Gemini AI."""
    user_input = request.form['user_input']
    gemini_response = send_to_gemini(user_input)
    return jsonify({'response': gemini_response})

def get_gene_info(gene_name):
    """Fetch gene information from Ensembl API."""
    url = f'https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_name}?content-type=application/json'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": f"Gene '{gene_name}' not found."}

def get_variant_info(rsid):
    """Fetch variant information from Ensembl API."""
    url = f'https://rest.ensembl.org/variation/human/{rsid}?content-type=application/json'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": f"Variant '{rsid}' not found."}

@app.route('/genomic', methods=['POST'])
def genomic():
    """Handle genomic information queries."""
    data = request.get_json()
    query_type = data.get('query_type')
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query input is missing"}), 400

    if query_type == 'gene':
        result = get_gene_info(query)
    elif query_type == 'variant':
        result = get_variant_info(query)
    else:
        result = {"error": "Invalid query type"}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
