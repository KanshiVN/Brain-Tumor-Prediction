import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
# Define the path for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Model Loading ---
# Load your trained model
try:
    model = load_model('brain_tumor_classification.keras')
except Exception as e:
    # If the model fails to load, we can't proceed.
    # In a real app, you'd log this error.
    print(f"Error loading model: {e}")
    model = None

# --- Constants ---
# Define image size and labels based on your training script
IMAGE_SIZE = 150
LABELS = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']

# --- Helper Function ---
def preprocess_image(image_path):
    """
    Preprocesses the user-uploaded image to match the model's input requirements.
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        # Resize it to the required 150x150 pixels
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        # Convert the image to a numpy array and expand dimensions to create a batch of 1
        img_array = np.expand_dims(img, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if model is None:
        # If the model didn't load, show an error message.
        return render_template('index.html', error="Model is not available. Please check server logs.")

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Create a secure filename and save the uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image for the model
            processed_image = preprocess_image(filepath)

            if processed_image is not None:
                # Make a prediction
                prediction = model.predict(processed_image)
                
                # Get the index of the highest probability
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                
                # Get the corresponding label and confidence score
                result = LABELS[predicted_class_index]
                confidence = f"{np.max(prediction) * 100:.2f}"

                # Render the page with the prediction results
                return render_template('index.html', prediction=result, confidence=confidence, image_file=filename)
            else:
                return render_template('index.html', error='Failed to process image.')

    # For a GET request, just render the initial page
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    """
    Serves the uploaded image file to be displayed on the results page.
    """
    return send_from_directory(UPLOAD_FOLDER, filename)

# This part is for local development, not used by Vercel
if __name__ == '__main__':
    app.run(debug=True)
