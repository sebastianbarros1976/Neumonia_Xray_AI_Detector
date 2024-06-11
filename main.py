import gradio as gr
import cv2
import numpy as np
import joblib
import yaml
import os

# Load configuration from config.yaml
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set parameters from the configuration
IMAGE_HEIGHT = config['model']['input_shape'][0]
IMAGE_WIDTH = config['model']['input_shape'][1]
DIMENSIONS = (IMAGE_WIDTH, IMAGE_HEIGHT)
CLASS_NAMES = {0: 'Normal', 1: 'Viral Pneumonia', 2: 'Covid'}
MODEL_PATH = config['deployment']['checkpoint_path']

# Load the saved model
model = joblib.load(MODEL_PATH)

# Define the prediction function
def predict_covid(image):
    image = cv2.resize(image, DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    image = image / 255.0
    image = image.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    prediction = model.predict(image).flatten()
    return {CLASS_NAMES[i]: float(prediction[i]) for i in range(3)}

# Create the Gradio interface
image_input = gr.Image()
label_output = gr.Label(num_top_classes=3)

demo = gr.Interface(
    fn=predict_covid,
    inputs=image_input,
    outputs=label_output,
    title=config['deployment']['title'],
    description=config['deployment']['description'],
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(inline=False, share=config['deployment']['gradio_share'])
pip install -r requirements.txt
python main.py
