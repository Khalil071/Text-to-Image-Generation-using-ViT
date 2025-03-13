Text-to-Image Generation using ViT

Overview

This project develops a model that generates images from text using Vision Transformer (ViT). Specifically, it leverages the pre-trained model google/vit-base-patch16-224 from Hugging Face to classify and generate image representations based on textual descriptions.

Features

Utilize ViT (google/vit-base-patch16-224) for image classification

Convert text descriptions into image representations

Fine-tune the model for better performance

Save and visualize generated images

Technologies Used

Python

Hugging Face Transformers

PyTorch

OpenCV/PIL for image processing

Matplotlib for visualization

Installation

Clone the repository:

git clone https://github.com/yourusername/text-to-image-vit.git
cd text-to-image-vit

Install dependencies:

pip install -r requirements.txt

Install Hugging Face Transformers and PyTorch:

pip install transformers torch torchvision torchaudio

Model Usage

Load the pre-trained ViT model:

from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

Process an input image:

image = Image.open("path/to/image.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
print("Predicted class:", predicted_class)

Future Improvements

Enhance text-to-image transformation by integrating additional generative models

Fine-tune ViT for better performance on custom datasets

Develop an interactive UI for text-to-image generation

Deploy as an API for broader accessibility

License

This project is licensed under the MIT License.
