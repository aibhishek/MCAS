import sys
import torch
import clip
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage

class FeatureExtractor:

    def __init__(self, model_name, image_path, texts) -> None:
        self.model_name = model_name
        self.image_path = image_path
        self.texts = texts

    def load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, preprocess = clip.load(self.model_name)
        model.cuda().eval()
        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        return model, preprocess

    def load_images(self, image_path):
        original_images = []
        images = []

        model, preprocess = self.load_model()

        for name in glob.glob(image_path):
            if name.endswith(".jpg") or name.endswith(".png")or name.endswith(".jpeg"):
                image = Image.open(name).convert("RGB")
                original_images.append(image)
                images.append(preprocess(image))
        
        return images

    def get_image_features(self):
        image_input = torch.tensor(np.stack(self.load_images(self.image_path))).cuda()
        model, preprocess = self.load_model()
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def get_text_features(self):
        texts = self.texts
        model, preprocess = self.load_model()
        text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

        


