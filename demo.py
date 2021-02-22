import pickle
from worker import FeatureExtractor
from worker import custom_prediction
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)
with open('feature_extractor.pkl', 'rb') as input:
    feature_extractor = pickle.load(input)
        
with open('input_caption.txt', 'r') as file:
    query = file.read().replace('\n', '')   

image_path = 'input_image.jpg'

#image_path = 'demo/1.jpg'
features, infos = feature_extractor.extract_features(image_path)

img = PIL.Image.open(image_path).convert('RGB')
img = torch.tensor(np.array(img))

plt.axis('off')
plt.imshow(img)
plt.show()
    
#query = "blue elephants"
custom_prediction(query, features, infos, tokenizer, model)
    


