import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
import faiss
import torch

device = torch.device("cpu")

data = [
        ['tiger.jpg', 'Animal'],
        ['donkey.jpg', 'Animal'],
        ['vehicle.jpg', 'Vehicle'],
        ['landscape.jpg', 'Landscape']
       ]

df = pd.DataFrame(data, columns=['image_path', 'category'])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    return image

# Load a pre-trained ResNet model for image embeddings
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

def get_image_embedding(image_tensor):
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy()

embeddings = np.vstack([get_image_embedding(load_image(path)) for path in df['image_path']])

vector_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
index.add(embeddings)

search_image_path = 'lion.jpg'
search_image = load_image(search_image_path)
search_vector = get_image_embedding(search_image)

search_vector = np.array([search_vector.squeeze()])

print("Shape of search_vector:", search_vector.shape)

k = index.ntotal
D, I = index.search(search_vector, k=k)  # Perform the search

print(D, I)

top_indices = I[0]
top_results = df.iloc[top_indices]

top_results['distance'] = D[0]

print(top_results)