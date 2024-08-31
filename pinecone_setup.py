import pinecone
from pinecone import Pinecone, ServerlessSpec
import json
from transformers import CLIPProcessor, CLIPModel
import torch
from dotenv import load_dotenv
import os
from PIL import Image
import requests
from io import BytesIO

load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=pinecone_api_key)

index_name = "clothing-embeddings"

# Create or connect to an existing Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,  # Adjust dimension to your model's output
        metric='cosine',  # You can change the metric as needed (e.g., euclidean, dotproduct)
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Load your pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load dataset
with open('MODEL/dataset.json', 'r') as f:
    dataset = json.load(f)

# Function to generate embeddings using CLIP model
def generate_image_embedding(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.numpy().tolist()

# Upload embeddings to Pinecone with metadata
for item in dataset:
    image_embedding = generate_image_embedding(item['image_url'])
    metadata = {
        'id': item['id'],
        'image_url': item['image_url'],
        'purchase_url': item['purchase_url']
    }
    index.upsert([(item['id'], image_embedding[0], metadata)])
    print(f"Uploaded item {item['id']} to Pinecone.")
