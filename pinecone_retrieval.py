import os
import pinecone
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from io import BytesIO

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Initialize Pinecone
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')

# Create a Pinecone instance
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Connect to the index
index = pc.Index("clothing-embeddings")

# Load CLIP Model and Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
def generate_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.numpy().tolist()

def query_pinecone(image_embedding):
    query_response = index.query(
        vector=image_embedding[0],  # Assuming [0] selects the first vector
        top_k=3,  # Retrieve top 3 most similar outfits
        include_metadata=True
    )
    return query_response

def format_context(pinecone_response):
    context = "\n".join([
        f"Outfit: {match['metadata']['image_url']} (Buy: {match['metadata']['purchase_url']})"
        for match in pinecone_response['matches']
    ])
    return context
