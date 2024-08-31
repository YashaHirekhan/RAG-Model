import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# Import retrieval functions
from pinecone_retrieval import generate_image_embedding, query_pinecone, format_context

# Configure Google Gemini API
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

def st_image_to_pil(st_image):
    image_data = st_image.read()
    pil_image = Image.open(io.BytesIO(image_data))
    return pil_image

def generate_gemini_response(prompt, image, context):
    combined_prompt = f"Use the context to answer the prompt : {prompt}, add links to two most similar products to the given image, where the user can buy similar products(Add only those links present in the context). Context : {context}" 
    # combined_prompt = f"{prompt} \nRelevant outfits:\n{context}"
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([combined_prompt, image])
    return response.text

# def resize_image(pil_image, scale=0.5):
#     width, height = pil_image.size
#     new_size = (int(width * scale), int(height * scale))
#     return pil_image.resize(new_size)

if __name__ == '__main__':
    st.title("Clothing Recommendation System")
    
    img = st.file_uploader('Select an Image: ', type=['jpg', 'jpeg', 'png', 'gif'])
    prompt = st.text_area('Enter Prompt: ')
    if img and prompt:
        pil_image = st_image_to_pil(img)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        # resized_image = resize_image(pil_image, scale=0.5)
        # Display resized image
        # st.image(resized_image, caption='Uploaded Image')

        with st.spinner('Processing...'):
            # Generate image embedding
            image_embedding = generate_image_embedding(pil_image)
            
            # Query Pinecone to retrieve relevant outfits
            pinecone_response = query_pinecone(image_embedding)
            
            # Format context from retrieved outfits
            retrieved_context = format_context(pinecone_response)
            
            # Generate response from Gemini using the prompt and retrieved context
            answer = generate_gemini_response(prompt, pil_image, retrieved_context)
            
            # Adjust height parameter as needed
            st.text_area('Gemini Answer: ', value=answer, height=300)  # Adjust height value as needed

        st.divider()
