import streamlit as st
import os
import numpy as np
from PIL import Image
import io
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import fitz
from sklearn.metrics.pairwise import cosine_similarity
# load env
from dotenv import load_dotenv
load_dotenv()


@st.cache_resource
def load_pdf_and_prepare_data(pdf_path):
    # Load PDF and prepare text data
    pdf_loader = PyPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    text_docs = text_splitter.split_documents(pdf_loader.load())
    print(f"Number of text documents: {len(text_docs)}")
    # Prepare embeddings and vector store for text
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(embedding_function= embeddings)

    batch_size = 100  # You can adjust this value, but keep it under 166
    for i in range(0, len(text_docs), batch_size):
        batch = text_docs[i:i+batch_size]
        vector_store.add_documents(batch)
    
    # Extract and prepare image data
    extracted_images = extract_images_from_pdf(pdf_path)
    openclip_embeddings = OpenCLIPEmbeddings(model_name="ViT-B-16", checkpoint="openai")
    img_features = [openclip_embeddings.preprocess(image).unsqueeze(0) for image in extracted_images]
    img_vector_store = np.array([openclip_embeddings.model.encode_image(feature).detach().numpy().squeeze(0) for feature in img_features])
    
    return vector_store, extracted_images, openclip_embeddings, img_vector_store

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images

def find_top_matches(text_features_np, img_vector_store, top_k=3):
    similarity = cosine_similarity(text_features_np, img_vector_store)
    similarity = similarity.squeeze(0)
    top_indices = np.argsort(similarity)[-top_k:][::-1]
    top_scores = similarity[top_indices]
    top_images_scores = list(zip(top_indices, top_scores))
    return top_images_scores

def main():
    st.title("Multimodal Search Engine")
    
    pdf_path = "./2307.06435v9.pdf"  
    
    vector_store, extracted_images, openclip_embeddings, img_vector_store = load_pdf_and_prepare_data(pdf_path)
    
    # User input
    query = st.text_input("Enter your query:")
    
    if query:
        # Process query
        text_features = openclip_embeddings.embed_documents([query])
        text_features_np = np.array(text_features)
        
        # Find top matches for images and text
        top_3_images_indices_with_scores = find_top_matches(text_features_np, img_vector_store, top_k=3)
        top_3_texts = vector_store.similarity_search_with_relevance_scores(query, k=5)
        
        # Combine and sort results
        combined_results = top_3_images_indices_with_scores + top_3_texts
        combined_results.sort(key=lambda x: x[1], reverse=True)
        top_3_results = combined_results[:3]
        
        st.subheader("Top 3 Results:")
        
        vision_model = ChatOpenAI(temperature=0.5, model="gpt-4-vision-preview", max_tokens=1024)
        gpt4_model = ChatOpenAI(temperature=0.5, model="gpt-4o", max_tokens=1024)
        
        final_results = []
        
        for i, result in enumerate(top_3_results):
            st.write(f"Result {i+1}:")
            if isinstance(result[0], np.int64):
                image = extracted_images[result[0]]
                st.image(image, caption=f"Image {result[0]}", use_column_width=True)
                
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                out = vision_model.invoke([
                    SystemMessage(content=f'Extract any relevant information to this query: {query} from the image'),
                    HumanMessage(content=[{"type": "image", "image": img_str}])
                ])
                st.write("Image Analysis:", out.content)
                final_results.append(out.content)
            else:
                st.write("Text:", result[0].page_content)
                final_results.append(result[0].page_content)
        
        # Generate final response
        results_str = "\n".join(final_results)
        final_response = gpt4_model.invoke([
            SystemMessage(content=f'Use these contexts to respond to the user query: {results_str}'),
            HumanMessage(content=query)
        ])
        
        st.subheader("Final Response:")
        st.write(final_response.content)

if __name__ == "__main__":
    main()