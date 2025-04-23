# import libraries
import os
from dotenv import load_dotenv
import PyPDF2
import pinecone
import cohere
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

torch.classes.__path__ = [] # add this line to manually set it to empty.

# 0   Load environment variables
def load_env_variables(key):
    """Load .env file and return necessary API keys."""
    # Load environment variables from .env
    # Try loading .env explicitly
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)
    if key == "pinecone" :
        return os.getenv("pinecone_key")
    else :
        return os.getenv("cohere_key")

# 1      Initialize pinecone
import os
from pinecone import Pinecone, ServerlessSpec
pine_key = load_env_variables("pinecone")
pc = Pinecone(api_key=pine_key)

# Reset index
pc.delete_index('rag')

# Create index
if 'rag' not in pc.list_indexes().names():
          pc.create_index(
              name='rag',
              dimension=384,
              metric='cosine',
              spec=ServerlessSpec(
                  cloud='aws',
                  region='us-east-1'
              )
          )

# Vector Database Setup
index = pc.Index('rag')

# 2         Initialize Cohere for text generation (alternatively, GPT-3/4 API can be used)
coh_key = load_env_variables("cohere")
cohere_client = cohere.Client(api_key=coh_key)

# 3     Load a pre-trained embedding model from Hugging Face (e.g., sentence-transformers)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 4      Extract pdf
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text


# 5    Generate embeddings
def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)  # Average pooling

    # Convert to numpy array and cast to float32
    embeddings_np = embeddings.numpy().astype(np.float32)

    # L2 normalization (make sure the norm of the vector is 1)
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    normalized_embeddings = embeddings_np / norms  # Apply L2 normalization

    return normalized_embeddings

# 6   Create pinecone vector DB
def vector_db(i, segment, embedding) :
    index.upsert([(f"doc_{i}", embedding, {"text": segment})])  # Store the embedding in Pinecone
    
# 7    Retrieve relevant doc
def retrieve_relevant_docs(query, document_segments, top_k=3):
    query_embedding = generate_embeddings([query])[0].tolist()

    results = index.query(vector=[query_embedding], top_k=top_k)
    relevant_docs = []
    if 'matches' in results and results['matches']:
        for match in results['matches']:
            doc_id = match['id']
            doc_index = int(doc_id.split("_")[1])  # Assuming "doc_X" format
            relevant_docs.append(document_segments[doc_index])
    else:
        print("No matches found in the query results.")
    return relevant_docs   

# 8  Generate the answer using Cohere or any generative model
def generate_answer(query, relevant_docs):
    context = "\n".join(relevant_docs)
    prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    response = cohere_client.chat(
        model="command-nightly",
        message=prompt,
        max_tokens=700,
        temperature=0.5
    )
    return response.text.strip()   

# 9   QA Bot Function
def qa_bot(query, document_segments):
    # Retrieve relevant documents based on the query
    relevant_docs = retrieve_relevant_docs(query, document_segments)

    # Generate a coherent answer using Cohere
    answer = generate_answer(query, relevant_docs)
    return answer, relevant_docs              

