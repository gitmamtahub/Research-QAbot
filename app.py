# import libraries
import streamlit as st
from main import extract_text_from_pdf , generate_embeddings , vector_db , qa_bot

# Streamlit App
st.title("Interactive QA Bot with Document Upload")

#  Step 1 :  Upload Funtion
def upload(uploaded_file) :
    # Step 1.1 : Extract text from PDF
    document_text = extract_text_from_pdf(uploaded_file)
    st.write("Document uploaded successfully!")

    # Step 1.2: Generate embeddings and store in Pinecone
    document_segments = document_text.split(". ")  # Split the document into sentences
    return document_segments
    
    
# Step 2: Upload PDF
n_files = st.number_input('How many research papers to upload :')
    
document_seg= []

for i in range(int(n_files)) :
    key = f'pdf_upload_{i+1}'
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key=key)
    if uploaded_file :
        document_seg.extend(upload(uploaded_file))
    
if document_seg:
    # Insert documents into Pinecone with their embeddings
    for i, segment in enumerate(document_seg):
          embedding = generate_embeddings([segment])[0].tolist()
          vector_db(i, segment, embedding)
          
    st.write(f"Stored {len(document_seg)} document segments in Pinecone.")

# Step 3: Query Input
query = st.text_input("Ask a question based on the reserch papers:")
if query and document_seg:
    # Display the generated answer
    answer, relevant_docs = qa_bot(query, document_seg)
    st.write("Answer:")
    st.write(answer)
    st.write("Relevant Document Segments:")
    for doc in relevant_docs:
        st.write(f"- {doc}")