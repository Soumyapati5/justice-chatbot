from PyPDF2.errors import PdfReadError  # Import the PdfReadError for handling exceptions
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import faiss

# Specify your directory path
dir_path = "/content/drive/MyDrive/JusticePdf"
file_url_lst = []

# Populate file_url_lst with only PDF files
for file_name in os.listdir(dir_path):
    if file_name.endswith(".pdf"):  # Ensure only PDF files are selected
        file_url = os.path.join(dir_path, file_name)
        file_url_lst.append(file_url)

all_documents = []

# Process each file
for file in file_url_lst:
    try:
        loader = PyPDFLoader(file)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Successfully loaded {file}")
    except PdfReadError:
        print(f"Error loading {file}: Not a valid PDF")
    except Exception as e:
        print(f"An unexpected error occurred while loading {file}: {str(e)}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
docs = text_splitter.split_documents(all_documents)

# Use SentenceTransformer embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS index from documents
db = FAISS.from_documents(docs, embedding_function)

# Save FAISS index and associated data as before
index_dir = "myenv/embdedDoc"
os.makedirs(index_dir, exist_ok=True)

# Save FAISS index
index_path = os.path.join(index_dir, 'faiss_index.bin')
faiss.write_index(db.index, index_path)

# Save document store
docstore_path = os.path.join(index_dir, 'docstore.pkl')
with open(docstore_path, 'wb') as f:
    pickle.dump(db.docstore, f)

# Save index to docstore ID mapping
index_to_docstore_id_path = os.path.join(index_dir, 'index_to_docstore_id.pkl')
with open(index_to_docstore_id_path, 'wb') as f:
    pickle.dump(db.index_to_docstore_id, f)

# Print saved paths
print(f"FAISS index saved to {index_path}")
print(f"Document store saved to {docstore_path}")
print(f"Index to docstore ID mapping saved to {index_to_docstore_id_path}")

# # Download the saved index and metadata files
# from google.colab import files

# files.download(index_path)
# files.download(docstore_path)
# files.download(index_to_docstore_id_path)
