from sentence_transformers import SentenceTransformer
import faiss
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
metadata = []

data_path = "data"

for file in os.listdir(data_path):
    with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
        text = f.read()

        chunks = text.split("\n\n")  # simple chunking

        for chunk in chunks:
            if chunk.strip():
                documents.append(chunk)
                metadata.append(file)

# Create embeddings
embeddings = model.encode(documents)

# Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "vector.index")

# Save text separately
import pickle
with open("docs.pkl", "wb") as f:
    pickle.dump((documents, metadata), f)

print("✅ Embeddings created!")