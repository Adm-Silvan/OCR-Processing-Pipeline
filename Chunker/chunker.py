from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

# Read your text file
with open("c:/Users/Silvan/Documents/Obsidian Fusion/Uni/CAS Data Engineering/OCR-Processing-Pipeline/Entity_Extractor/1973-12-18.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Set up Ollama embeddings (make sure Ollama is running locally)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")  # Use your preferred embedding model

# Initialize the semantic chunker
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # Adaptive thresholding
    # Optionally, tune the percentile, e.g., breakpoint_percentile_threshold=85
)

# Create semantic chunks
docs = chunker.create_documents([text])

# Print out each chunk
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print()
