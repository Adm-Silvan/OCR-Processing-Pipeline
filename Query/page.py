from flask import Flask, request, render_template_string
import weaviate
from weaviate.connect import ConnectionParams
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "jinaai/jina-embeddings-v4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

connection_params = ConnectionParams.from_params(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
)
client = weaviate.WeaviateClient(connection_params=connection_params)
client.connect()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
model.to(DEVICE)

def embed_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
        add_special_tokens=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    summed = torch.sum(outputs.last_hidden_state * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    embedding = (summed / counts).cpu().numpy()[0]
    return embedding.tolist()

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head><title>Semantic + Full-text Search</title></head>
  <body>
    <h2>Search Text Chunks</h2>
    <form method="post">
      <textarea name="query" rows="4" cols="60" placeholder="Enter your query here..."></textarea><br>
      <input type="submit" value="Search">
    </form>
    {% if results %}
      <h3>Top 5 Results</h3>
      <ol>
        {% for chunk in results %}
          <li><strong>Score:</strong> {{ '{:.3f}'.format(chunk['score']) }}<br>{{ chunk['content'] }}</li>
        {% endfor %}
      </ol>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        query_text = request.form.get("query", "").strip()
        if query_text:
            query_vector = embed_text(query_text)

            try:
                response = (
                    client.graphql.get()
                    .with_class_name("LateChunk")
                    .with_fields([
                        "content",
                        "chunk_order",
                        "_additional { certainty }"
                    ])
                    .with_near_vector({
                        "vector": query_vector,
                        "certainty": 0.7
                    })
                    .with_where({
                        "operator": "Like",
                        "path": ["content"],
                        "valueString": f"*{query_text}*"
                    })
                    .with_limit(5)
                    .do()
                )
                hits = response.get("data", {}).get("Get", {}).get("LateChunk", [])
                for hit in hits:
                    results.append({
                        "content": hit.get("content", ""),
                        "score": hit.get("_additional", {}).get("certainty", 0),
                        "chunk_order": hit.get("chunk_order", 0)
                    })
            except Exception as e:
                print("Query error:", e)

    return render_template_string(HTML_TEMPLATE, results=results)


if __name__ == "__main__":
    app.run(debug=True)
