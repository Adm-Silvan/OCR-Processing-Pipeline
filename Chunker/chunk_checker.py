import weaviate
from weaviate.connect import ConnectionParams

client = weaviate.WeaviateClient(
    connection_params=ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )
)
client.connect()

collection = client.collections.get("LateChunk")
results = collection.query.fetch_objects(limit=100)  # Adjust limit as needed

for obj in results.objects:
    print(f"Chunk ID: {obj.properties['chunk_id']}")
    print(f"Chunk Order: {obj.properties['chunk_order']}")
    print(f"Doc ID: {obj.properties['doc_id']}")
    print(f"Content: {obj.properties['content']}\n{'-'*40}")
client.close()