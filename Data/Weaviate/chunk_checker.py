import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.query import Sort   # <-- import the Sort class

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

collection = client.collections.get("Place")

# Fetch objects, ordered by chunk_order (ascending)
results = collection.query.fetch_objects(
    limit=1000,
)

for obj in results.objects:
    print(f"name_de: {obj.properties['name_de']}")
    print(f"name_fr: {obj.properties['name_fr']}")
    print(f"type: {obj.properties['type']}")
    print(f"url: {obj.properties['url']}")

client.close()
