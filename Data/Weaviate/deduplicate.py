from weaviate.classes.config import Property, DataType
from weaviate.connect import ConnectionParams
import weaviate
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
collection = client.collections.use("Documents")
seen = set()
to_delete = []

for obj in collection.iterator(return_properties="doc_id"):
    prop_value = obj.properties["doc_id"]
    if prop_value in seen:
        to_delete.append(obj.uuid)
    else:
        seen.add(prop_value)
for uuid in to_delete:
    collection.data.delete_by_id(uuid) 
client.close()