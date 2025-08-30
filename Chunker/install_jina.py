from transformers import AutoTokenizer, AutoModel
model_name = "jinaai/jina-embeddings-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
