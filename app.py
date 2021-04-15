from haystack import Finder
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.sql import SQLDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.sparse import TfidfRetriever
from haystack.utils import print_answers
from haystack.pipeline import ExtractiveQAPipeline
from fastapi import FastAPI, Request, UploadFile, File 
from pydantic import BaseModel
import uvicorn
import time
import logging

# In-Memory Document Store
# document_store = InMemoryDocumentStore()

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.write_documents(dicts)
retriever = TfidfRetriever(document_store=document_store)
reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)
pipe = ExtractiveQAPipeline(reader, retriever)

def inference(query):
    prediction = pipe.run(query=query, top_k_retriever=10, top_k_reader=5)
    return prediction

app = FastAPI(title="Haystack APIs", version="1.0.0")
logging.basicConfig(formate='%(asctime)s - %(message)s', datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class QueryRequest(BaseModel):
    text: str 

def request_time(start_time, end_time):
    total_time = start_time - end_time
    minutes = int(total_time)//60
    secs = total_time - minutes
    logger.info(f"Request Completed took {minutes}m {secs:.2f}s", )

@app.post("/api/query")
def post_query(request: Request, query_text: QueryRequest):
    start_time = time.time()
    results = inference(query_text.text)
    end_time = time.time()
    request_time(start_time, end_time)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8001)