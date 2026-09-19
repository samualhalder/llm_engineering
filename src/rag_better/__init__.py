import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


load_dotenv(override=True)

MODEL = "gemini/gemini-3.5-flash-lite"
DB_NAME="preprocessed_db"
collection_name="docs"
embedding_model = "text-embedding-3-large"
knowledgebase_path=Path("/home/samualhalder/My_Work/agentic_ai/llm_engineering/data/knowledge-base")
AVERAGE_CHUNK_SIZE=500


client=OpenAI()

class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")

    def as_result(self,document)->Result:
        metadata={"source":document["source"],"type":document["type"]}
        return Result(
            page_content=self.headline+ "\n\n" + self.summary + "\n\n" +self.original_text,
            metadata=metadata
        )

class Chunks(BaseModel):
    chunks: list[Chunk]

def fetchDocument():
    documents=[]

    for folder in knowledgebase_path.iterdir():
        doc_type=folder.name
        for file in folder.rglob("*.md"):
            with open(file,"r",encoding="utf-8") as f:
                documents.append({"type":doc_type,"source":file.as_posix(),"text":f.read()})

    print(f"loaded {len(documents)}") 

    return documents


documents=fetchDocument()
def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""

def make_message(document):
    return [
        {"role":"user","content":make_prompt(document=document)}
        # {"role":"user","content":"HI, its samual here"}

    ]


def process_document(document):
    messages=make_message(document=document)
    response=completion(model=MODEL,messages=messages,response_format=Chunks)
    reply=response.choices[0].message.content
    doc_as_chunks=Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    chunks=[]
    for doc in tqdm(documents):
        chunks.extend(process_document(document=doc))
    return chunks


chunks=create_chunks(documents=documents[:1])

print(chunks)

def create_embedding(chunks):
    print("hit here0")

    chroma=PersistentClient(path=DB_NAME)
    if collection_name in [ c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)
    print("hit here1")
    texts=[chunk.page_content for chunk in chunks]
    print("hit here2")

    emb=client.embeddings.create(model=embedding_model,input=texts).data
    print("hit here3")

    vectors=[e.embedding for e in emb]
    print("hit here4")

    collection=chroma.get_or_create_collection(collection_name)
    ids=[str(id) for id in range(len(chunks))]
    metas=[chunk.metadata for chunk in chunks]
    collection.add(ids=ids,embeddings=vectors,documents=texts,metadatas=metas)

    print(f"vector store cereated with {collection.count()} documents")

create_embedding(chunks)