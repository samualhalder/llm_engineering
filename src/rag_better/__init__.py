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

MODEL = "gpt-4.1-nano"
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


# chunks=create_chunks(documents=documents) # need when need to create chunks

# print(chunks)



def create_embedding(chunks):
    chroma=PersistentClient(path=DB_NAME)
    if collection_name in [ c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)
    texts=[chunk.page_content for chunk in chunks]
    emb=client.embeddings.create(model=embedding_model,input=texts).data

    collection=chroma.get_or_create_collection(collection_name)
    vectors=[e.embedding for e in emb]

    ids=[str(id) for id in range(len(chunks))]
    metas=[chunk.metadata for chunk in chunks]
    collection.add(ids=ids,embeddings=vectors,documents=texts,metadatas=metas)

    print(f"vector store cereated with {collection.count()} documents")

# create_embedding(chunks) # need when to create embedding 



class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]

RETRIVAL_K=10
chroma=PersistentClient(path=DB_NAME)
collection=chroma.get_or_create_collection(collection_name)

def fetch_context_unrancked(question):
    emb=client.embeddings.create(model=embedding_model,input=[question]).data[0].embedding
    results=collection.query(query_embeddings=[emb],n_results=RETRIVAL_K)
    chunks=[]
    for result in zip(results["documents"][0],results["metadatas"][0]):
        chunks.append(Result(page_content=result[0],metadata=result[1]))
    return chunks

question="who won IIOTY award?"

unranked_chunks=fetch_context_unrancked(question=question)

reranked=rerank(question,unranked_chunks)


def fetch_context(question):
    unraked=fetch_context_unrancked(question=question)
    return rerank(question=question,chunks=unraked)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    query = rewrite_query(question, history)
    print(query)
    chunks = fetch_context(query)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks

print(answer_question("Who won the IIOTY award?", [])[0])