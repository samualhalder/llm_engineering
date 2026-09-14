import os
import glob
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE

load_dotenv()

db_name='vector_db'
vector_store=None
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(db_name)==False:
    folders=glob.glob('./././data/knowledge-base/*')


    documents=[]
    for folder in folders:
        doc_type=os.path.basename(folder) #return folder name 

        loader=DirectoryLoader(
            folder,
            glob='**/*.md',
            loader_cls=TextLoader,
            loader_kwargs={"encoding":"utf-8"}
        )  #searches all the matchign md files read them as text use utf encoding
        folder_docs=loader.load() #load search results 
        for doc in folder_docs:
            doc.metadata["doc_type"]=doc_type
            documents.append(doc)


    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    chunks=text_splitter.split_documents(documents=documents)


    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name
    )
else:
    vector_store=Chroma(
            persist_directory=db_name,
            embedding_function=embeddings
        )
    


