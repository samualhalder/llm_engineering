from rag.implemantation.ingest import vector_store
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.messages import SystemMessage,HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

MODEL="gpt-4.1-mini"
db_name='vector_db'
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


retriver=vector_store.as_retriever()
llm=ChatOpenAI(model=MODEL,temperature=0)

SYSTEM_PROMPT="""
    You are an AI Agent for a company you have emplyee, contracts, company, prodcuts details.
    Apart from that you hae also the chat histroy of you and the user for you referance :
    {HISTORY}
    you should ans fron the data we have on a users quetion that is retvired from a RAG system 
    the data is:
    {DATA}
    You have both history and data context give answers accordingly .
"""

def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, str) or isinstance(part, dict)
        )
    return str(content)


def make_question(userinpt, history: list[dict] | None = None):
    history = history or []
    hist="\n\n".join(
        message_text(h["content"]) for h in history if h["role"] == "user"
    )
    return hist + "\n" + userinpt
    


def Chat(input,history):
    question=make_question(userinpt=input,history=history)
    docs=retriver.invoke(question)
    data="\n\n".join(doc.page_content for doc in docs)
    system_prompt=SYSTEM_PROMPT.format(DATA=data,HISTORY=history)
    res=llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content=input)])
    return res.content


