import time
import streamlit as st
from langchain_core.callbacks import StdOutCallbackHandler
import glob
import numpy as np
from langchain_chroma import Chroma
from langchain.schema import Document
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
load_dotenv()

llm = ChatGroq(model='llama-3.1-70b-versatile')


db_name = "vector_db2"

# folders = r'C:\Users\varsh\Documents\AI\LLM\RAG_youtube_langchain\*'
folders = glob.glob("knowledge_base/*")
# print(folders)

documents = []
for folder in folders:
    # print('Folder is:', folder)
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(
        folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)
# print(documents)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
# print(chunks)
# print(len(chunks))

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs)
if not os.path.exists(db_name):
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=db_name)
else:
    print("Vector store already exists. No need to initialize.")

vectorstore = Chroma(persist_directory=db_name,
                     embedding_function=embeddings)


memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,
                                                           callbacks=[StdOutCallbackHandler()])

# query = "Who received the prestigious IIOTY award in 2023?"
# result = conversation_chain.invoke({"question": query})
# answer = result['answer']
# print('\nAnswer:', answer)


st.title("RAG and MultiAgent with LangChain")


def stream_data(text, delay: float = 0.02):
    for word in text.split():
        yield word+" "
        time.sleep(delay)


if "messages" not in st.session_state:
    st.session_state.messages = []

# st.write(st.session_state.messages)
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get('content'))

question = st.chat_input(
    "Ask something: "
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    # QUERY = CSV_PROMPT_PREFIX+question+CSV_PROMPT_SUFFIX
    # Display user question
    with st.chat_message("user"):
        st.write(question)

    result = conversation_chain.invoke(question)
    # st.write(result)
    response = result['answer']

    # store response
    st.session_state.messages.append(
        {"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        # st.markdown(response)
        st.write_stream(stream_data(response))
