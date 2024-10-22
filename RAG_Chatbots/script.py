from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model='llama-3.1-70b-versatile')
parser = StrOutputParser()

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")])
chain = prompt_template | llm | parser


app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using LangChain's runnable interfaces",
)

add_routes(app, chain, path="/chain",)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
