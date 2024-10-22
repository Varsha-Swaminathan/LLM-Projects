from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
# from pathlib import Path

# dotenv_path = Path('path/to/.env')
# load_dotenv(dotenv_path=dotenv_path)
load_dotenv()

chat = ChatGroq(model='llama-3.1-70b-versatile')
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)


prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
