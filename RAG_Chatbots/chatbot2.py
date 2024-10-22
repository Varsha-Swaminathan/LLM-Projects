from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv
load_dotenv()

chat = ChatGroq(model='llama-3.1-70b-versatile')
memory = ConversationBufferMemory(chat_memory=FileChatMessageHistory(
    "messages.json"), memory_key="messages", return_messages=True)


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
