from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv
load_dotenv()

# ConversationSummaryMemory has its own chain with it's own PromptTemplate(not chat prompt template) and Language model(but we need to define what model it is going to use)
# It takes the output from our language model along with the input we fed to it anf pushed it into it's prompt template-> with a prompt
# to summarize (user,AI) message it received. It stores that summaru until we ask out chain/model a follow up question

# When we ask a follow up question like 'and 3 more?', ConversationSummaryMemory will add in an additional input variable to our
# prompt with the same memory_key we provided-> "messages". The value for that memorykey is the text summary wrapped up as a
# System message, like SystemMessage("The user asked what the sum of 1+1 was and the AI replied the sum of 1+1 was 2")

chat = ChatGroq(model='llama-3.1-70b-versatile', verbose=True)
memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"), ConversationSummaryMemory does not work with FilrCHatMessageHistory
    memory_key="messages", return_messages=True,
    # It has its own prompt but we need to tell it what model to use.
    llm=chat)

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
    memory=memory,
    verbose=True  # debug step
)

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
