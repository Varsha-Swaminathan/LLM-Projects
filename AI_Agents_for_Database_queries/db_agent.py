
import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent, create_csv_agent
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

llm = ChatGroq(model='llama-3.1-70b-versatile')
df = pd.read_csv('AI_Agents_for_Database_queries\salaries_2023.csv')

agent = create_pandas_dataframe_agent(
    llm=llm, df=df, verbose=True, allow_dangerous_code=True)

CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns, get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method. Then reflect on the answers of the two methods you
did and ask yourself if it answers the original question correctly.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result, reflect and try again until
you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that you are not sure of the answer
- If you are sure of the correct answer, create a beautiful and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got to the answer on
a section that starts with: "\n\n Explanation: \n"
In the explanation, mention the column names that you used to get to the final answer.
"""


st.title("Database AI Agent with LangChain")

st.write("### Dataset Preview")
st.write(df.head())

st.write("### Ask a Question")
question = st.text_input(
    "Enter your question about the dataset: ",
    "Which grade has the highest average base salary, and compare the average female pay vs male pay?"
)

if st.button("Run Query"):
    QUERY = CSV_PROMPT_PREFIX+question+CSV_PROMPT_SUFFIX
    res = agent.invoke(QUERY)
    st.write("### Final answer")
    st.markdown(res['output'])
