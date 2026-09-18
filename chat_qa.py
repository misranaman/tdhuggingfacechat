import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI Assistant"),
        ("human", "Question: {question}")
    ]
)


def generate_query(query, llm, temperature, max_token, api_key):
    model = ChatOpenAI(model=llm, temperature=temperature, api_key=api_key, max_tokens=max_token)
    parser = StrOutputParser()
    chain = prompt | model | parser
    result = chain.invoke({"question": query})
    return result


st.title("Q & A Chatbot with OpenAI")

query = st.text_input("What's your query ?")

st.sidebar.title("Settings ⚙️")
api_key = st.sidebar.text_input("Provide API Key", type="password")
llm = st.sidebar.selectbox("Select the model LLM:",
                           ["gpt-5", "gpt-4o", "gpt-4o-turbo", "gpt-3.5-mini", "o3"])

temperature = st.slider("Select Temperature:", min_value=0.0, max_value=2.0, value=0.7)

max_token = st.slider("Max Token", min_value=50, max_value=200, value=100)

if st.button("Answer"):
    if query and api_key:
        response = generate_query(query=query, temperature=temperature, max_token=max_token, api_key=api_key, llm=llm)
        st.write(response)
    else:
        st.warning("Missing API key or Query")
