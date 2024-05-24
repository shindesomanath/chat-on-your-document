import re
import fitz  # PyMuPDF
import nltk

import streamlit as st
import functools, operator, requests, os, json
from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings

from langchain_community.embeddings import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.agents import Tool
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define a function to load and extract text from a PDF

openai.api_type = "azure"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = "91a2c0eb89e24d8faa9c8075b63be35a"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://cog-ovavoxv55nlby.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"

llm = AzureChatOpenAI(
    model= 'gpt-3.5-turbo',
    temperature=0.1,
    api_version='2024-02-15-preview',
    api_key='91a2c0eb89e24d8faa9c8075b63be35a',
    azure_endpoint='https://cog-ovavoxv55nlby.openai.azure.com/',
    azure_deployment='chat'
)
def extract_text_from_pdf(pdf_path,chunk_size=800,chunk_overlap=100):
    #loader = UnstructuredPDFLoader(pdf_path,mode='elements', strategy='fast')

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # text_splitter = CharacterTextSplitter(
    #     separator="\&",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    #doc = text_splitter.create_documents(documents)
    doc = text_splitter.split_documents(documents)
    return doc



pdf_path = '/Users/HALCYON007/Downloads/Error_codes (1).pdf'

# Extract text from the PDF

pdf_text =extract_text_from_pdf(pdf_path)


# Convert error codes text into Document objects


embeddings = AzureOpenAIEmbeddings(
    deployment="embeddingsmall",
    model="text-embedding-3-small",
    chunk_size=1000
)



db = Chroma.from_documents(pdf_text, embeddings)

def pdf_reader(query,k=2):
    matching_docs = db.similarity_search(query,k=k)
    return matching_docs


pdf_tool = Tool(
    name='pdf_reader',
    func=pdf_reader,
    description="Useful for whenever you ask questions related to error code documentation to find the solution."
)

tools = [pdf_tool]
def create_agents(llm, tools: list, system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}



members = ['rag_controller']

system_prompt = (
    "As a supervisor, your role is to oversee the insight between these"
    " workers: {members}. Based on the user's request,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back their findings and progress."
    " Once all tasks are completed, indicate 'FINISH'."
)

options = ["FINISH"] + members

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
        "required": ["next"]
    }
}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system",
     "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), members=", ".join(members))

supervisor_chain = (prompt | llm.bind_functions(
    functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

rag_controller = create_agents(
    llm,
    tools,
    """Refer strictly to the provided error code documentation. Extract and list only the 'Possible Causes' and all 'Remedies' separately for the specified error code.
     Do not include any information that is not explicitly present in the documentation."""
)

# sql_agent_node = functools.partial(agent_node, agent=sql_agent, name="sql_agent")
rag_controller_node = functools.partial(agent_node, agent=rag_controller, name="rag_controller")




class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    schema: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# Create workflow or graph
workflow = StateGraph(AgentState)

# adding nodes
workflow.add_node(key="supervisor", action=supervisor_chain)
#workflow.add_node(key="sql_agent", action=sql_agent_node)
workflow.add_node(key="rag_controller", action=rag_controller_node)

#workflow.add_node(key="Rag_reader",action = Rag_reader_node)


for member in members:
    workflow.add_edge(start_key=member, end_key="supervisor")

conditional_map = {k: k for k in members}
conditional_map['FINISH'] = END

# if task is FINISHED, supervisor won't send task to agent, else,
# the supervisor will keep on sending task to agent untill done, this is
# what the conditional edge does.
workflow.add_conditional_edges(
    "supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()
# while True:
#     input_error_code = input("Enter error code (or 'exit' to quit): ")
#     if input_error_code.lower() == 'exit':
#         break
#
#     input_data = {"messages": [HumanMessage(content=input_error_code)]}
#     response = graph.invoke(input_data)
#     messages = response["messages"]
#
#     for message in messages:
#         if isinstance(message, HumanMessage):
#             print("question:", message.content)
#         else:
#             print(message)
# Initialize Streamlit app
st.title("Error Code Processor with Chat History")

# List to hold all chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for chat history
with st.sidebar:
    st.header("Chat History")
    for idx, entry in enumerate(st.session_state.chat_history):
        st.write(f"**Input {idx + 1}:** {entry['input']}")
        for message in entry["responses"]:
            if isinstance(message, HumanMessage):
                st.write(f"question: {message.content}")
            else:
                st.write(message)

# Input error code
error_code = st.text_input("Enter error code:")

# Add a button to submit the error code
if st.button("Submit"):
    if error_code:
        # Prepare input data
        input_data = {"messages": [HumanMessage(content=error_code)]}

        # Invoke graph function
        response = graph.invoke(input_data)
        messages = response["messages"]

        # Store the input and its responses in chat history
        st.session_state.chat_history.append({"input": error_code, "responses": messages})
    else:
        st.warning("Please enter an error code.")

# Add an exit button to stop the app
if st.button("Exit"):
    st.stop()