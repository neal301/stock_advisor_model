import langchain
import langgraph
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
import langchain_core
import langchain_openai
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent, create_tool_calling_agent, initialize_agent, create_openai_tools_agent
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.tools import BaseTool
from langchain_core.agents import AgentAction, AgentFinish
import orjson
import json
import os
import openai
from typing import Literal, Union, Sequence, TypedDict, Annotated
import pandas as pd
from openai import OpenAI
import functools
import operator
import logging
import streamlit as st
from langchain_openai import AzureChatOpenAI
from typing import Optional
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.chains.query_constructor.ir import Comparator, Comparison, Operation, Operator, StructuredQuery
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState

os.environ["AZURE_OPENAI_API_KEY"]="b229fdc63a3d4b06b4adcc67660474f3"
os.environ["AZURE_OPENAI_ENDPOINT"]="https://openai-541.openai.azure.com/"
os.environ["ALPHAVANTAGE_API_KEY"]="5TW2G5K4N9GKMGRD"

pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', None)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)

from langchain_openai import AzureOpenAIEmbeddings
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint='https://aiall9596864698.cognitiveservices.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15',
    api_key='6d440529fad24ffc8aef6d5f9ef52593',
    openai_api_version="2024-07-01-preview"
)

options=["Ten-K-Expert"] + ["FINISH"]
import sys
import os
from io import StringIO

class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass


#original_stdout = sys.stdout  # Save a reference to the original stdout
#captured_output = StringIO()  # Create a new StringIO object



class ChromaRetrieverTool:
    def __init__(self, directory, collection, embedding, llm):
        self.directory = directory
        self.collection = collection
        self.embedding = embedding
        self.llm = llm

    def initiate_retriever(self):
        metadata_field_info = [
            AttributeInfo(
                name="date",
                description="The date of 10-k filing",
                type="string or list[string]",
            ),
            AttributeInfo(
                name="section",
                description="The section of the 10-k filing",
                type="string or list[string]",
            ),
            AttributeInfo(
                name="company",
                description="The name of the company",
                type="string or list[string]",
            ),
            ]
        
        document_content_description = "Chunked content of the 10-k filing"

        self.vector_store = Chroma(
            collection_name=self.collection,
            persist_directory=self.directory,
            embedding_function=self.embedding
        )

        self.retriever=SelfQueryRetriever.from_llm(
            self.llm,
            self.vector_store,
            document_content_description,
            metadata_field_info,
            verbose=True
        )


        self.retriever_tool=create_retriever_tool(
            self.retriever,
            '10-k document retriever',
            'Query a retriever to get 10-k document information',
        )

class AgentState(MessagesState):
    """State dictionary for the agent.
    
    Holds the messages and the next state to transition to.
    """
    next: str
    #intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

def create_agent(llm: AzureChatOpenAI, tools: list, system_prompt: str):
    """Creates an agent using the provided LLM and tools.
    
    Sets up a prompt template and creates an agent executor.
    
    :param llm: Language model to be used by the agent.
    :param tools: List of tools available to the agent.
    :param system_prompt: System-level prompt for the agent.
    :return: AgentExecutor instance configured with the provided settings.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)

    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return executor

def agent_node(state, agent, name):
    """Executes the agent with the given state and returns the result.
    
    Handles errors and formats the result into a message.
    
    :param state: Current state of the agent.
    :param agent: Agent instance to be executed.
    :param name: Name of the agent for identifying the source of the message.
    :return: Dictionary containing the result message.
    """
    try:
        result = agent.invoke(state)

        return {"messages": [HumanMessage(content=result["output"], name=name)]}
    except Exception as e:
        print(f"Error with agent: {e}")
        return {"messages": [HumanMessage(content="Error occurred", name=name)]}
    
def create_llm(
    model: str = "gpt-35-turbo",
    api_version: str = "2024-07-01-preview",
    temperature: float = 0,
    max_tokens: int = 500,
    timeout: int = None,
    max_retries: int = 2
) -> AzureChatOpenAI:
    """Creates an instance of ChatOpenAI with specified parameters.
    
    :param model: The model to use (default is "gpt-4o").
    :param temperature: Controls the randomness of the output (default is 0).
    :param max_tokens: Maximum number of tokens in the output (default is 500).
    :param timeout: Timeout duration for the request (default is None).
    :param max_retries: Number of retry attempts in case of failure (default is 2).
    :return: An instance of ChatOpenAI configured with the given parameters.
    """
    try:
        
        llm = AzureChatOpenAI(
            azure_deployment=model,  # or your deployment
            api_version=api_version,  # or your api version
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        return llm
    
    except Exception as e:
        # Log error message and re-raise exception for further handling if needed
        print(f"Error with LLM creation: {e}")
        raise

class AgentManager:
    def __init__(self, llm):
        """
        Initialize the AgentManager with the given language model.
        
        Args:
            llm: The language model instance to use for creating agents.
        """
        self.llm = llm
        
        # Initialize tools for different agents
        chroma_retriever = ChromaRetrieverTool(directory="../data/chroma_langchain_db_test", collection="test_collection", embedding=embeddings, llm=create_llm())
        chroma_retriever.initiate_retriever()
        self.tenk_tools = [chroma_retriever.retriever_tool]
        
        # Create the agents
        self.create_agents()
        
    def create_agents(self):
        """
        Create agents for Snowflake table operations, SQL querying, and documentation searching.
        """
        # Create Snowflake Expert Agent
        tenk_agent_system_prompt = (
            "You are an expert in 10-k filings. You can provide information about a company's 10-k filings."
        )
        self.tenk_agent = create_agent(
            self.llm, self.tenk_tools, tenk_agent_system_prompt
        )
        self.tenk_node = functools.partial(agent_node, agent=self.tenk_agent, name="Ten-K-Expert")
        




        # Create SQL Coder Agent
        #sql_agent_tools = [SnowparkQueryTool(snowpark_adapter=self.snowpark_adapter)]
        #sql_agent_system_prompt = "You are a Snowflake SQL expert."
        #self.sql_agent = create_agent(self.llm, sql_agent_tools, sql_agent_system_prompt)
        #self.sql_node = functools.partial(agent_node, agent=self.sql_agent, name="SQL-Coder")
        
        # Create Documentation Agent
        #documentation_agent_system_prompt = "You are an expert in the company's documentation."
        #self.documentation_agent = create_agent(self.llm, self.documentation_tools, documentation_agent_system_prompt)
        #self.documentation_node = functools.partial(agent_node, agent=self.documentation_agent, name="Documentation-Expert")

        
    def get_nodes(self):
        """
        Retrieve the nodes for each agent.
        
        Returns:
            dict: A dictionary where keys are agent names and values are their corresponding nodes.
        """
        return {
            "Ten-K-Expert": self.tenk_node
            #"SQL-Coder": self.sql_node,
            #"Snowflake-Expert": self.snowflake_node
        }

class WorkflowManager:
    def __init__(self, agent_manager, llm):
        """
        Initialize the WorkflowManager with the given AgentManager.
        
        Args:
            agent_manager: The instance of AgentManager to manage the workflow.
        """
        self.agent_manager = agent_manager
        self.workflow = StateGraph(AgentState)
        self.setup_workflow()
        self.llm = llm
    
    def setup_workflow(self):
        """
        Set up the workflow by adding nodes, configuring the supervisor, defining edges, and compiling the graph.
        """
        # Add nodes for each agent to the workflow
        for role, node in self.agent_manager.get_nodes().items():
            self.workflow.add_node(role, node)
        
        # Add the supervisor node to the workflow
        self.setup_supervisor()
        
        # Define the edges in the workflow
        self.define_edges()
        
        # Compile the graph to finalize the workflow setup
        self.graph = self.workflow.compile()
    
    def setup_supervisor(self):
        """
        Set up the supervisor node to manage the conversation between agents.
        """
        members = list(self.agent_manager.get_nodes().keys())
        options = ["FINISH"] + members
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the "
            "following workers: {members}. Then respond with the worker to act next. Each worker will perform a task and respond with their results "
            "and status. Continue this until you have enough information."
        )
        
        def supervisor_node(state: AgentState) -> AgentState:
            messages = [
                {"role": "system", "content": system_prompt},
            ] + state["messages"]
            response = self.llm.with_structured_output(Router).invoke(messages)
            next_ = response["next"]
            if next_ == "FINISH":
                next_ = END

            return {"next": next_}

        # Add the supervisor chain as a node in the workflow
        self.workflow.add_node("supervisor", supervisor_node)
    
    def define_edges(self):
        """
        Define the edges between nodes in the workflow.
        """
        members = list(self.agent_manager.get_nodes().keys())
        # Add edges from each member to the supervisor
        for member in members:
            self.workflow.add_edge(member, "supervisor")
        
        # Map member names to corresponding nodes and define conditional edges
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        self.workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
        # Add edge from the start node to the supervisor
        self.workflow.add_edge(START, "supervisor")

def my_query(myquery, graph):
    """
    Initiates the workflow graph with the provided query and streams the results.

    This function takes a user query, feeds it into the specified workflow graph, and processes the results as they are produced. 
    It prints each step of the result until the workflow reaches the end.

    Args:
        myquery (str): The user query to be processed by the workflow graph.
        graph (StateGraph): The workflow graph instance that manages the execution and processing of the query.

    Usage:
        Call this function with a query string and a graph instance to start the workflow and see the intermediate results.

    Example:
        my_query("change password", graph_instance)
    """
    # Stream the results of the graph execution based on the provided query

    print('Hello WOrld')

    final_state=graph.invoke(
        {"messages": [HumanMessage(content=myquery)]},
        {"recursion_limit": 100},
        #return_intermediate_steps=True,
    )

    print("Final State:", final_state)

     # Check the structure of final_state
    if 'messages' not in final_state or not final_state['messages']:
        print("Error: No messages in final_state")


    llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2024-07-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    )

    system_prompt_one='You are a special agent who has the key purpose of inspecting a conversation and answering the original user query given the conversation.'
    # Create the initial list
    initial_messages = [
        ("system", system_prompt_one)
    ]
    
    # Extend it with final_state['messages']
    initial_messages.extend(final_state['messages'])
    final=llm.invoke(initial_messages)

    if not final or not final.get("content"):
        print("Error: LLM returned an empty or invalid response.")
        final = {"content": "Error occurred, no valid response."}
    
    
    #st.session_state.messages.append({"text": final.content, "role": "assistant"})

    #st.markdown(final.content)
    #st.markdown(final_state["messages"][-1].content)

    return final_state, final






def main():
    
    """
    Main function to execute the workflow.
    
    This function sets up a Snowflake connection, initializes components such as the SQL adapter 
    and language model, creates agents and workflow, and then executes a query.
    """
    st.title("Ask Questions to your Own Personal Data Assistant:")
    st.write("""Ask your questions here.""")
    

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Initialize components
    llm = create_llm()
    
    if not llm:
        # Log error if LLM instance creation fails
        logging.error("Failed to create LLM instance.")
        st.error("Failed to create LLM instance.")
        return

    # Create agents and workflow
    try:
        # Initialize AgentManager with LLM and SQL adapter
        agent_manager = AgentManager(llm)
        # Initialize WorkflowManager with the created agents
        workflow_manager = WorkflowManager(agent_manager, llm)
        # Access the compiled workflow graph
        graph = workflow_manager.graph

    except Exception as e:
        # Log any exceptions that occur during agent or workflow setup
        logging.error(f"An error occurred while setting up agents or workflow: {e}")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if myquery := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(myquery)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": myquery})
    
        try:
            # Execute the query and process results
            final_state, final=my_query(myquery, graph)
        except Exception as e:
            # Log any exceptions that occur during query execution
            logging.error(f"An error occurred while executing the query: {e}")
        
    
        response = f"Bot: {final.content}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        return final_state, final

if __name__ == "__main__":
    #sys.stdout = captured_output


    final_state=main()
    #sys.stdout = original_stdout
    #output = captured_output.getvalue()

   # with st.expander("Verbose Output", expanded=False):
    #    if output:
    #        st.text(output)  # or st.write(output) for more flexibility
    #with st.expander("Agent Conversation", expanded=False):
     #   if final_state:
      #      for i in final_state[0]["messages"]:
       #         st.text(i.pretty_repr())  # Display each message as plain text



