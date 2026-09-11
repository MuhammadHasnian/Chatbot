import os
import sqlite3
import uuid
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()


def get_date():
    """Get the current date"""
    print("I was called")
    return datetime.now().strftime("%Y-%m-%d")


search_tool = TavilySearchResults()

conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)  # Keeping variable clean
checkpoint.setup()

llm = ChatOllama(model="qwen2.5:0.5b")


system_prompt = """
You are a helpful assistant.
Answer all user's queries.
ONLY use the get_date tool if the user is explicitly asking about today's date.
Use the search tool for answering questions that require up-to-date information.
"""

agent = create_agent(
    model=llm,
    tools=[get_date, search_tool],
    system_prompt=system_prompt,
    checkpointer=checkpoint,
)


def chat(message, history, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]}, config
    )
    last_response = response["messages"][-1].content
    return last_response


with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    thread_id = gr.State(value=lambda: str(uuid.uuid4()))

    # Buttons
    new_chat = gr.Button("New Chat")
    previous_chats = gr.Button("Previous Chats")

    # Chat_interface 
    with gr.Column(visible=False) as chatbot__container:
        gr.ChatInterface(fn=chat, additional_inputs=[thread_id])

    #  Implement Buttons
    def new_chat_clicked():
        return str(uuid.uuid4()),gr.update(visible=True)

    new_chat.click(
        fn=new_chat_clicked,
        outputs =[thread_id,chatbot__container]
    )

    
        

   





    

  


demo.launch()

