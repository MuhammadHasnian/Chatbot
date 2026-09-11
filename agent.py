import os
import sqlite3
import uuid
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()


class DataBase:
    @classmethod
    def database_connection(cls):
        conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
        checkpoint = SqliteSaver(conn)  # Keeping variable clean
        checkpoint.setup()
        return checkpoint

    @classmethod
    def chat_history(cls):
        # Getting thread_id 
        connection = sqlite3.connect("chatbot_memory.db")
        cursor = connection.cursor()

        thread_ids = cursor.execute("SELECT distinct thread_id FROM checkpoints")
        readable_thread_ids = thread_ids.fetchall()
      
        readable_thread_ids = [thread_id[0] for  thread_id in readable_thread_ids]

        # Getting first message of the user 
        chat_name_id = {}
        message_list = []
        for thread_id in readable_thread_ids:
            conn = sqlite3.connect("chatbot_memory.db",check_same_thread=False)
            Checkpointer = SqliteSaver(conn)
            config = {"configurable":{"thread_id":thread_id}}
            checkpoint = Checkpointer.get(config)
            checkpoint = checkpoint['channel_values']['messages']
            
           
            for message in checkpoint:
                message = list(message)

                message = message[0]
               
                message = message[1]
                # print(message)
                
                
                if message not in message_list:
                   
                    chat_name_id[message] = thread_id
                    message_list.append(message)
                  
                break
    
        return chat_name_id 

    @classmethod
    def chat_history_data(cls,selected_id):
        conn = sqlite3.connect("chatbot_memory.db",check_same_thread=False)
        Checkpointer = SqliteSaver(conn)
        config = {"configurable":{"thread_id":selected_id}}
        checkpoint = Checkpointer.get(config)
        checkpoint = checkpoint['channel_values']['messages']
   
        chat_data_list = []
        for idx,message in enumerate(checkpoint):
            message = list(message)
            if idx%2 != 0:
                chat_data_list.append({"role":"assistant","content":message[0][1]})
               
            else :
                chat_data_list.append({"role":"user","content":message[0][1]})

        return chat_data_list
                


class AiAgent:
    @classmethod
    def get_date(cls):
        """Get the current date"""
       
        return datetime.now().strftime("%Y-%m-%d")

    def __init__(self):
        search_tool = TavilySearchResults()

        data_base = DataBase()
        checkpoint = DataBase.database_connection()
        llm = self.llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

        system_prompt = """
        You are a helpful assistant.
        Whenever user said who build you or who made you then answer them exactly i That chatbot is made by Muhammad Hasnain
        Answer all user's queries.
        ONLY use the get_date tool if the user is explicitly asking about today's date.
        Use the search tool for answering questions that require up-to-date information.
        """

        self.agent = create_agent(
            model=llm,
            tools=[self.get_date, search_tool],
            system_prompt=system_prompt,
            checkpointer=checkpoint,
        )

    
    def chat(self,message, history, thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]}, config
        )
        last_response = response["messages"][-1].content
        return last_response



class WebInterFace:
    def __init__(self):
        ai_agent = AiAgent()
        with gr.Blocks() as demo:
            gr.Markdown("# AI Chatbot")
            thread_id = gr.State(value=lambda: str(uuid.uuid4()))

            # Buttons
            new_chat = gr.Button("New Chat")
           
            # Chat_interface 
            with gr.Column(visible=False) as chatbot__container:
               chat_interface = gr.ChatInterface(fn=ai_agent.chat, 
                                                 additional_inputs=[thread_id])

            #  Implement New Chat button
            def new_chat_clicked():
                return str(uuid.uuid4()),gr.update(visible=True),[]

            new_chat.click(
                fn=new_chat_clicked,
                outputs =[thread_id,chatbot__container,chat_interface.chatbot]
            )

           
            chat_name_id = DataBase.chat_history()
            
            # Implement Previous Chat feauture
            previous_chat_id = gr.Radio(
                 choices=list(chat_name_id.keys()),
                label = "previous_chats",
               
            )

            def check_previous_chat_id(selected_id):
                database = DataBase()
                thread_id = chat_name_id[selected_id]
                
                chat_data_list = database.chat_history_data(thread_id)
                return  thread_id,chat_data_list,gr.update(visible=True)

        
            previous_chat_id.change(fn=check_previous_chat_id,
                                    inputs=previous_chat_id,outputs=[thread_id,chat_interface.chatbot,chatbot__container])

            

        demo.launch()


web_interface = WebInterFace()

