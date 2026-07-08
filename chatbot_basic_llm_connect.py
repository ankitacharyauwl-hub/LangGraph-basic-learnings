import os
from typing import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
load_dotenv()
# declaring State
class State(TypedDict):
    messages: Annotated[list,add_messages]
# initializing llm
llm = ChatOpenAI(model="gpt-4o-mini")
# The node function receives the current state and returns an update
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
# 1. Initialize the graph builder
builder = StateGraph(State)

# 2. Add the chatbot node to the graph
builder.add_node("chatbot", chatbot)

# 3. Connect the entry point and exit point to the node
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 4. Initialize the memory saver and compile
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
if __name__ == "__main__":
    # The thread_id determines the memory session
    config = {"configurable": {"thread_id": "session-1"}}
    
    print("Chatbot Ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Stream the response back
        events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
        for event in events:
            latest_message = event["messages"][-1]
            
        if latest_message.type == "ai":
            print(f"Bot: {latest_message.content}\n")
