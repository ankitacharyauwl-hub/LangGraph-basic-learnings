import os
from dotenv import load_dotenv
load_dotenv()
from typing import List,TypedDict
from langgraph.graph import START, END, StateGraph
from openai import OpenAI

client = OpenAI()

class Name_Modify(TypedDict):
    Input_Name : str
    Meaning : str
    synonym_list :List[str]

def suggest_name_node(state: Name_Modify):
    print(f"finding the synonym for {state['Input_Name']}")
    prompt = f""" The user likes the baby name '{state['Input_Name']}'. 
    1. Find the core meaning or vibe of this name.
    2. Suggest 3 other names that share a similar meaning or synonym-like vibe.
    Format your response exactly like this:
    Meaning: <meaning>
    Suggestions: <name1>, <name2>, <name3>"""
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = response.choices[0].message.content.strip().split("\n")

    meaning_text = response_text[0].replace("Meaning:", "").strip()
    suggestions_text = response_text[1].replace("Suggestions:", "").strip()

    suggestions_list = [name.strip() for name in suggestions_text.split(",")]

    return {
        "Meaning": meaning_text,
        "synonym_list": suggestions_list
    }

builder = StateGraph(Name_Modify)

builder.add_node("generator", suggest_name_node)

builder.add_edge(START, "generator")  # Start goes straight to generator
builder.add_edge("generator", END)    # Generator goes straight to the end

baby_name_agent = builder.compile()

initial_input = {"Input_Name": "aakrisha"}
final_output = baby_name_agent.invoke(initial_input)

print("\n--- Final Results ---")
print(f"Original: {final_output['Input_Name']}")
print(f"Meaning: {final_output['Meaning']}")
print(f"Synonyms: {final_output['synonym_list']}")
