# Using an LLM to loop through conversations and iteratively add more customer intents to the ontology

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialising the OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ok nice the openai integration works now :)!!

def chat(message):
    """Send a message and get a response"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

# Simple chat loop
print("Chat started! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    response = chat(user_input)
    print(f"AI: {response}")

print("Chat ended!")