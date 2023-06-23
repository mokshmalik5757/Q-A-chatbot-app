import os

from sentence_transformers import SentenceTransformer
import pinecone
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Chroma
import openai
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
api_key = st.secrets["openai"]["api_key"]
openai_api_key= api_key
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='ac1df929-518e-4118-b215-3a446ab9215f', environment='asia-southeast1-gcp-free')
index = pinecone.Index('cqa-chatbot')

def find_match(input, k):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=k, includeMetadata=True)
    matches = result['matches']
    text_results = ""
    for match in matches:
        text_results += match['metadata']['text'] + "\n"
    return text_results
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(api_key=openai_api_key,
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base. You don't have to do that for every query. Only and Only do it for those queries which have a relevant conversation log with them.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):

        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
