from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
# import easygui
# from langchain.vectorstores import Chroma
# from langchain.indexes import VectorstoreIndexCreator
import os
# from pypdf import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import tiktoken
from tempfile import NamedTemporaryFile, gettempdir
import pyautogui

# from utils_test import *

load_dotenv()

api_key = st.secrets["openai"]["api_key"]

st.set_page_config(page_title="Chat with your Documents",
                   page_icon=":file_folder:", layout="wide")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
colT1,colT2,colT3 = st.columns([1, 8, 18])
with colT3:
    st.title("Document Talk" + ":file_folder:")
st.divider()

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3,return_messages=True)

if "documents" not in st.session_state:
    st.session_state.documents = []

if "process_button" not in st.session_state:
    st.session_state.process_button = False

if "end_session" not in st.session_state:
    st.session_state.end_session = False

if "memory" not in st.session_state:
    st.session_state.memory = {}

# declaring LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context only, 
and if the answer is not contained within the text below, just say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


# container for chat history
response_container = st.container()
# container for text box
# textcontainer = st.container()

# def get_pdf_text(pdf_docs):
#     for pdf in pdf_docs:
#         file_path = os.path.abspath(pdf.name)
#         mod_file_path = str(file_path).replace(r"\\\\", r"\\")
#         loader = UnstructuredFileLoader(mod_file_path)
#         documents = loader.load()
#         return documents

def get_file_text(file):
    loader = UnstructuredFileLoader(file)
    return loader.load()

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks = []):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #initialize pinecone
    pinecone.init(
    api_key="ac1df929-518e-4118-b215-3a446ab9215f",  # find at app.pinecone.io
    environment="asia-southeast1-gcp-free"  # next to api key in console
    )
    #
    index_name = "cqa-chatbot"
    vectorstore = Pinecone.from_documents(text_chunks, embedding=embeddings, index_name="cqa-chatbot")
    # vectorstore = Chroma.from_documents(text_chunks, embedding=embeddings)
    return vectorstore


def delete_session_state():
    del st.session_state['documents']


def reset_chat():
    del st.session_state.responses

    del st.session_state.memory

    del st.session_state.requests

    del st.session_state.buffer_memory

def price_tokens(input_string: str or list, output_string: str, model_name: str) -> float:
    # Returns number of tokens in a string
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens_input = len(encoding.encode(input_string))
    num_tokens_output = len(encoding.encode(output_string))
    if model_name == "text-davinci-003":
        return ((num_tokens_input + num_tokens_output) * (0.02 / 1000))
    elif model_name == "gpt-3.5-turbo":
        return ((num_tokens_input * (0.0015 / 1000)) + (num_tokens_output * (0.002 / 1000)))


def main():
    css = r'''
        <style>
            [data-testid="stForm"] {border: 0px}
        </style>
    '''

    st.markdown(css, unsafe_allow_html=True)
    hide_label = """
    <style>
        .css-7oyrr6 eex3ynb0{
            display: none
    </style>
    """
    st.markdown(hide_label, unsafe_allow_html=True)
    raw_text = ""
    with st.sidebar:
        colT1, colT2, colT3 = st.columns([1, 8, 44])
        with colT2:
            st.sidebar.write("**Upload your own documents and chat!**" + " " + "💬")
        if "docs" not in st.session_state:
            st.session_state.docs = False
        colT1, colT2, colT3 = st.columns([1, 8, 64])
        with st.form(clear_on_submit=True, key="file-upload-form"):
            st.session_state.docs = st.file_uploader("File upload", accept_multiple_files=True)
            colT1, colT2, colT3 = st.columns([1, 8, 24])
            with colT3:
                submitted = st.form_submit_button("Upload")

        if (st.session_state.docs is not None) and (submitted is not False):
            for uploaded_file in st.session_state.docs:
                file_extension = uploaded_file.name.split(".")[-1]

                with NamedTemporaryFile(dir=gettempdir(), suffix='.' + file_extension, delete=False) as f:
                    f.write(uploaded_file.getbuffer())
                    raw_text = get_file_text(f.name)
            # file_absolut_path = easygui.fileopenbox(title='Add File', default="*", multiple=True)
            # if file_absolut_path is not None:
            #     for file in file_absolut_path:
            #         emoji_shortcode = ":page_with_curl:"
            #         st.write(emoji_shortcode + " " + os.path.basename(file))
            #         raw_text = st.session_state.documents.append(get_file_text(file))

        # process button
        colT1, colT2, colT3 = st.columns([1, 8, 28])
        with colT3:
            st.session_state.process_button = st.button("Process" + " " + "⌛", key="process")
            st.text("")
        if st.session_state.process_button:
            with st.spinner("Processing..."):

                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

        # reloading session state
        colT1, colT2, colT3 = st.columns([1, 8, 27])
        with colT3:
            end_session = st.button("Reload" + " " + "🔄" , key="reload")
            st.text("")
        if end_session:
            with st.spinner("Reloading..."):
                delete_session_state()
                st.write("Session state expired. Please, upload your documents again. \U0001F614")

    form = st.form(key = "form_1")
    with form:
        query = st.text_input("Query: ", key="input", placeholder="Write your query")
        submit_button = st.form_submit_button(label = "Submit")
        with st.container():
            slider = st.slider("Source of answer:", min_value=1, max_value=5, value=1, help="This slider lets you decide how many relevant pieces of information should the model use for output")
            st.divider()
            checkbox_col_1, checkbox_col_2 = st.columns(2)
            with checkbox_col_1:
                source = st.checkbox("Source", key = "source_checkbox", help="For source of documents")
            with checkbox_col_2:
                refined_query_checkbox = st.checkbox(" Refined Query", key = "refined_query", help = "For seeing refined query of your original query")
                st.divider()
                # price_checkbox = st.checkbox("Price of query", key="price_checkbox")

        col_1, col_2 = st.columns(2)
        with col_2:
            price_checkbox = st.checkbox("Price of query", key = "price_checkbox")
        if (query != "") and (source is not False):
            with st.spinner("Typing..."):
                conversation_string = get_conversation_string()
                # st.code(conversation_string)
                refined_query = query_refiner(conversation_string, query)
                price_1 = price_tokens(input_string=conversation_string, output_string=refined_query, model_name="text-davinci-003")
                with st.container():
                    with checkbox_col_2:
                        if refined_query_checkbox:
                            st.write("<b> <font color = 'grey'> Refined Query  </font> </b> \n\n ", refined_query, unsafe_allow_html=True)
                            st.divider()
                    context = find_match(refined_query, k = slider)
                    with checkbox_col_1:
                        st.divider()
                        st.write("<b> --- Source documents --- </b>\n\n",unsafe_allow_html=True)
                        st.divider()
                        st.write(context)
                        st.divider()
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                with st.container():
                    with checkbox_col_2:
                        # price_checkbox = st.checkbox("Price of query", key="price_checkbox")
                        if price_checkbox is not False:
                            with col_2:
                                price_1 = price_tokens(input_string=conversation_string, output_string=refined_query,
                                                       model_name="text-davinci-003")
                                price_2 = price_tokens(input_string=context + query, output_string=response,
                                                       model_name="gpt-3.5-turbo")
                                st.write("Price:💲" + str(round(price_1 + price_2, 4)))
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
        elif query != "":
            with st.spinner("Typing..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                with st.container():
                    with checkbox_col_2:
                        if refined_query_checkbox:
                            st.write("<b> <font color = 'grey'> Refined Query  </font> </b> \n\n ", refined_query, unsafe_allow_html=True)
                            st.divider()
                context = find_match(refined_query, k=slider)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                with st.container():
                    with col_2:
                        # price_checkbox = st.checkbox("Price of query", key = "price_checkbox")
                        if price_checkbox is not False:
                            with col_2:
                                price_1 = price_tokens(input_string=conversation_string, output_string=refined_query,model_name="text-davinci-003")
                                price_2 = price_tokens(input_string=context+query, output_string=response, model_name="gpt-3.5-turbo")
                                st.write("Price:💲"+ str(round(price_1 + price_2,4)))
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
        elif query == "":
            st.error("Can't send an empty query" + " "+"🚨")


    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

    # if (st.button("Reset Chat")) and (query != ""):
    #     pyautogui.hotkey("ctrl", "F5")

    # st.button("Reset", on_click=reset_chat())


if __name__ == '__main__':
    main()

