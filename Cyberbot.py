import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import docx
import openpyxl
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_from_files(files):
    text = ""
    for file in files:
        if file.name.endswith('.pdf'):
            text += get_pdf_text(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            text += get_excel_text(file)
        elif file.name.endswith('.csv'):
            text += get_csv_text(file)
        elif file.name.endswith('.txt'):
            text += get_txt_text(file)
        elif file.name.endswith('.docx'):
            text += get_docx_text(file)
    return text


def get_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_excel_text(file):
    df = pd.read_excel(file)
    return df.to_string()


def get_csv_text(file):
    df = pd.read_csv(file)
    return df.to_string()


def get_txt_text(file):
    return file.read().decode('utf-8')


def get_docx_text(file):
    doc = docx.Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.


    Context:
    {context}


    Question: 
    {question}


    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Welcome to the CyberBot")
    user_question = st.text_input("Ask a Question from the Uploaded Files")


    if user_question:
        user_input(user_question)


    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload your files and click on the Submit & Process button", accept_multiple_files=True, type=['pdf', 'xls', 'xlsx', 'csv', 'txt', 'docx'])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_files(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
