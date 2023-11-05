from dotenv import load_dotenv as load_env
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


load_env()

st.set_page_config(page_title='hackathon')
st.header('pdf knowledge library')

pdf = st.file_uploader('upload pdf file', type='pdf')

# extract text
if pdf is not None:
    text = ''
    with pdfplumber.open(pdf) as reader:
        for page in reader.pages:
            text += page.extract_text()

    # text splitting
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len
                                          )

    print(text)
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    user_question = st.text_input("please ask me:")

    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
        st.write(response)
