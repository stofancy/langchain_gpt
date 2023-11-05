from io import StringIO
from dotenv import load_dotenv as load_env
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

load_env()

st.set_page_config(page_title='hackathon')
st.header('Hackathon')

uploaded_file = st.file_uploader(
    'upload meeting transcript file', type=['txt', 'md'])

if uploaded_file is not None:
    st.write(f"upload finished, parsing......")

    # To read file as bytes:
    file_bytes = uploaded_file.getvalue()
    # st.write(f"file bytes:\n {file_bytes}")

    # To convert to a string based IO:
    transcript = StringIO(file_bytes.decode('utf-8'))
    # st.write(f"file content:\n {transcript}")

    # To read file as string
    transcript_str = transcript.read()
    # st.write(f"file content:\n {transcript_str}")

    target_len = 100
    chunk_size = 3000
    chunk_overlap = 200

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_text(transcript_str)
    docs = [Document(page_content=t) for t in texts[:]]
    # st.write(len(transcript_str))
    # st.write(len(texts))
    # st.write(len(docs))

    openaichat = ChatOpenAI(temperature=0, model="gpt-4")
    prompt_template = """Act as a professional technical meeting minutes writer. 
    Tone: formal
    Format: Technical meeting summary
    Length:  200 ~ 300
    Tasks:
    - highlight action items and owners
    - highlight the agreements
    - Use bullet points if needed
    {text}
    CONCISE SUMMARY IN ENGLISH:"""
    PROMPT = PromptTemplate.from_template(prompt_template)
    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        f"Given the new context, refine the original summary in English within {target_len} words: following the format"
        "Participants: <participants>"
        "Discussed: <Discussed-items>"
        "Follow-up actions: <a-list-of-follow-up-actions-with-owner-names>"
        "If the context isn't useful, return the original summary. Highlight agreements and follow-up actions and owners."
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    chain = load_summarize_chain(
        openaichat,
        chain_type="refine",
        question_prompt=PROMPT,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
        verbose=True
    )
    resp = chain({"input_documents": docs}, return_only_outputs=True)
    st.write(resp["intermediate_steps"])
    st.write("*******FINAL RESULT********")
    st.write(resp["output_text"])
