import streamlit as st
import os
import io
import zipfile
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM

load_dotenv()
st.set_page_config(page_title="Resume Analyzer")
st.subheader("Analyze Your Resume with Respect to a Job")
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class GoogleGeminiLLM(LLM):
    model_name: str = "gemini-2.0-flash"
    
    def _call(self, prompt: str, stop=None) -> str:
        model_instance = genai.GenerativeModel(self.model_name)
        response = model_instance.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return ""
    
    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}
    
    @property
    def _llm_type(self) -> str:
        return "google_gemini"

llm = GoogleGeminiLLM()
job_desc = st.text_area("Enter Job Description...")
def get_pdf_text(pdf_files):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_chunks_text(text):
    """Split the extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectors(chunks):
    """Create a FAISS vector store from text chunks and save it locally."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")

def save_output(key, content):
    """Store the content under the given key in session_state."""
    if "saved_outputs" not in st.session_state:
        st.session_state["saved_outputs"] = {}
    st.session_state["saved_outputs"][key] = content

def create_zip_file(saved_outputs):
    """Create an in-memory zip file containing each output as a separate text file."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, content in saved_outputs.items():
            filename = f"{key}.txt"
            zf.writestr(filename, content)
    buffer.seek(0)
    return buffer

if "summary_output" not in st.session_state:
    st.session_state["summary_output"] = None
if "advice_output" not in st.session_state:
    st.session_state["advice_output"] = None
if "missing_output" not in st.session_state:
    st.session_state["missing_output"] = None
if "percentage_output" not in st.session_state:
    st.session_state["percentage_output"] = None
if "ats_output" not in st.session_state:
    st.session_state["ats_output"] = None
if "vss" not in st.session_state:
    st.session_state["vss"] = None

with st.sidebar:
    st.title("📂 Upload PDFs")
    pdf_docs = st.file_uploader("Upload your PDF files and click on 'Submit & Process'", type="pdf", accept_multiple_files=True)
    if st.button("Submit & Process", key="submit"):
        if pdf_docs:
            with st.spinner("Processing..."):
                text = get_pdf_text(pdf_docs)
                # Store the raw resume text for ATS scoring
                st.session_state["resume_text"] = text
                chunks = get_chunks_text(text)
                get_vectors(chunks)
                st.success("Processing Completed ✅")
        else:
            st.warning("Please upload at least one PDF file.")
    
   
    if "saved_outputs" in st.session_state and st.session_state["saved_outputs"]:
        zip_buffer = create_zip_file(st.session_state["saved_outputs"])
        st.download_button("Download Outputs Folder", data=zip_buffer, file_name="outputs.zip", mime="application/zip")

prompt_template_summary = """
You are an expert resume analyzer. Based on the resume provided in the context, provide a summary of each section in a clear, organized manner, and finally give an overall short summary.
Context:
{context}
Answer:
"""

prompt_template_ATS = """
You are an expert resume analyzer. Based on the resume provided in the context, provide the accurate ATS score of the resume.
Context:
{context}
Answer:
"""

prompt_template_advice = """
You are a career advisor. Analyze the resume provided in the context along with the job description provided below. Provide detailed advice on how the candidate can improve their skills to better match the job requirements.
Context:
{context}
Job Description:
{question}
Answer:
"""

prompt_template_missing_keywords = """
You are a resume expert. Given the resume provided in the context and the job description provided below, identify all relevant keywords for the job role and determine which keywords are missing in the resume. Explain your findings in detail.
Context:
{context}
Job Description:
{question}
Answer:
"""

prompt_template_percentage = """
You are a resume evaluator. Analyze the resume provided in the context along with the job description provided below, and determine a percentage match indicating how well the candidate's skills align with the job requirements. Provide a brief explanation of your evaluation.
Context:
{context}
Job Description:
{question}
Answer:
"""

col1, col2, col3 = st.columns([1, 0.1, 1])
with col1:
    button_summary = st.button("Tell me about the resume", key="summary")
    button_advice  = st.button("How can I improve my skills", key="advice")
with col3:
    button_missing    = st.button("Missing Keywords", key="missing")
    button_percentage = st.button("Percentage match", key="percentage")
    
col_left, col_center, col_right, x = st.columns([2, 2, 1, 1])
with col_center:
    button_ATS = st.button("ATS score", key="ats")

st.write('---------------------------------------------')
def load_vector_store_and_docs(query: str, k: int = 10):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        if query.strip() == "":
            query = "resume"
        docs = db.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error("Error loading vector store or retrieving documents. Please ensure you've processed the PDFs first.")
        return None
    

if button_summary:
    st.session_state["expander_state"] = "summary"
    docs = load_vector_store_and_docs(query="resume", k=10)
    if docs:
        prompt = PromptTemplate(template=prompt_template_summary, input_variables=["context"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": ""}, return_only_outputs=True)
        output_text = response['output_text']
        st.session_state["summary_output"] = output_text
        save_output("summary", output_text)

if button_advice:
    st.session_state["expander_state"] = "advice"
    if not job_desc:
        st.warning("Please enter the job description.")
    else:
        docs = load_vector_store_and_docs(query=job_desc, k=10)
        if docs:
            prompt = PromptTemplate(template=prompt_template_advice, input_variables=["context", "question"])
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain({"input_documents": docs, "question": job_desc}, return_only_outputs=True)
            output_text = response['output_text']
            st.session_state["advice_output"] = output_text
            save_output("advice", output_text)

if button_missing:
    st.session_state["expander_state"] = "missing"
    if not job_desc:
        st.warning("Please enter the job description.")
    else:
        docs = load_vector_store_and_docs(query=job_desc, k=10)
        if docs:
            prompt = PromptTemplate(template=prompt_template_missing_keywords, input_variables=["context", "question"])
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain({"input_documents": docs, "question": job_desc}, return_only_outputs=True)
            output_text = response['output_text']
            st.session_state["missing_output"] = output_text
            save_output("missing_keywords", output_text)

if button_percentage:
    st.session_state["expander_state"] = "percentage"
    if not job_desc:
        st.warning("Please enter the job description.")
    else:
        docs = load_vector_store_and_docs(query=job_desc, k=10)
        if docs:
            prompt = PromptTemplate(template=prompt_template_percentage, input_variables=["context", "question"])
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain({"input_documents": docs, "question": job_desc}, return_only_outputs=True)
            output_text = response['output_text']
            st.session_state["percentage_output"] = output_text
            save_output("percentage_match", output_text)

if button_ATS:
    st.session_state["expander_state"] = "ats"
    docs = load_vector_store_and_docs(query="resume", k=10)
    if docs:
        prompt = PromptTemplate(template=prompt_template_ATS, input_variables=["context"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": ""}, return_only_outputs=True)
        output_text = response['output_text']
        st.session_state["ats_output"] = output_text
        save_output("ATS Score", output_text)

if st.session_state["summary_output"]:
    with st.expander("Summary of the resume"):
        st.write(st.session_state["summary_output"])
        st.write('----------------')

if st.session_state["advice_output"]:
    with st.expander("How can I improve my skills"):
        st.write(st.session_state["advice_output"])
        st.write('----------------')

if st.session_state["missing_output"]:
    with st.expander("Missing Keywords in the resume"):
        st.write(st.session_state["missing_output"])
        st.write('----------------')

if st.session_state["percentage_output"]:
    with st.expander("Percentage Matched"):
        st.write(st.session_state["percentage_output"])
        st.write('----------------')

if st.session_state["ats_output"]:
    with st.expander("ATS Score"):
        st.write(st.session_state["ats_output"])
        st.write('----------------')

with st.sidebar:
    if st.session_state["summary_output"] != None and st.session_state["advice_output"] != None and st.session_state["percentage_output"] != None and st.session_state["ats_output"] != None and st.session_state ["missing_output"] != None:
        short_summary = st.button("overall short summary" , key = "short_summary_button")
        if short_summary:
            prompt = (
                f"{st.session_state['summary_output']} "
                f"{st.session_state['advice_output']} "
                f"{st.session_state['percentage_output']} "
                f"{st.session_state['ats_output']} "
                f"{st.session_state['missing_output']} "
                "Analyze the above summary_output , advice_output  ,percentage_output ,ats_output , missing_output and create a very short conclusive summary  with key points  for each topic only , then return it in a well-structured manner."
            )
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            st.session_state['vss'] = response.text

if st.session_state['vss']:
    with st.expander("Overall summary", expanded=True):
        st.write(st.session_state['vss'])
        st.write('----------------')