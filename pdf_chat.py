from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import gradio as gr
load_dotenv()

def build_index(pdf_path):
    pages    = PyPDFLoader(pdf_path).load()                    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100) #settings 
    chunks   = splitter.split_documents(pages)                #creating chunks with settings
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")   #settings for embedding model
    db       = Chroma.from_documents(chunks, embedder)       #mapping the data in db based upon chunks and embedding model 
    return db

model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
) #settings for model

prompt = ChatPromptTemplate.from_template("""
    You are a helpful PDF assistant. Answer the question using ONLY the context below.
    If the context doesn't contain the answer, say "I couldn't find that in the document."
    After your answer, list the page numbers you used as: Sources: page X, page Y.

Context:
{context}

Question: {question}
""")

chain = prompt | model

def ask_question(db, question):
    docs = db.similarity_search(question, k=3) #searching for the question in the db and returning top 3 results
    context = "\n".join([doc.page_content for doc in docs]) #joining the content of the top 3 results
    response = chain.invoke({"context": context, "question": question}) #running the model with the context and question
    return response.content

state = {"db": None}                    # ① remember the index across turns

def upload(pdf):
    state["db"] = build_index(pdf.name)   # ② re-index whenever a new PDF arrives
    return "✅ PDF indexed! Ask me anything about it."

def chat(message, history):
    if state["db"] is None:
        return "Please upload a PDF first 📄"
    return ask_question(state["db"], message)

with gr.Blocks(title="📄 Chat with your PDF") as demo:
    gr.Markdown("## 📄 Chat with your PDF (powered by RAG)")
    pdf    = gr.File(label="Upload a PDF", file_types=[".pdf"])
    status = gr.Markdown()
    pdf.upload(upload, inputs=pdf, outputs=status)
    gr.ChatInterface(fn=chat)

demo.launch(share=True)     
