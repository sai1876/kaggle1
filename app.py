import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

# Map Streamlit Secrets to environment variables if running in Streamlit Cloud
try:
    for key, value in st.secrets.items():
        if not os.getenv(key):
            os.environ[key] = str(value)
except Exception:
    pass

# App configuration
possible_paths = []
env_corpus = os.getenv("CORPUS_PATH")
if env_corpus:
    possible_paths.append(env_corpus.strip().replace('"', '').replace("'", ""))
possible_paths.extend([
    "zyro-dynamics-hr-corpus",
    ".",
    r"C:\Users\cheru\Downloads\niat-masterclass-rag-challenge\zyro-dynamics-hr-corpus"
])
CORPUS_PATH = None
for p in possible_paths:
    if p and os.path.exists(p):
        if os.path.exists(os.path.join(p, "00_Company_Profile.pdf")):
            CORPUS_PATH = p
            break
if not CORPUS_PATH:
    CORPUS_PATH = "zyro-dynamics-hr-corpus"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash") # Default to gemini-2.5-flash as gemini-3.5-flash is not a valid model name

st.set_page_config(
    page_title="Acrux Dynamics HR Portal",
    page_icon="🤖",
    layout="centered"
)

# Custom premium CSS (Glassmorphism & Gradients)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Base body adjustments */
.main {
    background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
    color: #f8fafc;
}

.stApp {
    background: transparent;
}

/* Glassmorphism Title Card */
.title-container {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
}

.title-text {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    margin: 0;
    letter-spacing: -0.5px;
}

.subtitle-text {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 10px;
    font-weight: 300;
}

/* Modern chat bubbles */
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}

/* Source Expander styling */
.streamlit-expanderHeader {
    background-color: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-container"><h1 class="title-text">Acrux Dynamics</h1><div class="subtitle-text">Internal HR Help Desk & Policy Assistant</div></div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing RAG pipeline & indexing documents...")
def init_rag_pipeline(corpus_path):
    if not os.path.exists(corpus_path):
        return None, None, f"Corpus path not found: {corpus_path}. Please check your environment configuration or set the correct CORPUS_PATH environment variable."
    
    # Load and process docs
    try:
        loader = PyPDFDirectoryLoader(corpus_path)
        documents = loader.load()
        if not documents:
            return None, None, f"No PDF documents found in corpus path: {corpus_path}."
    except Exception as e:
        return None, None, f"Error loading documents: {str(e)}"
    
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        return None, None, f"Error building vector index: {str(e)}"
    
    # Initialize LLM
    try:
        if LLM_PROVIDER == "groq":
            llm = ChatGroq(model=LLM_MODEL, temperature=0.1)
        elif LLM_PROVIDER == "openai":
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
        else:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None, None, "API key for Gemini is missing. Please set GOOGLE_API_KEY or GEMINI_API_KEY in the Streamlit Secrets or environment variables."
            
            # Map gemini-3.5-flash or others to gemini-2.5-flash
            model_name = LLM_MODEL
            if "3.5" in model_name:
                model_name = "gemini-2.5-flash"
            
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)
    except Exception as e:
        return None, None, f"Error initializing LLM: {str(e)}"
        
    return retriever, llm, None

# Initialize RAG Pipeline
retriever, llm, init_error = init_rag_pipeline(CORPUS_PATH)

if init_error:
    st.error(init_error)
    if "API key" in init_error:
        st.info("""
        **How to configure API Keys on Streamlit Cloud:**
        1. Go to your **Streamlit Cloud Dashboard**.
        2. Click the three dots next to your deployed app and choose **Settings**.
        3. Select **Secrets** on the left menu.
        4. Enter your secrets in TOML format:
           ```toml
           GOOGLE_API_KEY = "your_actual_gemini_api_key"
           # Optional:
           LANGCHAIN_API_KEY = "your_langsmith_api_key"
           LANGCHAIN_TRACING_V2 = "true"
           LANGCHAIN_PROJECT = "zyro-rag-challenge"
           ```
        5. Click **Save** and the app will redeploy automatically.
        """)
elif retriever is None or llm is None:
    st.error(f"Corpus path not found: {CORPUS_PATH}. Please check your environment configuration or set the correct CORPUS_PATH environment variable.")
else:
    # Set up prompts
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an HR Assistant for Acrux Dynamics.
Answer the employee's question using ONLY the provided policy context. Do not assume or extrapolate.
If the answer is not in the context, respond with: "I am sorry, but I do not have that information in the internal policy documents."

Note: The context refers to the company as 'Zyro Dynamics' and its products as 'ZyroCRM'. In your final answer, you must swap 'Zyro' with 'Acrux' (e.g., 'Acrux Dynamics', 'AcruxCRM') so that the response is tailored to 'Acrux Dynamics'.

Context:
{context}
"""),
        ("human", "{question}")
    ])
    
    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a classification assistant for the Acrux Dynamics HR Help Desk.
Your job is to classify if the employee's question is IN_SCOPE or OUT_OF_SCOPE.

IN_SCOPE questions are related to employee HR policies and processes of Acrux Dynamics, such as:
- Leave policies (Earned Leave, Sick Leave, Maternity/Paternity Leave, etc.)
- Salary payment credit dates and cut-off cycles
- Compensation, CTC ranges, and bonuses for employee grades (e.g., L4 Senior)
- Health insurance coverage, premium arrangements, and who is covered
- Performance reviews, appraisals, APR timelines, promotion letters, and Performance Improvement Plans (PIP)
- Work from home (WFH) eligibility and arrangements
- Job application processes and recruitment guidelines for Acrux Dynamics
- Employee stock options (ESOPs) vesting schedules and allocations

OUT_OF_SCOPE questions include:
- Questions about other companies' policies (e.g., Zoho, Freshworks, etc.)
- Questions about software product features, sales, comparison with competitors (e.g., AcruxCRM product features, Salesforce comparison)
- Questions about company financials, revenue, business performance (e.g., revenue last year)
- General knowledge, coding, or unrelated queries.

Respond with exactly "IN_SCOPE" or "OUT_OF_SCOPE". Do not add any other words or punctuation.
"""),
        ("human", "Question: {question}. Classification:")
    ])
    
    REFUSAL_MESSAGE = "I am sorry, but I can only answer questions related to Acrux Dynamics' internal HR policies, employee benefits, and HR processes."
    
    def get_clean_content(response):
        content = response.content if hasattr(response, 'content') else response
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, dict) and "text" in item)
        return str(content)

    def ask_bot(question: str):
        prompt_val = OOS_PROMPT.format_prompt(question=question)
        response = llm.invoke(prompt_val.to_messages())
        classification = get_clean_content(response).strip().upper()
        
        if "OUT_OF_SCOPE" in classification:
            return {"answer": REFUSAL_MESSAGE, "source_documents": []}
        else:
            # Preprocess query: Rewrite Acrux -> Zyro
            query_for_retrieval = question.replace("Acrux", "Zyro").replace("acrux", "zyro")
            docs = retriever.invoke(query_for_retrieval)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt_val = RAG_PROMPT.format_prompt(context=context, question=question)
            res = llm.invoke(prompt_val.to_messages())
            answer = get_clean_content(res)
            return {"answer": answer, "source_documents": docs}

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View Policy Sources"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source {idx}:** {src['file']} (Page {src['page']})")
                        st.caption(src["text"])

    # User input
    if user_query := st.chat_input("Ask a question about leave, salary, insurance, etc..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            with st.spinner("Searching internal policies..."):
                res = ask_bot(user_query)
                st.markdown(res["answer"])
                
                sources = []
                for doc in res["source_documents"]:
                    sources.append({
                        "file": os.path.basename(doc.metadata.get("source", doc.metadata.get("source_file", "Policy Document"))),
                        "page": doc.metadata.get("page", 0) + 1,
                        "text": doc.page_content
                    })
                
                if sources:
                    with st.expander("View Policy Sources"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**Source {idx}:** {src['file']} (Page {src['page']})")
                            st.caption(src["text"])
                            
        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "sources": sources
        })
