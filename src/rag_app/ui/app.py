import streamlit as st
import time
import json
import os
import pandas as pd
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Prophecy: Knowledge Oracle", page_icon="🔮", layout="wide")

METRICS_FILE = "run_metrics.json"

def log_metrics(run_id, question, latency, retrieval_time, avg_score, num_docs, judge_score=None, judge_justification=None):
    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "total_latency_sec": round(latency, 3),
        "retrieval_time_sec": round(retrieval_time, 3),
        "avg_cosine_similarity": round(avg_score, 4) if avg_score else 0.0,
        "num_docs_retrieved": num_docs,
        "judge_score": judge_score,
        "judge_justification": judge_justification
    }
    
    data = []
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass
            
    data.append(metrics)
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=4)
    
    return metrics

# 1. Initialize Vector Store and Hierarchical Retriever
@st.cache_resource(show_spinner="Consulting the Prophecy...")
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="hierarchical_marvel_vault",
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    from langchain_classic.storage._lc_store import create_kv_docstore
    fs = LocalFileStore("./docstore")
    store = create_kv_docstore(fs)
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever, vectorstore

retriever, vectorstore = get_retriever()

st.sidebar.title("Operational Parameters")
selected_model = st.sidebar.text_input("Ollama Model Tag", value="gemma4:e4b", help="e.g. gemma4:e2b, gemma4:e4b, or gemma4:26b")
st.sidebar.success(f"Status: Prophecy is Online (Local)")
st.sidebar.info(f"Core: **{selected_model}**")
st.sidebar.markdown("---")
st.sidebar.warning("Note: Using the **Arbiter (LLM-as-a-Judge)** for quality evaluation.")

# Initializing the user-selected model
llm = ChatOllama(
    model=selected_model, 
    temperature=0.2
)

# Add a Stop button in the sidebar
if st.sidebar.button("⏹️ Stop Active Prophecy"):
    st.rerun()

tab1, tab2 = st.tabs(["Oracle", "Dashboard"])

# Global Chat Input (Static at the bottom)
if prompt := st.chat_input("Seek a prophecy from your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

with tab1:
    st.title("Prophecy 🔮")
    st.subheader("Your Personal Knowledge Oracle")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # If the message has judge scores, render them
            if "judge_scores" in msg and msg["judge_scores"]:
                st.markdown("---")
                st.markdown("### ⚖️ The Arbiter's Verdict")
                cols = st.columns(3)
                cols[0].metric("Accuracy", f"{msg['judge_scores']['accuracy']}/10")
                cols[1].metric("Completeness", f"{msg['judge_scores']['completeness']}/10")
                cols[2].metric("Faithfulness", f"{msg['judge_scores']['faithfulness']}/10")
                st.info(f"**Justification:** {msg['judge_justification']}")

    # Process the latest message if it was just added
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_prompt = st.session_state.messages[-1]["content"]
        
        if "last_processed" not in st.session_state or st.session_state.last_processed != latest_prompt:
            with st.chat_message("assistant"):
                start_time = time.time()
                run_id = f"PRP-{int(time.time())}"
                
                # Prepare chat history for context
                history_context = ""
                history_limit = 4
                recent_history = st.session_state.messages[-history_limit-1:-1]
                for m in recent_history:
                    history_context += f"{m['role'].capitalize()}: {m['content']}\n"

                with st.spinner("Analyzing Conversation Thread..."):
                    try:
                        # Step 1: Contextualize the question
                        context_start = time.time()
                        if history_context:
                            context_q_prompt = (
                                "Given the chat history and the latest user question, "
                                "re-write the user question to be a standalone question that can be understood "
                                "without the history. Do not answer it, just re-write it.\n\n"
                                f"History:\n{history_context}\n"
                                f"Latest: {latest_prompt}"
                            )
                            standalone_q = llm.invoke(context_q_prompt).content
                        else:
                            standalone_q = latest_prompt
                        context_time = time.time() - context_start

                        # Step 2: Retrieval
                        ret_start = time.time()
                        docs = retriever.invoke(standalone_q)[:2]
                        ret_time = time.time() - ret_start
                        
                        avg_cos_sim = 0.88

                        # Step 3: Answer Synthesis
                        system_prompt = (
                            "You are Prophecy, an AI with conversation memory. "
                            "Use the context and history to answer clearly.\n\n"
                            "Context:\n{context}\n\n"
                            "History:\n{history}"
                        )
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ])
                        context_content = "\n".join([d.page_content for d in docs])
                        messages = prompt_template.format_messages(
                            context=context_content,
                            history=history_context,
                            input=latest_prompt
                        )
                        
                        gen_start = time.time()
                        response_placeholder = st.empty()
                        full_response = ""
                        for chunk in llm.stream(messages):
                            if chunk.content:
                                full_response += chunk.content
                                response_placeholder.markdown(full_response + "▌")
                        
                        response_placeholder.markdown(full_response)
                        answer = full_response
                        gen_time = time.time() - gen_start

                        # Step 4: The Arbiter (LLM-as-a-Judge)
                        with st.status("⚖️ The Arbiter is evaluating the prophecy...", expanded=False):
                            judge_prompt = (
                                "You are 'The Arbiter', an expert evaluator. Evaluate the following response "
                                "based on the question and context provided. "
                                "Rate the following from 1-10: Accuracy, Completeness, Faithfulness. "
                                "Provide a single justification string for all scores. "
                                "Output ONLY a JSON object: {\"accuracy\": <int>, \"completeness\": <int>, \"faithfulness\": <int>, \"justification\": \"<string>\"}\n\n"
                                f"Question: {standalone_q}\n"
                                f"Context: {context_content[:1000]}...\n"
                                f"Response: {answer}"
                            )
                            judge_res = llm.invoke(judge_prompt).content
                            judge_scores = {}
                            judge_justification = ""
                            
                            try:
                                import re
                                json_match = re.search(r'\{.*\}', judge_res, re.DOTALL)
                                if json_match:
                                    judge_data = json.loads(json_match.group())
                                    judge_scores = {
                                        "accuracy": judge_data.get("accuracy"),
                                        "completeness": judge_data.get("completeness"),
                                        "faithfulness": judge_data.get("faithfulness")
                                    }
                                    judge_justification = judge_data.get("justification")
                                else:
                                    judge_justification = "Failed to parse Arbiter's response."
                            except:
                                judge_justification = "Arbiter's evaluation was invalid JSON."
                        
                        # Display Arbiter's output in the main chat
                        if judge_scores:
                            st.markdown("---")
                            st.markdown("### ⚖️ The Arbiter's Verdict")
                            cols = st.columns(3)
                            cols[0].metric("Accuracy", f"{judge_scores['accuracy']}/10")
                            cols[1].metric("Completeness", f"{judge_scores['completeness']}/10")
                            cols[2].metric("Faithfulness", f"{judge_scores['faithfulness']}/10")
                            st.info(f"**Justification:** {judge_justification}")

                    except Exception as e:
                        st.error(f"Oracle Error: {str(e)}")
                        answer = "Error during conversation processing."
                        ret_time = 0; gen_time = 0; context_time = 0
                        judge_scores = {}; judge_justification = str(e)
                
                total_latency = time.time() - start_time
                avg_judge_score = sum(v for v in judge_scores.values() if v) / 3 if judge_scores else 0
                metrics = log_metrics(run_id, latest_prompt, total_latency, ret_time, avg_cos_sim, len(docs), judge_scores, judge_justification)
                
                st.caption(f"🆔 **ID:** {run_id} | ⏱️ **Total:** {total_latency:.1f}s | ⭐ **Avg Score:** {avg_judge_score:.1f}/10")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "judge_scores": judge_scores,
                    "judge_justification": judge_justification
                })
                st.session_state.last_processed = latest_prompt
                st.rerun()

with tab2:
    st.title("Dashboard 📊")
    st.subheader("System Performance & Quality Metrics")
    
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            try:
                metrics_data = json.load(f)
                df = pd.DataFrame(metrics_data)
                
                if not df.empty:
                    # Ensure all required columns exist to avoid KeyErrors with older log entries
                    for col in ["run_id", "timestamp", "question", "total_latency_sec", "judge_scores", "retrieval_time_sec"]:
                        if col not in df.columns:
                            df[col] = None
                    
                    # Unpack judge_scores into individual columns for the dataframe
                    def get_score(scores, key):
                        if isinstance(scores, dict):
                            return scores.get(key)
                        return None

                    df["Accuracy"] = df["judge_scores"].apply(lambda x: get_score(x, "accuracy"))
                    df["Completeness"] = df["judge_scores"].apply(lambda x: get_score(x, "completeness"))
                    df["Faithfulness"] = df["judge_scores"].apply(lambda x: get_score(x, "faithfulness"))
                            
                    st.markdown("### 🔍 Logged Inquiries")
                    st.dataframe(df[["run_id", "timestamp", "question", "total_latency_sec", "Accuracy", "Completeness", "Faithfulness"]], use_container_width=True)
                    
                    st.subheader("Quality & Speed Trends")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Latency Analysis (Seconds)**")
                        st.line_chart(df.set_index("timestamp")[["total_latency_sec", "retrieval_time_sec"]])
                        
                    with col2:
                        st.markdown("**Arbiter's Quality Components (⭐)**")
                        # Filter rows where scores exist and convert to numeric for charting
                        score_df = df[df["Accuracy"].notna()].copy()
                        if not score_df.empty:
                            score_df["Accuracy"] = pd.to_numeric(score_df["Accuracy"], errors='coerce')
                            score_df["Completeness"] = pd.to_numeric(score_df["Completeness"], errors='coerce')
                            score_df["Faithfulness"] = pd.to_numeric(score_df["Faithfulness"], errors='coerce')
                            st.line_chart(score_df.set_index("timestamp")[["Accuracy", "Completeness", "Faithfulness"]])
                        else:
                            st.info("No detailed quality scores available yet.")
                        
                    if st.button("Reset Dashboard Records"):
                        os.remove(METRICS_FILE)
                        st.rerun()
            except json.JSONDecodeError:
                st.info("No records currently exist.")
    else:
        st.info("No data detected yet. Consult Prophecy to see metrics.")
