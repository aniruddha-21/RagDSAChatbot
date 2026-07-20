import streamlit as st
from flashrank import Ranker
from langchain_ollama import ChatOllama


from Ragv2_Chatbot import (
    answer_question,
    get_vector_store,
    judge_answer,
    validate_question,
)

st.set_page_config(
    page_title="DSA RAG Chatbot",
    page_icon="📚",
    layout="centered",
)
DETAILS_LABEL = "Show all details"

@st.cache_resource
def load_pipeline():
    """Load expensive local resources only once."""
    vector_store = get_vector_store()

    reranker = Ranker(
        max_length=256,
    )

    llm = ChatOllama(
        model="mistral",
        temperature=0,
    )

    return vector_store, reranker, llm


def make_source_lines(scored_documents):
    """Format PDF pages and reranking scores for display."""
    return [
        (
            f"PDF page {document.metadata.get('page', 0) + 1} "
            f"| rerank score: {score:.4f}"
        )
        for document, score in scored_documents
    ]


st.title("📚 DSA RAG Chatbot")
st.caption("Ask any questions Related to Data Structure and Algorithms")

with st.sidebar:
    st.header("Controls")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by  ChromaDB, FlashRank, Ollama- Mistral \n ")


vector_store, reranker, llm = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("sources") or message.get("judge_result"):
            with st.expander(DETAILS_LABEL):
                for source in message.get("sources", []):
                    st.write(f"- {source}")

                if message.get("judge_result"):
                    st.divider()
                    st.markdown("**LLM-as-a-Judge evaluation**")
                    st.text(message["judge_result"])

question = st.chat_input(
    "Ask a DSA question...",
    max_chars=500,
)

if question:
    judge_result = None
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        guardrail_message = validate_question(question)

        if guardrail_message:
            answer = guardrail_message
            sources = []
            judge_result = None

        else:
            with st.spinner("Searching the DSA guide..."):
                answer, scored_documents = answer_question(
                    vector_store,
                    reranker,
                    llm,
                    question,
                )

            sources = make_source_lines(scored_documents)

            judge_result = judge_answer(
                llm,
                question,
                answer,
                scored_documents,
            )

        st.markdown(answer)

        if sources or judge_result:
            with st.expander(DETAILS_LABEL):
                print("Sources: ")
                for source in sources:
                    st.write(f"- {source}")
                
                if judge_result:
                    st.markdown("**LLM-as-a-Judge evaluation**")
                    st.text(judge_result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "judge_result": judge_result,
        }
    )