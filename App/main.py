import streamlit as st
import tempfile
from Field_Extraction.Fields_extraction import extract_fields
from indexing.text_loading_and_preparation import document_loading, creating_and_storing_embeddings
from output_generators.generators import generate_json, generate_pdf
from langchain.schema import HumanMessage, AIMessage

# Page config
st.set_page_config(page_title="Transcript Parser", layout="wide")
st.title("Customer Support Transcript Parser")

# Input method
st.markdown("### Choose how you want to provide the transcript")
mode = st.radio("Input Method", ("Upload Text File", "Paste Text"))

transcript_text = ""

if mode == "Upload Text File":
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded_file is not None:
        transcript_text = uploaded_file.read().decode("utf-8")

elif mode == "Paste Text":
    transcript_text = st.text_area("Paste the transcript text here", height=250)

# Process button
if transcript_text:
    if st.button("Process Transcript"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as temp_file:
            temp_file.write(transcript_text)
            temp_path = temp_file.name

        # Step 1: Create FAISS vectorstore
        creating_and_storing_embeddings(temp_path)

        # Step 2: Field extraction
        transcript = document_loading(temp_path)
        chain = extract_fields()
        res = chain.invoke({'transcript': transcript})
        st.write("Extracted transcript for parsing.")

        # Step 3: Output generation
        json_output = generate_json(res)
        generate_pdf(res)

        # Cache for reuse
        st.session_state["json_output"] = json_output
        st.session_state["pdf_path"] = "outputs/pdf_summary.pdf"
        st.session_state["chat_enabled"] = True
        st.session_state["messages"] = []  
        st.session_state["chat_history"] = []  

        st.success("Transcript processed successfully!")

        # Display extracted fields
        st.subheader("Extracted Fields")
        st.json(json_output)

# Downloads
if "json_output" in st.session_state and "pdf_path" in st.session_state:
    st.markdown("### Download Results")
    st.download_button(
        "Download JSON",
        data=str(st.session_state["json_output"]),
        file_name="json_format.json",
        mime="application/json"
    )

    with open(st.session_state["pdf_path"], "rb") as f:
        st.download_button(
            "Download PDF",
            data=f,
            file_name="pdf_summary.pdf",
            mime="application/pdf"
        )

# Chat interface
if st.session_state.get("chat_enabled", False):
    st.markdown("---")
    st.header("Ask Questions about the Transcript")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)

    user_query = st.chat_input("Ask something about the transcript")
    if user_query:
        # Import chat function
        from Retrieval.retrieving_similar_embeddings import chat_with_transcript

        # Show user input
        user_msg = HumanMessage(content=user_query)
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get Gemini answer
        response = chat_with_transcript(user_query)

        # Display response
        ai_msg = AIMessage(content=response)
        st.session_state.messages.append(ai_msg)
        with st.chat_message("assistant"):
            st.markdown(response)
