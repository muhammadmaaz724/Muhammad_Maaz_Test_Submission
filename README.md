# Transcript Parser - Simplii Test Submission

This project implements a customer support transcript parser using **LangChain** and **Gemini (Google Generative AI)**. It extracts structured information from customer support transcripts, stores vectorized representations for retrieval, and provides a Streamlit-based user interface for parsing, downloading results, and querying the transcript.

---

## Features

- Field extraction using Gemini + LangChain + Pydantic schema  
- Semantic search over transcripts using FAISS vectorstore  
- Streamlit UI for:
  - Uploading or pasting transcripts  
  - Viewing extracted data  
  - Downloading structured outputs (JSON, PDF)  
  - Chat-based querying of transcript content  

---

## Folder Structure

Muhammad_Maaz_Test_Submission/

├── App/

│ ├── Field_Extraction/ # Pydantic-based field extraction logic

│ ├── indexing/ # FAISS vectorstore creation and document splitting

│ ├── output_generators/ # JSON and PDF generation

│ ├── Retrieval/ # Semantic search and Gemini-based chat

│ ├── vectorstore/ # Saved FAISS index files

├── input/

│ └── transcript.txt # Sample input file

├── outputs/

│ ├── json_format.json # Generated structured output (JSON)

│ └── pdf_summary.pdf # Generated summary file (PDF)

├── main.py # Streamlit application entry point

├── requirements.txt # Required Python dependencies

└── .env # API keys and environment variables

---

## Setup Instructions

1. **Install dependencies:**
   python -m venv venv
   
   venv/Scripts/activate   # For Windows
   
   pip install -r requirements.txt
   
3. **Set up environment variables:**
   Create a .env file in the root directory and add:

   GOOGLE_API_KEY=your_google_api_key

5. **Run the application:**
streamlit run App/main.py

Notes:
- Uses Gemini gemini-2.5-flash for field extraction and transcript Q&A
  
- Embedding model used: models/embedding-001
  
- Outputs are stored in the outputs/ folder
  
- FAISS index is saved in App/vectorstore/faiss_index
