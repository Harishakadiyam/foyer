
# Insurance Policy RAG Assistant

## About the Project

I built this project to understand how RAG (Retrieval-Augmented Generation) can be used to get answers from insurance policy documents.

The idea is simple: instead of reading a complete PDF to find a specific policy detail, a user can upload the PDF, ask a question, and get an answer based on the document.

I am using a sample home insurance policy to test the application.

## What I Have Implemented

- Upload insurance policy PDFs using Streamlit.
- Extract text from PDFs using `pdfplumber`.
- Extract tables and convert the rows into readable text.
- Combine the extracted text and table content.
- Keep the PDF filename and page number for reference.
- Split the document into smaller chunks.
- Generate embeddings using Hugging Face sentence transformers.
- Store and search the embeddings using FAISS.
- Use Google Gemini to generate answers from the retrieved content.
- Display the answer along with the source document and page number.

## Tools and Technologies

- Python
- Streamlit
- LangChain
- pdfplumber
- Hugging Face Sentence Transformers
- FAISS
- Google Gemini API

## How the Application Works

1. Upload an insurance policy PDF.
2. Extract the text and tables from the PDF.
3. Clean the extracted content and keep the page information.
4. Split the content into chunks.
5. Generate embeddings and store them in FAISS.
6. Enter a question about the policy.
7. Retrieve relevant content and send it to Gemini to generate an answer.
8. Display the answer with its source and page reference.

## How to Run the Project

### 1. Install the required packages

```bash
python -m pip install -r requirements.txt
```

### 2. Configure the Gemini API key

Create a `.env` file in the project folder and add your API key:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your actual key. Do not upload the `.env` file to GitHub.

### 3. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Sample Test

**Question:** What is the fire damage coverage?

The application retrieves relevant information from the uploaded policy and generates an answer with a source filename and page number.

I am using a synthetic insurance policy for testing. The policy details are fictional and are not intended to represent an actual insurance product.

## Next Steps

I plan to improve the project further by:

- Testing how accurately the application retrieves information from tables.
- Comparing different chunk sizes and overlap settings.
- Testing more questions and checking the answers against the original PDF.
- Improving the handling of complex tables and scanned documents.
- Checking that answers and citations are supported by the retrieved content.

## Author

Harisha Kommoju
