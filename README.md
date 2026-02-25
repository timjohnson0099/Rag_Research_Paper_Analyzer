# RAG-based Research Paper Analyzer

A containerized Streamlit app that analyzes research papers and generates answers to queries using vector embeddings and a Retrieval-Augmented Generation (RAG) model.

## Key Features

- **PDF Upload & Processing**: Easily upload PDF research papers for analysis.
- **Document Chunking**: Splits documents into manageable chunks and stores them in a Chroma vector database.
- **Question Answering**: Allows users to enter questions related to the paper's content.
- **Contextual Retrieval**: Retrieves relevant document chunks using vector-based search.
- **RAG-based Answer Generation**: Generates accurate answers based on retrieved content using a RAG model.
- **Passage Highlighting**: Highlights relevant passages directly within the PDF document.
- **Detailed Responses**: Displays the answer, source passages, and reasoning for each query.

## Usage

1. **Pull the Docker image** from Docker Hub:
   ```bash
   docker pull salmanaliajaz/rag_streamlit_app_research_paper_analyzer

2. **Run the Docker container**:
   Start the container and expose it on port 8501.
   ```bash
   docker run -p 8501:8501 salmanaliajaz/rag_streamlit_app_research_paper_analyzer

3. **Access the Streamlit app**:
   Once the container is running, open your web browser and navigate to [http://localhost:8501](http://localhost:8501) to access the app interface.

4. **Using the App**:
   - **Upload a PDF**: Click the "Upload" button to select and upload a research paper in PDF format.
   - **Enter Queries**: After the PDF is uploaded, type a question related to the content of the paper in the provided query box.
   - **Get Answers**: The app will retrieve relevant chunks from the document using vector search, and the RAG model will generate a contextually informed answer.
   - **View Highlighted Passages**: The app highlights relevant passages in the PDF document for easy reference.
   - **Review Detailed Results**: Each query provides an answer, source passages, and a reasoning summary, allowing for transparent and thorough insights into the document content.

5. **Interactive Exploration**:
   - You can ask multiple questions in succession, with the app retrieving and highlighting new relevant passages for each query.
   - Explore different sections of the paper by adjusting your questions to cover various aspects of the research content.

## Technical Details

- **Core Technologies**:
  - **Python**, **Streamlit**, and **LangChain** power the app's interface and logic.
- **Vector Storage**:
  - Uses **Chroma** vector database to efficiently store and retrieve document chunks based on embeddings.
- **Answer Generation**:
  - Employs a **RAG (Retrieval-Augmented Generation)** model for generating accurate, context-driven answers based on the retrieved content.
- **PDF Annotation**:
  - Highlights relevant text passages within the PDF using **PyMuPDF** for an intuitive review of the document.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to suggest features, fix bugs, or improve the performance and functionality of the app.

---




