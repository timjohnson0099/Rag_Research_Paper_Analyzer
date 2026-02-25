import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import JinaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import os
import tempfile
import uuid
import pandas as pd
import re
import fitz
import base64

def highlight_paragraph_from_chunk(pdf_bytes, chunks, num_initial_words=5):
    """
    Highlights paragraphs in a PDF based on the initial words of each chunk.

    Args:
        pdf_bytes (bytes): PDF file as bytes.
        chunks (list of dict): List of chunks to highlight, each chunk should include:
                               - 'text': Text to highlight.
                               - 'page': Page number where the text appears.
        num_initial_words (int): Number of initial words to use for locating the paragraph.

    Returns:
        fitz.Document: The highlighted PDF document
    """
    # Open the PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Iterate over each chunk to highlight
    for chunk in chunks:
        try:
            full_text = chunk['text']
            page_num = chunk['page']
            
            # Take the initial few words from the chunk for matching
            initial_words = ' '.join(full_text.split()[:num_initial_words])
            
            # Access the specified page
            page = pdf_document[page_num]
            
            # Search for the initial words in the page
            text_instances = page.search_for(initial_words)
            
            # If a match is found, expand the highlight to include the paragraph
            for inst in text_instances:
                # Use the rectangle containing the matched initial words
                paragraph_rects = [inst]
                
                # Expand vertically to approximate the full paragraph
                # Get all text blocks on the page and find the one containing our initial match
                for block in page.get_text("blocks"):
                    block_rect, block_text = block[:4], block[4]
                    if inst.intersects(fitz.Rect(block_rect)):
                        # Add a highlight around the block that contains the initial match
                        paragraph_rects.append(fitz.Rect(block_rect))
                
                # Add highlight annotations for each rectangle in the paragraph
                for rect in paragraph_rects:
                    highlight = page.add_highlight_annot(rect)
                    highlight.update()  # Ensure the highlight is properly applied
                    
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue
    
    return pdf_document
    # # Save the modified PDF with highlights
    # pdf_document.save(output_path)
    # pdf_document.close()
    # print(f"Highlighted PDF saved to {output_path}")

def highlight_and_display_pdf(pdf_bytes, chunks):
    try:
        # Apply highlighting
        highlighted_pdf = highlight_paragraph_from_chunk(pdf_bytes, chunks)
        
        # Get the highlighted PDF as bytes
        output_bytes = highlighted_pdf.write()
        
        # Convert to base64 for display
        base64_pdf = base64.b64encode(output_bytes).decode('utf-8')
        
        # Create an iframe to display the PDF
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        
        # Display the PDF in the Streamlit app
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Close the PDF document
        highlighted_pdf.close()
        
    except Exception as e:
        st.error(f"Error highlighting PDF: {e}")
def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename

def get_pdf_text(uploaded_file): 
    """
    Load a PDF document from an uploaded file and return it as a list of documents.

    Parameters:
        uploaded_file (file-like object): The uploaded PDF file to load.

    Returns:
        list: A list of documents created from the uploaded PDF file, or an empty list if parsing fails.
    """
    try:
        # Read file content
        input_file = uploaded_file.read()

        # Create a temporary file to allow PyPDFLoader to read the PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()
        # Attempt to load PDF document
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        return documents

    except Exception as e:
        # Handle and log the error
        st.error(f"Failed to load the PDF document. Error: {e}")
        return []

    finally:
        # Ensure the temporary file is deleted
        os.unlink(temp_file.name)


def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks.
    chunk_size: The desired maximum size of each chunk (default: 400)
    chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

    Returns:
        list: A list of smaller text chunks created from the generic text
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       length_function=len,
                                                       separators=["\n\n", "\n", " "])
        
        # Attempt to split the documents
        return text_splitter.split_documents(documents)
    
    except Exception as e:
        # If any error occurs, print the error message and return an empty list
        print(f"An error occurred while splitting documents: {e}")
        return []


def get_embedding_function(api_key):
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
    as an argument to the function.

    Parameters:
        api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    try:
        # Initialize the JinaEmbeddings object with the provided API key
        embeddings = JinaEmbeddings(
        jina_api_key= api_key,
        model_name="jina-embeddings-v3",
        )
        return embeddings
    
    except Exception as e:
        # If any error occurs during the initialization, print the error message
        print(f"An error occurred while initializing the embeddings function: {e}")
        return None


def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    """
    Create a vector store from a list of text chunks.

    :param chunks: A list of generic text chunks
    :param embedding_function: A function that takes a string and returns a vector
    :param file_name: The name of the file to associate with the vector store
    :param vector_store_path: The directory to store the vector store

    :return: A Chroma vector store object
    """
    try:
        # Create a list of unique ids for each document based on the content
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
        
        # Ensure that only unique docs with unique ids are kept
        unique_ids = set()
        unique_chunks = []
        for chunk, id in zip(chunks, ids):     
            if id not in unique_ids:       
                unique_ids.add(id)
                unique_chunks.append(chunk)        
        
        # Create a new Chroma database from the documents
        vectorstore = Chroma.from_documents(
            documents=unique_chunks, 
            collection_name=clean_filename(file_name),
            embedding=embedding_function, 
            ids=list(unique_ids), 
            persist_directory=vector_store_path
        )

        return vectorstore
    
    except Exception as e:
        # If any error occurs during the vector store creation, print the error message
        print(f"An error occurred while creating the vector store: {e}")
        return None


def create_vectorstore_from_texts(documents, api_key, file_name):
    
    # Step 2 split the documents  
    """
    Create a vector store from a list of texts.

    :param documents: A list of generic text documents
    :param api_key: The OpenAI API key used to create the vector store
    :param file_name: The name of the file to associate with the vector store

    :return: A Chroma vector store object
    """
    docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3 define embedding function
    embedding_function = get_embedding_function(api_key)

    # Step 4 create a vector store  
    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
    return vectorstore


def load_vectorstore(file_name, api_key, vectorstore_path="db"):
    """
    Load a previously saved Chroma vector store from disk.

    Parameters:
        file_name (str): The name of the file to load (without the path).
        api_key (str): The OpenAI API key used to create the vector store.
        vectorstore_path (str): The path to the directory where the vector store was saved (default: "db").
    
    Returns:
        Chroma: A Chroma vector store object if loaded successfully, or None if an error occurred.
    """
    try:
        # Get the embedding function using the provided API key
        embedding_function = get_embedding_function(api_key)
        
        # Attempt to load the vector store from the specified directory
        vectorstore = Chroma(
            persist_directory=vectorstore_path, 
            embedding_function=embedding_function, 
            collection_name=clean_filename(file_name)
        )
        
        return vectorstore

    except Exception as e:
        # Handle and log the error
        print(f"Failed to load vector store: {e}")
        return None

# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to the user question.")
    sources: str = Field(description="Direct text or references from the context used to answer the question. heading of the paragraph if any")
    reasoning: str = Field(description="Explanation of the reasoning for the answer based on the sources.")

def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects

    :return: A string containing the text of all the documents joined by two newlines
    """
    try:
        # Join the page content of each document with two newlines
        return "\n\n".join(doc.page_content for doc in docs)
    
    except AttributeError as e:
        # Handle the case where a doc does not have the 'page_content' attribute
        print(f"An error occurred: One of the documents does not have 'page_content'. Error: {e}")
        return ""
    
    except TypeError as e:
        # Handle the case where docs is not a list or is improperly structured
        print(f"An error occurred: Invalid input for docs. Error: {e}")
        return ""
    
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred while formatting the documents: {e}")
        return ""

# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings;
# RunnablePassthrough() passes through the input question unchanged.
def query_document(vectorstore, query, api_key):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.2-90b-vision-preview"
    )
    retriever = vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    retrieved_docs = retriever.invoke(query)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(AnswerWithSources)
    )

    try:
        # Invoke the RAG chain with the query
        structured_response = rag_chain.invoke(query)

        # Access the fields directly, assuming structured_response is an AnswerWithSources object
        response = {
            "answer": structured_response.answer,
            "source": structured_response.sources,
            "reasoning": structured_response.reasoning,
            "chunk_text": [{'text': doc.page_content, 'page':doc.metadata['page']} for doc in retrieved_docs],
            "page":retrieved_docs[0].metadata['page']
        }

        # Convert the response to a DataFrame and return it
        response_df = pd.DataFrame([response])

        return response_df

    except Exception as e:
        # Handle error
        st.error(f"An error occurred while processing the query: {e}")
        error_response = {
            "answer": "An error occurred while generating the answer.",
            "source": [],
            "reasoning": "",
            "page": "",
            "chunk_text": []
        }
        
        # Return error response as DataFrame
        error_df = pd.DataFrame([error_response])

        return error_df
