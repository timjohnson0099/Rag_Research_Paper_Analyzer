import streamlit as st
from functions import *
import base64
from dotenv import load_dotenv
import os
import fitz
load_dotenv()

# Initialize session states
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

def display_pdf(pdf_bytes, container):
    try:
        # Convert the bytes to base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Create an iframe to display the PDF
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        
        # Clear the container and display new PDF
        container.empty()
        container.markdown(pdf_display, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def load_streamlit_page():
    try:
        st.set_page_config(layout="wide", page_title="LLM Tool")
        
        # Create two columns with different widths
        col1, col2 = st.columns([0.5, 0.5], gap="large")

        with col1:
            st.header("Input your OpenAI API key")
            api_key = st.text_input('OpenAI API key', type='password', key='api_key', label_visibility="collapsed")
            
            if not api_key:
                st.warning("Please enter your OpenAI API key.")
            
            st.header("Upload file")
            uploaded_file = st.file_uploader("Please upload your PDF document:", type="pdf")
            
            if uploaded_file is not None:
                if uploaded_file.type != "application/pdf":
                    st.error("Uploaded file is not a PDF. Please upload a valid PDF document.")
                    uploaded_file = None

        return col1, col2, uploaded_file

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None

# Make a Streamlit page
col1, col2, uploaded_file = load_streamlit_page()

# Create a container for the PDF in col2
with col2:
    st.header("Document Viewer")
    # Create an empty container for the PDF
    pdf_container = st.empty()

# Handle file upload and display
if uploaded_file is not None:
    # Store the original PDF bytes in session state
    st.session_state.current_pdf = uploaded_file.getvalue()
    # Display the original PDF
    display_pdf(st.session_state.current_pdf, pdf_container)
    
    documents = get_pdf_text(uploaded_file)
    embd_api_key = st.session_state.api_key or os.getenv("JINA_API_KEY")
    st.session_state.vector_store = create_vectorstore_from_texts(documents, api_key=embd_api_key, file_name=uploaded_file.name)
    
    # Move success message to col1
    with col1:
        st.success("Document processed successfully.")

# Input for user prompt in col1
with col1:
    user_query = st.text_input("Enter your query:")

    if st.button("Generate Answer"):
        if uploaded_file is None:
            st.error("Please upload a PDF document first.")
        else:
            with st.spinner("Generating answer..."):
                llm_api_key = st.session_state.api_key or os.getenv("GROQ_API_KEY")
                answer_df = query_document(vectorstore=st.session_state.vector_store, query=user_query, api_key=llm_api_key)
                
                if not answer_df.empty:
                    # Display the results
                    st.subheader("Answer")
                    st.write(answer_df['answer'][0])
                    
                    st.subheader("Most Relvant Page")
                    st.write(answer_df['page'][0]+1)

                    st.subheader("Sources")
                    st.write(answer_df['source'][0])

                    st.subheader("Reasoning")
                    st.write(answer_df['reasoning'][0])

                    try:
                        # Apply highlighting
                        highlighted_pdf = highlight_paragraph_from_chunk(st.session_state.current_pdf, answer_df['chunk_text'][0])
                        
                        # Get the highlighted PDF as bytes
                        output_bytes = highlighted_pdf.write()
                        
                        # Update the display
                        display_pdf(output_bytes, pdf_container)
                        
                        # Close the PDF document
                        highlighted_pdf.close()
                        
                    except Exception as e:
                        st.error(f"Error highlighting PDF: {e}")
                        # Fallback to original PDF if highlighting fails
                        display_pdf(st.session_state.current_pdf, pdf_container)

    # Add a button to reset highlights
    if st.button("Reset Highlights"):
        if st.session_state.current_pdf is not None:
            display_pdf(st.session_state.current_pdf, pdf_container)

            
