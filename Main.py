import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import httpx
import time
import zipfile
import io
from pypdf import PdfReader
import magic # Requires pip install python-magic

# --- Configuration ---
client = httpx.Client(verify=False) 

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-6Mxu9roYUgglLHoJUtl-Xw",
    http_client=client
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "medical_summary" not in st.session_state:
    st.session_state["medical_summary"] = None

# === Core Logic: File Processing Functions ===
def extract_text_from_pdf(file_object):
    """Extracts text from a PDF file object."""
    try:
        reader = PdfReader(file_object)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return None

def extract_text_from_zip(file_object):
    """Extracts text from all files inside a ZIP archive."""
    combined_text = ""
    try:
        with zipfile.ZipFile(file_object, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                # Only process text-like files inside the zip
                if file_name.endswith('.txt') or file_name.endswith('.pdf'):
                    with zip_ref.open(file_name) as file_in_zip:
                        file_type = magic.from_buffer(file_in_zip.read(2048), mime=True)
                        file_in_zip.seek(0) # Reset buffer after type check
                        
                        if 'text' in file_type:
                            combined_text += f"\n--- Start of {file_name} (from ZIP) ---\n"
                            combined_text += file_in_zip.read().decode("utf-8")
                            combined_text += f"\n--- End of {file_name} ---\n"
                        elif 'pdf' in file_type:
                            combined_text += f"\n--- Start of {file_name} (from ZIP) ---\n"
                            combined_text += extract_text_from_pdf(io.BytesIO(file_in_zip.read()))
                            combined_text += f"\n--- End of {file_name} ---\n"
        return combined_text
    except Exception as e:
        st.error(f"Error processing ZIP file: {e}")
        return None

# === Core Logic: Generate Medical Summary ===
def generate_report(text_input):
    """Generates the structured medical summary using the LLM."""
    prompt = """
You are a medical AI assistant. You will receive medical text extracted from one or more full-body checkup reports.
Your tasks:
1. Extract patient history with name (if present).
2. Identify key abnormal findings from all reports and highlight abnormal values using **bold**.
3. Based on the findings, provide:
    - Likely diagnosis
    - Suggested treatment plan
    - Medicines
4. Generate a short summary of the case.
Only include medically relevant data. Format the output clearly with headings.
    """
    full_prompt = f"{prompt}\n\n{text_input}"
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing report(s)..."):
            response = llm.invoke([HumanMessage(content=full_prompt)])
            summary = response.content.strip()
            
            message_placeholder = st.empty()
            full_response = ""
            for chunk in summary.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
            message_placeholder.markdown(full_response)
            
            st.session_state["medical_summary"] = summary
            st.session_state["messages"] = [
                {"role": "assistant", "content": summary}
            ]

# === Core Logic: Chat Interaction ===
def handle_chat_input(prompt_input):
    """Handles the user's follow-up questions in the chat window."""
    st.session_state["messages"].append({"role": "user", "content": prompt_input})
    
    initial_summary = st.session_state["medical_summary"]
    
    chat_prompt = f"""
    Based on the following medical report summary, answer the user's question. 
    Maintain a helpful, medically-aware, and cautious tone. If the information 
    is not in the summary, state that fact and remind them to consult a doctor.
    
    --- MEDICAL SUMMARY CONTEXT ---
    {initial_summary}
    --- USER QUESTION ---
    {prompt_input}
    """
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke([HumanMessage(content=chat_prompt)])
            assistant_response = response.content.strip()
            
            message_placeholder = st.empty()
            full_response = ""
            for chunk in assistant_response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)
    
    st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

# ==================================
# === Streamlit UI ===
# ==================================
st.set_page_config(page_title="Automated Medical Report Summarizer (Chat)", layout="wide")
st.title("🩺 Automated Medical Report Summarizer")

# --------------------------
# 1. LEFT: File Upload (Sidebar)
# --------------------------
with st.sidebar:
    st.header("Upload Report(s)")
    # accept_multiple_files=True allows multiple files
    # The 'type' parameter is now more generic
    uploaded_files = st.file_uploader("Upload one or more reports (.txt, .pdf, .zip)", 
                                      type=["txt", "pdf", "zip"], 
                                      accept_multiple_files=True)

    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")
        
        combined_text = ""
        with st.expander("📄 View Original Reports"):
            for file in uploaded_files:
                file.seek(0)
                file_type = magic.from_buffer(file.read(2048), mime=True)
                file.seek(0)
                
                if 'text/plain' in file_type:
                    raw_text = file.read().decode("utf-8")
                    combined_text += f"\n\n--- Start of {file.name} ---\n{raw_text}\n--- End of {file.name} ---"
                    st.subheader(f"Content of {file.name}")
                    st.text_area(f"Report Content - {file.name}", raw_text, height=150)
                
                elif 'application/pdf' in file_type:
                    pdf_text = extract_text_from_pdf(file)
                    if pdf_text:
                        combined_text += f"\n\n--- Start of {file.name} ---\n{pdf_text}\n--- End of {file.name} ---"
                        st.subheader(f"Content of {file.name}")
                        st.text_area(f"Report Content - {file.name}", pdf_text, height=150)
                
                elif 'application/zip' in file_type:
                    zip_text = extract_text_from_zip(io.BytesIO(file.read()))
                    if zip_text:
                        combined_text += zip_text
                        st.subheader(f"Content of {file.name}")
                        st.text_area(f"Report Content - {file.name}", "Text extracted from ZIP file(s).", height=150)
                
                else:
                    st.warning(f"Skipped unsupported file type: {file.name}")

        if st.button("Generate Summary & Start Chat", use_container_width=True):
            if combined_text:
                st.session_state["messages"] = []
                st.session_state["medical_summary"] = None
                generate_report(combined_text)
            else:
                st.error("No extractable text found in the uploaded files.")

# --------------------------
# 2. MAIN: Summary and Chat Window
# --------------------------
if st.session_state["medical_summary"] or (uploaded_files and "combined_text" in locals() and combined_text):
    st.header("AI Summary & Interactive Q&A")
    st.markdown("The **AI-Generated Summary** appears as the first message below. It has been created by analyzing all uploaded reports. You can now ask follow-up questions about the findings or treatments.")
    
    if st.session_state["medical_summary"]:
        st.download_button(
            label="📥 Download Summary Report",
            data=st.session_state["medical_summary"],
            file_name="medical_summary.txt",
            mime="text/plain",
            key="download_summary_button"
        )
        st.markdown("---")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state["medical_summary"]:
        if prompt := st.chat_input("Ask a follow-up question about the summary..."):
            handle_chat_input(prompt)
else:
    st.info("Please upload one or more medical reports (.txt, .pdf, or .zip) in the sidebar to generate a summary and begin the interactive Q&A session.")