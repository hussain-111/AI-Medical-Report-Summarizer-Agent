# 🩺 AI Medical Report Summarizer

A Streamlit-based web app that uses LangChain and OpenAI to analyze full-body medical checkup reports and generate concise, medically relevant summaries. Built for hackathons and healthcare innovation.

---

## 🚀 Features

- Upload `.txt` medical reports  
- Extract patient history (if present)  
- Highlight abnormal findings and **bold** abnormal values  
- Suggest likely diagnosis and treatment plan  
- Generate a short, structured summary  
- View original report and download the summary

---

## 🧠 Tech Stack

| Layer       | Technology Used                          |
|-------------|-------------------------------------------|
| Frontend    | Streamlit                                |
| AI Model    | LangChain + OpenAI (DeepSeek-V3)         |
| Backend     | Python + HTTPX                           |
| Hosting     | GenAI Lab (TCS Azure AI)                 |

---

## 🛠️ How It Works

1. User uploads a `.txt` medical report  
2. App sends a structured prompt to the AI model via LangChain  
3. Model returns a formatted summary with key findings  
4. Summary is displayed in the UI and can be downloaded

---

## 📦 Installation

```bash
git clone https://github.com/your-username/medical-report-summarizer.git
cd medical-report-summarizer
pip install -r requirements.txt
streamlit run app.py

MODEL UI
<img width="869" height="511" alt="image" src="https://github.com/user-attachments/assets/10c6b447-374c-459b-8e1a-bdc6477112f1" />

