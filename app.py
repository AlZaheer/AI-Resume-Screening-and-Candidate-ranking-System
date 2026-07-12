import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import os

# Fetch the key securely from Streamlit secrets
API_KEY = st.secrets["GEMINI_API_KEY"] 

# Initialize the new Google GenAI Client
client = genai.Client(api_key=API_KEY)

# Function to extract text from pdf
def extract_text_from_pdf(file):
    pdf = PdfReader(file) 
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Function to rank resumes based on job description
'''def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities'''

def rank_resumes(job_description, resumes):
    documents = []

    # Clean Job Description
    jd = str(job_description).replace("\x00", " ").strip()
    documents.append(jd)

    # Clean each resume
    for i, resume in enumerate(resumes):
        resume = str(resume)

        st.write(f"Resume {i+1}")
        st.write(f"Length: {len(resume)}")

        # Remove NULL characters
        resume = resume.replace("\x00", " ")

        # Remove non-printable characters
        resume = "".join(ch for ch in resume if ch.isprintable() or ch.isspace())

        st.write(f"Cleaned Length: {len(resume)}")

        documents.append(resume)

    st.write("Starting TF-IDF")

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(documents)

    st.write("TF-IDF completed")

    scores = cosine_similarity(vectors[0], vectors[1:]).flatten()

    return scores

# Function to generate a reason using the new GenAI SDK syntax
def generate_reason(job_description, resume_text):
    prompt = f"""
    You are an expert technical recruiter. 
    Job Description: {job_description}
    
    Resume Snippet: {resume_text[:3000]} 
    
    In one concise sentence, explain exactly why this candidate stands out or is a good fit for this role based on their resume.
    """
    try:
        # Call the content generation using the modern 2.5-flash model
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        # Print the full error to your running terminal for easier debugging
        print(f"Gemini API Error: {e}") 
        # Display a truncated version of the error message inside the table
        return f"Error: {str(e)[:50]}..."

# Streamlit app
st.title("AI Resume Screening and Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload resumes")
uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# Custom Top N Filter
st.header("Ranking Settings")
top_n = st.number_input("Number of top candidates to display", min_value=1, max_value=100, value=10)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    resume_names = []
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        resume_names.append(file.name)

    # Rank resumes using TF-IDF
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": resume_names, "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    # Filter down to Top N candidates
    top_results = results.head(top_n).copy()

    # Button to generate AI reasoning for the top candidates
    if st.button(f"Generate AI Reasoning for Top {len(top_results)} Candidates"):
        with st.spinner("Analyzing top resumes with AI..."):
            reasons = []
            for index, row in top_results.iterrows():
                # Find the original text for the matching resume
                resume_idx = resume_names.index(row["Resume"])
                original_text = resumes[resume_idx]
                
                # Generate the contextual breakdown reason
                reason = generate_reason(job_description, original_text)
                reasons.append(reason)
            
            # Add the reasoning column to our filtered table
            top_results["Reason"] = reasons
            
        st.success("Analysis Complete!")
        
        # 1. Show the clean table (without forcing the long reason column inside a tiny cell)
        st.dataframe(top_results[["Resume", "Score"]], use_container_width=True)
        
        # 2. Show beautiful, fully visible AI Insight Cards right underneath
        st.subheader("💡 Deep AI Insights")
        for index, row in top_results.iterrows():
            with st.expander(f"📄 {row['Resume']} (Score: {row['Score']:.4f})", expanded=True):
                st.write(row["Reason"])
