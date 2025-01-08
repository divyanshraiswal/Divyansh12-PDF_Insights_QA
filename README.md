## PDF_Insights_QA

This project is a Retrieval-Augmented Generation (RAG) system designed to answer user queries by retrieving relevant context from PDFs and generating responses using an LLM. It uses LangChain, Pinecone for vector storage, and Groq for model inference.


# Setup Instructions

## 1. Clone the Repository

``` bash
git clone https://github.com/divyanshraiswal/Divyansh12-PDF_Insights_QA.git
cd Divyansh12-PDF_Insights_QA
```
## 2. Environment Variables

Create a .env file in the project root and add the following keys:

``` bash
GROQ_API_KEY=<your_groq_api_key>
HUGGINGFACE_API_KEY=<your_huggingface_api_key>
PINECONE_API_KEY=<your_pinecone_api_key>
```

## 3. Install the necessary libraries
``` bash
pip install -r requirements.txt
```

## 4. Run the app using 
``` bash
python app.py
```


# the project is hosted on hugging face
``` bash
https://huggingface.co/spaces/Divyansh12/PDF_Insights_QA
```

# Screenshot of the file is:
<img width="959" alt="image" src="https://github.com/user-attachments/assets/6d7b9cb8-5fa5-4c21-b3aa-2369df3d0c82" />
