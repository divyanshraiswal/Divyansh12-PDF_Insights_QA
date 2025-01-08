import os
import nest_asyncio
import time
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import gradio as gr 
from langchain import hub

# Allow nested async calls
nest_asyncio.apply()

# Load environment variables
_ = load_dotenv(find_dotenv())
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone()
index_name = "intern"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

#  embeddings model
print("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and split documents
print("Loading documents...")
loader = PyPDFDirectoryLoader("Data")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

def are_documents_indexed(index):
    try:
        # Create a simple test embedding
        test_embedding = embedding_model.embed_query("test")
        # Query the index
        results = index.query(vector=test_embedding, top_k=1)
        return len(results.matches) > 0
    except Exception as e:
        print(f"Error checking indexed documents: {e}")
        return False

# Initialize vector store
print("Initializing vector store...")
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

# Add documents only if not already indexed
print("Checking if documents are already indexed...")
if not are_documents_indexed(index):
    print("Adding documents to index...")
    vector_store.add_documents(docs)
    print("Documents added successfully!")
else:
    print("Documents are already indexed.")

print("Setting up retriever and LLM...")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, max_retries=4)
str_output_parser = StrOutputParser()

prompt = hub.pull("jclemens24/rag-prompt")

relevance_prompt_template = PromptTemplate.from_template(
    """
    Given the following question and retrieved context, determine if the context is relevant to the question.
    Provide a score from 1 to 5, where 1 is not at all relevant and 5 is highly relevant.
    Return ONLY the numeric score, without any additional text or explanation.
    Question: {question}
    Retrieved Context: {retrieved_context}
    Relevance Score: """
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_score(llm_output):
    try:
        return float(llm_output.strip())
    except ValueError:
        return 0

def conditional_answer(x):
    relevance_score = extract_score(x["relevance_score"])
    return "I don't know." if relevance_score < 4 else x["answer"]

# RAG pipeline
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | RunnableParallel(
        {
            "relevance_score": (
                RunnablePassthrough()
                | (lambda x: relevance_prompt_template.format(question=x["question"], retrieved_context=x["context"]))
                | llm
                | str_output_parser
            ),
            "answer": (
                RunnablePassthrough()
                | prompt
                | llm
                | str_output_parser
            ),
        }
    )
    | RunnablePassthrough().assign(final_answer=conditional_answer)
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever,
     "question": RunnablePassthrough()
    }
).assign(answer=rag_chain_from_docs)

async def process_question(question):
    try:
        result = await rag_chain_with_source.ainvoke(question)
        final_answer = result["answer"]["final_answer"]
        sources = [doc.metadata.get("source") for doc in result["context"]]
        source_list = ", ".join(sources)
        return final_answer, source_list
    except Exception as e:
        return f"Error: {str(e)}", "Error retrieving sources"

# Gradio 
print("Gradio interface...")
demo = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(label="Enter your question", value=""),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources"),
    ],
    title="RAG Question Answering",
    description="Enter a question and get an answer from the PDFs.",
)

if __name__ == "__main__":
    print("Launching the application...")
    demo.queue().launch()