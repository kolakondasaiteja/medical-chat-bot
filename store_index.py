from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = "6e20a326-c875-477f-8059-748ee0facf27"
PINECONE_API_ENV = "gcp-starter"

print(PINECONE_API_KEY)
print(PINECONE_API_ENV)

extracted_data = load_pdf(r"C:\Users\SAI TEJA\OneDrive\Desktop\Generative AI\Medical chatbot using llama2\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)


index_name="medical-chat-bot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
