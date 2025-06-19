from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import shutil
from langchain_community.llms import HuggingFaceHub
shutil.rmtree("./docs/chroma")
import os

loader = NotionDirectoryLoader("./docs/DB")
pages = loader.load()
# print(pages[0].metadata)
# print(len(pages))
# print(pages[50].page_content[:100])


# for notion best is context aware splitting
"""for example
- for a markdown file we are using MarkDownHeaderTextSplitter
- for  a csv file check is there is a context aware text splitter
it lacks a csv splitter either use charactrer text splitting or 
create a custom csv splitter : which divides the file into chuncks suppose 10 rows in
one chunk and then wraps it in a Langchain document """

txt = " ".join([ d.page_content for d in pages])
headers_to_split_on = [
    ('#', 'Header 1'),
    ('##', 'Header 2')
]
markdown_splitter =  MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
md_splits = markdown_splitter.split_text(txt)
#print(md_splits[0])

embedding =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=md_splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# #print(vectordb._collection.count()) # (Number of vectors) size of vector db

question = "  Why were these documents made public"
docs_mmr = vectordb.max_marginal_relevance_search(question, k=3)
#print(docs_mmr[0].page_content[:100])

import openai


import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_openai import ChatOpenAI
llm_name = "gpt-4o"
llm = ChatOpenAI(model = llm_name, temperature =  0)

from langchain.prompts import PromptTemplate

template = """ Use the following piece of text to answer the question
{context}
Question: {input}
Helpful Answer: """
prompt = PromptTemplate.from_template(template)

from langchain.chains import RetrievalQA

# retriever = vectordb.as_retriever()
# docs = retriever.get_relevant_documents(question)
# print("Retrieved Context Documents:\n")
# for i, doc in enumerate(docs):
#     print(f"--- Document {i+1} ---\n{doc.page_content}\n")

# qa_chain =  RetrievalQA.from_llm(llm, retriever = vectordb.as_retriever(), return_source_documents=True, prompt = prompt , verbose = True)

# print(question)
# result = qa_chain.invoke({"query":question})
# print(result['result'])


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
retrieval_qa_chat_prompt = PromptTemplate.from_template(template)

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(vectordb.as_retriever(), combine_docs_chain)

result = rag_chain.invoke({"input": question})

# print(result['input'])
# print()
# print(result['context'])
# print()
# print(result['answer'])




from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
question = "Why is the document public?"
result = qa.invoke({"question": question})
print(result["answer"])
print()
question = "how can it inspire me?"
result = qa.invoke({"question": question})
print(result["answer"])