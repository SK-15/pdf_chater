# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 21:29:41 2023

@author: saura
"""

from chainlit import on_message, on_chat_start
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import os
from langchain.callbacks import ContextCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import pinecone
# from langchain import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

if __name__ == '__main__':
    
    pine_cone_key = None
    openai_api_key = None
    cohere_key = None
    huggin_face_api = None
    
    template = """
    You are a data retriever and your job is answere wheather we have relevant accoring to asked question or not.
    
    {question}
    """
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    
    pinecone.init(
        api_key=pine_cone_key,  # find at app.pinecone.io
        environment='gcp-starter',  # next to api key in console,
    )
    index_name = "fundamental"
    # 
    # openai_llm = ChatOpenAI(temperature=0.7, verbose=True, openai_api_key = Opeai_key, streaming=True)
    
    
    
    huggingfacehub_api_token = huggin_face_api
    repo_id = "tiiuae/falcon-7b-instruct"
    
    falcon_llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                          repo_id=repo_id, 
                          model_kwargs={"temperature":0.6, "max_new_tokens":2000})
    
    memory = ConversationTokenBufferMemory(llm=falcon_llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=1000)
    embeddings = CohereEmbeddings(model='embed-english-light-v2.0',cohere_api_key=cohere_key)
    docsearch = Pinecone.from_existing_index(
    index_name=index_name, embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})
    
    
    
    prompt = PromptTemplate(template=template, input_variables=["question"])
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    # doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=prompt)
    # chain_type_kwargs = {"prompt": prompt}
    qa = ConversationalRetrievalChain.from_llm(llm=falcon_llm, retriever=retriever, memory=memory, condense_question_prompt=CONDENSE_QUESTION_PROMPT)
    
    query = "Do we have data for symbol apple"
    res = qa({'question':query })
