# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 15:58:20 2023

@author: saura
"""

import pandas as pd
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
# from langchain.vectorstores import Pinecone



# encoding_name = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == '__main__':
    
    pineconekey = None
    cohere_key = None
    Your_text_data = None
    raw_text = Your_text_data
    
    #### Number of tokens #########
    num_tokens = num_tokens_from_string(raw_text,"cl100k_base")
    
    #### text splitter #######
    text_splitter = TokenTextSplitter(chunk_size=500, )
    docs = text_splitter.split_text(raw_text)
    
    ####### embedding , vector db ##########
    embeddings = CohereEmbeddings(model='embed-english-light-v2.0',cohere_api_key=cohere_key)
    # initialize pinecone
    pinecone.init(chunk_overlap=25,
        api_key=pineconekey,  # find at app.pinecone.io
        environment='gcp-starter',  # next to api key in console,
    )
    
    index_name = "fundamental"
    
    docsearch = Pinecone.from_texts(docs, embeddings, index_name=index_name)
    
    ######### retiriever ##########
    # load index
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    # initialize base retriever
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})
    # Set up cohere's reranker
    compressor = CohereRerank()
    reranker = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    
    #######