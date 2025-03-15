"""
Module for creating and managing article embeddings.
"""
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class EmbeddingEngine:
    """
    Class for creating and managing article embeddings.
    """
    def __init__(self, 
                embedding_model: Optional[Embeddings] = None, 
                vector_store_type: str = "chroma",
                persist_directory: str = "./vector_db"):
        """
        Initialize the EmbeddingEngine with embedding model and vector store.
        
        Args:
            embedding_model: LangChain compatible embedding model.
            vector_store_type: Type of vector store to use ("chroma" or "faiss").
            persist_directory: Directory to persist vector store.
        """
        # Updated to use non-deprecated HuggingFaceEmbeddings
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.vector_store_type = vector_store_type.lower()
        self.persist_directory = persist_directory
        self.vector_store = None
        os.makedirs(persist_directory, exist_ok=True)
    
    def _sanitize_topic(self, topic: str) -> str:
        """
        Sanitize the topic name to be compatible with vector store requirements.
        """
        # Ensure the topic contains only alphanumeric characters, underscores, and hyphens
        topic = re.sub(r'[^a-zA-Z0-9-_]', '_', topic).strip('_')
        
        # Ensure the topic starts with a letter to avoid issues with topics like "AI"
        if not topic or not topic[0].isalpha():
            topic = "topic_" + topic
            
        # Ensure the topic is at least 3 characters
        if len(topic) < 3:
            topic = topic + "_collection"
            
        # Ensure the topic is not more than 63 characters
        return topic[:63]

    def create_embeddings(self, articles: List[Dict[str, Any]], topic: str) -> None:
        """
        Create embeddings for articles and store them in the vector store.
        """
        topic = self._sanitize_topic(topic)
        texts = []
        metadatas = []
        
        # Filter out articles with empty content
        for article in articles:
            content = self._get_article_content(article)
            if not content:
                continue
            texts.append(content)
            metadatas.append({
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "published_at": article.get("published_at", ""),
                "topic": topic
            })
        
        if not texts:
            print("Warning: No valid article content found to create embeddings.")
            return
            
        # Updated to use non-deprecated Chroma
        if self.vector_store_type == "chroma":
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=f"{self.persist_directory}/{topic}",
                collection_name=topic
            )
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            # No longer need to call persist() since Chroma persists automatically
        elif self.vector_store_type == "faiss":
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas
            )
            self.vector_store.save_local(f"{self.persist_directory}/{topic}")
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def search_articles(self, query: str, topic: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for articles by similarity to query.
        """
        try:
            topic = self._sanitize_topic(topic)
            if self.vector_store_type == "chroma":
                vector_store = Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=f"{self.persist_directory}/{topic}",
                    collection_name=topic
                )
            elif self.vector_store_type == "faiss":
                vector_store = FAISS.load_local(
                    f"{self.persist_directory}/{topic}",
                    self.embedding_model
                )
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
                
            results = vector_store.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                } for doc, score in results
            ]
        except Exception as e:
            print(f"Error searching for articles: {e}")
            return []
    
    def _get_article_content(self, article: Dict[str, Any]) -> str:
        """
        Extract content from article.
        """
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        
        # Skip articles with no meaningful content
        if not title and not content:
            return ""
            
        return f"Title: {title}\n\nDescription: {description}\n\nContent: {content}"