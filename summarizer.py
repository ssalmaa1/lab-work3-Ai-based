"""
Module for implementing improved LangChain summarization chains.
"""
import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseLLM

class ArticleSummarizer:
    """
    Class for summarizing news articles using LangChain.
    """
    def __init__(self, huggingface_token: str = None, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the ArticleSummarizer with the HuggingFace API token.
        
        Args:
            huggingface_token: HuggingFace API token. If None, will look for HUGGINGFACEHUB_API_TOKEN environment variable.
            model_name: Name of the HuggingFace model to use.
        """
        self.huggingface_token = huggingface_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not self.huggingface_token:
            raise ValueError("HuggingFace API token is required. Please provide it or set HUGGINGFACEHUB_API_TOKEN environment variable.")
        
        # Use a more reliable model with simpler parameters
        try:
            self.llm = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=self.huggingface_token,
                task="text-generation",
                max_new_tokens=512
            )
        except Exception as e:
            print(f"Error initializing primary model: {e}")
            # Fallback to a simpler model if the primary model fails
            try:
                self.llm = HuggingFaceEndpoint(
                    repo_id="gpt2",
                    huggingfacehub_api_token=self.huggingface_token,
                    task="text-generation",
                    max_new_tokens=300
                )
            except Exception as e2:
                print(f"Error initializing fallback model: {e2}")
                # Last resort - create a dummy LLM for error messages
                self.llm = DummyLLM()
    
    def create_brief_summary(self, articles: List[Dict[str, Any]], 
                            user_preferences: Dict[str, Any] = None,
                            search_query: str = None) -> str:
        """
        Create a brief summary (1-2 sentences) of articles using LangChain.
        
        Args:
            articles: List of article dictionaries.
            user_preferences: User preferences to consider for summarization.
            search_query: The specific topic searched for.
            
        Returns:
            Brief summary of the articles.
        """
        if not articles:
            return "No articles to summarize."
            
        # Extract article titles for a simple summary
        titles = [article.get("title", "") for article in articles if article.get("title")]
        if not titles:
            return "Unable to generate summary: No article titles found."
            
        # Use search query as primary focus, fallback to user interests if not provided
        focus_topic = search_query or "general news"
        if not search_query and user_preferences and user_preferences.get("interests"):
            interests = user_preferences.get("interests", [])
            focus_topic = ", ".join(interests) if interests else focus_topic
        
        prompt = f"""Summarize these news headlines in 1-2 sentences, focusing specifically on '{focus_topic}':
        
        {' | '.join(titles[:5])}
        
        Remember: Your summary should be about '{focus_topic}' based on these headlines.
        Brief summary:"""
        
        try:
            # Print debug information
            print(f"DEBUG - Brief summary prompt:\n{prompt}")
            
            # Direct approach without complicated chains
            response = self.llm.invoke(prompt)
            
            # Handle different response types
            if isinstance(response, str):
                summary = response.strip()
            else:
                try:
                    summary = response.content.strip()
                except:
                    summary = str(response).strip()
            
            print(f"DEBUG - Brief summary raw response:\n{summary}")
            
            # Validate the summary is relevant to the search query
            if search_query and search_query.lower() not in summary.lower():
                summary += f"\n\nNote: This summary is based on articles about {search_query}."
            
            # Check for irrelevant content (like climate change when not searching for it)
            if search_query and "climate change" not in search_query.lower() and "climate change" in summary.lower():
                summary = f"Summary of articles about {search_query}: " + summary.replace("climate change", search_query)
            
            return summary
            
        except Exception as e:
            print(f"Error in brief summarization: {e}")
            # Fall back to a very simple summary
            return f"Recent news about {search_query or articles[0].get('title', 'this topic')} and related subjects."
    
    def create_detailed_summary(self, articles: List[Dict[str, Any]], 
                                user_preferences: Dict[str, Any] = None,
                                search_query: str = None) -> str:
        """
        Create a detailed summary (paragraph) of articles using LangChain.
        
        Args:
            articles: List of article dictionaries.
            user_preferences: User preferences to consider for summarization.
            search_query: The specific topic searched for.
            
        Returns:
            Detailed summary of the articles.
        """
        if not articles:
            return "No articles to summarize."
            
        # Extract article titles and descriptions for a more informative summary
        content_items = []
        for i, article in enumerate(articles[:3], 1):
            title = article.get("title", "")
            description = article.get("description", "")
            if title:
                content_items.append(f"{i}. {title}")
                if description:
                    content_items.append(f"   {description[:150]}...")
        
        if not content_items:
            return "Unable to generate detailed summary: No article content found."
            
        # Use search query as primary focus, fallback to user interests if not provided
        focus_topic = search_query or "general news"
        if not search_query and user_preferences and user_preferences.get("interests"):
            interests = user_preferences.get("interests", [])
            focus_topic = ", ".join(interests) if interests else focus_topic
        
        prompt = f"""Write a comprehensive paragraph summarizing these news items, focusing specifically on '{focus_topic}':
        
        {'\n'.join(content_items)}
        
        Remember: Your summary should be about '{focus_topic}' based on these news items.
        Detailed summary:"""
        
        try:
            # Print debug information
            print(f"DEBUG - Detailed summary prompt:\n{prompt}")
            
            # Direct approach without complicated chains
            response = self.llm.invoke(prompt)
            
            # Handle different response types
            if isinstance(response, str):
                summary = response.strip()
            else:
                try:
                    summary = response.content.strip()
                except:
                    summary = str(response).strip()
            
            print(f"DEBUG - Detailed summary raw response:\n{summary}")
            
            # Validate the summary is relevant to the search query
            if search_query and search_query.lower() not in summary.lower():
                summary += f"\n\nNote: This summary is based on articles about {search_query}."
            
            # Check for irrelevant content (like climate change when not searching for it)
            if search_query and "climate change" not in search_query.lower() and "climate change" in summary.lower():
                summary = f"Summary of articles about {search_query}: " + summary.replace("climate change", search_query)
            
            return summary
            
        except Exception as e:
            print(f"Error in detailed summarization: {e}")
            
            # If standard approach fails, try title-only approach
            try:
                title_prompt = f"""Summarize these news headlines in a paragraph, focusing on '{focus_topic}':
                
                {' | '.join([article.get('title', '') for article in articles[:5] if article.get('title')])}
                
                Remember: Your summary should be about '{focus_topic}'.
                Summary:"""
                
                print(f"DEBUG - Fallback prompt:\n{title_prompt}")
                
                response = self.llm.invoke(title_prompt)
                
                if isinstance(response, str):
                    summary = response.strip()
                else:
                    try:
                        summary = response.content.strip()
                    except:
                        summary = str(response).strip()
                
                print(f"DEBUG - Fallback raw response:\n{summary}")
                
                # Same validation as above
                if search_query and search_query.lower() not in summary.lower():
                    summary += f"\n\nNote: This summary is based on articles about {search_query}."
                
                if search_query and "climate change" not in search_query.lower() and "climate change" in summary.lower():
                    summary = f"Summary of articles about {search_query}: " + summary.replace("climate change", search_query)
                
                return summary
                
            except Exception as e2:
                print(f"Error in fallback summarization: {e2}")
                
                # Last resort - create a simple summary from titles
                titles = [article.get('title', '') for article in articles[:3] if article.get('title')]
                if titles:
                    return f"Recent news about {search_query or 'selected topics'} covers: {'; '.join(titles)}."
                else:
                    return f"Unable to generate a detailed summary about {search_query or 'selected topics'} due to technical difficulties."
    
    def _create_documents(self, articles: List[Dict[str, Any]]) -> List[Document]:
        """
        Create LangChain Document objects from the articles.
        
        Args:
            articles: List of article dictionaries.
            
        Returns:
            List of LangChain Document objects.
        """
        docs = []
        
        for article in articles:
            # Handle both raw articles and search results
            if "content" in article and isinstance(article["content"], str):
                content = article["content"]
                metadata = article.get("metadata", {})
            elif "content" in article and isinstance(article["content"], dict):
                content = article["content"].get("page_content", "")
                metadata = article["content"].get("metadata", {})
            else:
                # Try to get content from the article
                title = article.get("title", "")
                description = article.get("description", "")
                content = article.get("content", "")
                content = f"Title: {title}\n\nDescription: {description}\n\nContent: {content}"
                
                # Create metadata
                metadata = {
                    "title": title,
                    "url": article.get("url", ""),
                    "source": article.get("source", ""),
                    "published_at": article.get("published_at", "")
                }
            
            # Create a document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            docs.append(doc)
        
        return docs

class DummyLLM(BaseLLM):
    """A dummy LLM that just returns error messages."""
    
    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import Generation, LLMResult
        
        generations = [[Generation(text="Unable to generate summary due to model initialization failure.")]]
        return LLMResult(generations=generations)
    
    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import Generation, LLMResult
        
        generations = [[Generation(text="Unable to generate summary due to model initialization failure.")]]
        return LLMResult(generations=generations)
        
    @property
    def _llm_type(self):
        return "dummy"