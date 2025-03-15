"""
Module for retrieving news articles from NewsAPI.
"""
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class NewsRetriever:
    """
    Class for retrieving news articles from NewsAPI.
    """
    BASE_URL = "https://newsapi.org/v2/everything"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the NewsRetriever with the NewsAPI key.
        
        Args:
            api_key: NewsAPI key. If None, will look for NEWSAPI_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("NewsAPI key is required. Please provide it or set NEWSAPI_KEY environment variable.")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_articles(self, 
                    topic: str, 
                    days_back: int = 7, 
                    language: str = "en", 
                    sort_by: str = "relevancy",
                    page_size: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve news articles for a specific topic with retry capability.
        
        Args:
            topic: The topic to search for.
            days_back: Number of days to look back for articles.
            language: Language of articles (default: English).
            sort_by: Sort order (relevancy, popularity, publishedAt).
            page_size: Number of articles to return.
            
        Returns:
            List of article dictionaries.
        """
        # Calculate the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for the API
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")
        
        # Prepare the API request
        params = {
            "q": topic,
            "from": from_date,
            "to": to_date,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "apiKey": self.api_key
        }
        
        # Make the API request
        response = requests.get(self.BASE_URL, params=params)
        
        # Check if the request was successful
        if response.status_code != 200:
            error_msg = f"Failed to retrieve articles: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        # Parse the response
        data = response.json()
        
        # Check for API errors
        if data.get("status") != "ok":
            error_msg = f"API error: {data.get('message', 'Unknown error')}"
            raise Exception(error_msg)
        
        # Extract the articles
        articles = data.get("articles", [])
        
        # Process the articles to include only the relevant information
        processed_articles = []
        for article in articles:
            # Skip articles with missing content
            if not article.get("content") or not article.get("title"):
                continue
                
            processed_article = {
                "title": article.get("title"),
                "author": article.get("author"),
                "source": article.get("source", {}).get("name"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "content": article.get("content"),
                "description": article.get("description")
            }
            processed_articles.append(processed_article)
        
        return processed_articles

    def get_article_content(self, article: Dict[str, Any]) -> str:
        """
        Extract the content from an article for embedding and summarization.
        
        Args:
            article: Article dictionary.
            
        Returns:
            Formatted article content as a string.
        """
        # Create a formatted version of the article content
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        author = article.get("author", "Unknown")
        source = article.get("source", "Unknown")
        
        # Combine all text parts with improved formatting
        full_content = (
            f"Title: {title}\n\n"
            f"Author: {author}\n"
            f"Source: {source}\n\n"
            f"Description: {description}\n\n"
            f"Content: {content}"
        )
        return full_content