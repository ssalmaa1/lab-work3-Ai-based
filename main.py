"""
Main application interface for the news summarization application.
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Any
from dotenv import load_dotenv

from news_retriever import NewsRetriever
from embedding_engine import EmbeddingEngine
from summarizer import ArticleSummarizer
from user_manager import UserManager

def display_welcome():
    """Display welcome message and instructions."""
    print("\n" + "=" * 80)
    print("NEWS SUMMARIZER - Powered by LangChain".center(80))
    print("=" * 80)
    print("\nThis application retrieves news articles on specific topics")
    print("and creates concise summaries according to your preferences.")
    print("\nCommands:")
    print("  search <topic>  - Search for news on a specific topic")
    print("  save <topic>    - Save a topic of interest")
    print("  list            - List saved topics of interest")
    print("  remove <topic>  - Remove a topic of interest")
    print("  history         - View search history")
    print("  summary <type>  - Set summary type (brief or detailed)")
    print("  clear           - Clear search history")
    print("  help            - Display this help message")
    print("  exit            - Exit the application")
    print("=" * 80 + "\n")

def setup_environment():
    """Load environment variables and check for required API keys."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required API keys
    missing_keys = []
    
    if not os.environ.get('NEWSAPI_KEY'):
        missing_keys.append('NEWSAPI_KEY')
    
    if not os.environ.get('HUGGINGFACEHUB_API_TOKEN'):
        missing_keys.append('HUGGINGFACEHUB_API_TOKEN')
    
    if missing_keys:
        print("Error: The following environment variables are missing:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set these environment variables in a .env file or your system.")
        print("Example .env file:")
        print('NEWSAPI_KEY="your_newsapi_key_here"')
        print('HUGGINGFACEHUB_API_TOKEN="your_huggingface_token_here"')
        sys.exit(1)

def search_and_summarize(topic: str, user_manager: UserManager):
    """
    Search for news on a topic and create a summary.
    
    Args:
        topic: Topic to search for.
        user_manager: UserManager instance.
    """
    try:
        # Get user preferences
        preferences = user_manager.get_preferences()
        summary_type = preferences.get("summary_type", "brief")
        
        print(f"\nSearching for news on: {topic}")
        print(f"Summary type: {summary_type}")
        
        # Initialize the NewsRetriever
        news_retriever = NewsRetriever()
        
        # Retrieve articles
        print("Retrieving articles...")
        articles = news_retriever.get_articles(topic)
        
        if not articles:
            print("No articles found for this topic.")
            return
        
        print(f"Found {len(articles)} articles.")
        
        # Initialize the EmbeddingEngine
        embedding_engine = EmbeddingEngine(vector_store_type="chroma")
        
        # Create embeddings
        print("Creating embeddings...")
        embedding_engine.create_embeddings(articles, topic)
        
        # Initialize the ArticleSummarizer
        summarizer = ArticleSummarizer()
        
        # Create summary based on user preference
        print("Generating summary...")
        if summary_type == "brief":
            # Pass the search query explicitly to the summarizer
            summary = summarizer.create_brief_summary(articles, preferences, search_query=topic)
        else:  # detailed
            # Pass the search query explicitly to the summarizer
            summary = summarizer.create_detailed_summary(articles, preferences, search_query=topic)
        
        # Display the summary
        print("\n" + "=" * 80)
        print(f"SUMMARY: {topic.upper()}".center(80))
        print("=" * 80)
        print(f"\n{summary}\n")
        print("=" * 80)
        
        # Add to history
        user_manager.add_to_history(topic, summary_type)
        
        # Display article sources
        print("\nSources:")
        for i, article in enumerate(articles[:5], 1):
            print(f"{i}. {article.get('title', 'No title')} - {article.get('source', 'Unknown')}")
            print(f"   URL: {article.get('url', 'No URL')}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main application function."""
    # Setup environment
    setup_environment()
    
    # Initialize user manager
    user_manager = UserManager()
    
    # Display welcome message
    display_welcome()
    
    # Main application loop
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            # Parse the input
            parts = user_input.split(' ', 1)
            command = parts[0].lower() if parts else ""
            args = parts[1] if len(parts) > 1 else ""
            
            # Process the command
            if command == "exit":
                print("Goodbye!")
                break
            
            elif command == "help":
                display_welcome()
            
            elif command == "search":
                if not args:
                    print("Please specify a topic to search for.")
                    continue
                
                search_and_summarize(args, user_manager)
            
            elif command == "save":
                if not args:
                    print("Please specify a topic to save.")
                    continue
                
                user_manager.add_interest(args)
                print(f"Saved '{args}' to your interests.")
            
            elif command == "list":
                interests = user_manager.get_preferences().get("interests", [])
                if interests:
                    print("\nYour topics of interest:")
                    for i, interest in enumerate(interests, 1):
                        print(f"{i}. {interest}")
                else:
                    print("You haven't saved any topics of interest yet.")
            
            elif command == "remove":
                if not args:
                    print("Please specify a topic to remove.")
                    continue
                
                user_manager.remove_interest(args)
                print(f"Removed '{args}' from your interests.")
            
            elif command == "history":
                history = user_manager.get_history()
                if history:
                    print("\nYour search history:")
                    for i, item in enumerate(history, 1):
                        timestamp = item.get("timestamp", "").split("T")[0]
                        print(f"{i}. {item.get('topic')} ({item.get('summary_type')}) - {timestamp}")
                else:
                    print("Your search history is empty.")
            
            elif command == "summary":
                if args not in ["brief", "detailed"]:
                    print("Summary type must be 'brief' or 'detailed'.")
                    continue
                
                user_manager.update_preferences({"summary_type": args})
                print(f"Set summary type to '{args}'.")
            
            elif command == "clear":
                user_manager.clear_history()
                print("Search history cleared.")
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for a list of commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()