# lab-work3-Ai-based
# News Summarization and Embedding System

This project provides a system for retrieving, summarizing, and embedding news articles using LangChain, HuggingFace, and NewsAPI. It includes modules for retrieving news articles, creating embeddings, summarizing articles, and managing user preferences and history.

---

## Features

1. **News Retrieval**: Fetch news articles from NewsAPI based on a topic.
2. **Embedding Creation**: Generate embeddings for articles using HuggingFace models and store them in a vector database (Chroma or FAISS).
3. **Article Summarization**: Generate brief or detailed summaries of articles using HuggingFace LLMs.
4. **User Management**: Track user preferences and search history.

---

## Setup

### Prerequisites

1. **Python 3.8+**: Ensure Python 3.8 or higher is installed.
2. **API Keys**:
   - **NewsAPI**: Sign up at [NewsAPI](https://newsapi.org/) and get an API key.
   - **HuggingFace**: Sign up at [HuggingFace](https://huggingface.co/) and get an API token.

### Installation

1. Install dependencies:
   pip install -r requirements.txt
   
2. Set up environment variables:
Create a .env file in the root directory and add your API keys:

### Usage
1. Retrieve News Articles: Use the NewsRetriever class to fetch articles from NewsAPI

2. Create Embeddings: Use the EmbeddingEngine class to generate and store embeddings

3. Summarize Articles: Use the ArticleSummarizer class to generate summaries

4. Manage User Preferences: Use the UserManager class to track user preferences and history
   
5. The user_data.json file is a JSON (JavaScript Object Notation) file used by the application to store and manage user preferences and search history. It is created and maintained by the UserManager class. Here's a detailed explanation of its structure and purpose:

The file contains two main sections:

    Preferences: Stores the user’s saved topics of interest and their preferred summary type.

    History: Stores a log of the user’s search queries, including the topic, summary type, and timestamp.


### Demonstration of the Application
### Example Workflow

#### 1. Launching the Application
```bash
PS C:\> python main.py
```
Upon running the script, the application starts and displays a list of available commands.

Commands:

  search <topic>  - Search for news on a specific topic
  
  save <topic>    - Save a topic of interest
  
  list            - List saved topics of interest
  
  remove <topic>  - Remove a topic of interest
  
  history         - View search history
  
  summary <type>  - Set summary type (brief or detailed)
  
  clear           - Clear search history
  
  help            - Display this help message
  
  exit            - Exit the application

#### 2. Searching for News on "AI"
```bash
> search AI
```
- The application retrieves and processes news articles related to AI.
- Generates a detailed summary of the retrieved articles.

##### Output:
```bash
Searching for news on: AI
Summary type: detailed
Retrieving articles...
Found 6 articles.
Creating embeddings...
Generating summary...

================================================================================
                                 SUMMARY: AI
================================================================================
Artificial Intelligence (AI) has been a hot topic in technology news recently, with advancements and discoveries being made in various sectors.   
Firstly, researchers from the University of California, Berkeley, and the University of Texas at Austin have identified significant flaws in      
popular AI models and are advocating for a new reporting system to ensure the timely and effective disclosure of such vulnerabilities. This comes 
as a response to the growing use and reliance on AI models in critical applications, and the potential consequences of undetected bugs or biases. 

        Meanwhile, Google has taken strides forward in the physical application of AI with its new Gemini Robotics AI model. The system, which wasannounced at the company's annual I/O developer conference, aims to provide humanoids and other robots with greater intelligence, enabling them tolearn from their environments and adapt to new situations more effectively. Google also unveiled a tool designed to help these robots make ethicaldecisions, which could be crucial for the safe and responsible implementation of AI in the real world.

        Lastly, Sony is reportedly experimenting with AI-powered game characters for its PlayStation platform. According to an anonymous tipster, 
the Japanese tech giant has been working on a prototype for at least one of its game characters, with the goal of enhancing their performance and 
creating more immersive gaming experiences. This move demonstrates the growing potential for AI in the entertainment industry, as well as the     
ongoing efforts to push the boundaries of what is possible with this technology.
```

#### 3. Changing Summary Type to Brief
```bash
> summary brief
```
- Sets summary type to brief for concise news summaries.

#### 4. Searching for News on "Dental Care"
```bash
> search dental care
```
##### Output:
```bash
Searching for news on: dental care
Summary type: brief
Retrieving articles...
Found 10 articles.
Creating embeddings...
Generating summary...

================================================================================
                              SUMMARY: DENTAL CARE
================================================================================
- Holistic dental care practices are gaining popularity to improve patient experiences.
- Some individuals seek affordable dental treatment options in Mexico.
- Pharmacies are evolving to provide dental services.
```

#### 5. Saving Topics of Interest
```bash
> save AI
> save dental care
```
- Saves "AI" and "Dental Care" topics for quick access later.

#### 6. Listing Saved Topics
```bash
> list
```
##### Output:
```bash
Your topics of interest:
1. CLIMATE CHANGE
2. dental care
3. AI
```

#### 7. Viewing Search History
```bash
> history
```
##### Output:
```bash
Your search history:
1. dental care (brief) - 2025-03-15
2. AI (detailed) - 2025-03-15
3. AI (detailed) - 2025-03-15
4. dental care (brief) - 2025-03-15
5. AI (brief) - 2025-03-15
```

#### 8. Exiting the Application
```bash
> exit
```
- The user exits the application after completing the session.

### Conclusion
This demonstration showcases the key functionalities of the News Summarizer, including searching, summarizing, saving topics, and retrieving history.



