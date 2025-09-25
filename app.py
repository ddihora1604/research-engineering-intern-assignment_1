from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import pandas as pd
from datetime import datetime
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer
import traceback
import math
import glob
import re
import google.generativeai as genai
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialMediaConnector:
    """Base class for social media platform data connectors."""
    
    def __init__(self, platform_name):
        self.platform_name = platform_name
        self.data = None
    
    def load_data(self, source_path):
        """Load data from the source path."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def normalize_data(self):
        """Normalize data to a standard format."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_data(self):
        """Return the normalized data."""
        return self.data

class RedditConnector(SocialMediaConnector):
    """Connector for Reddit data in JSONL format."""
    
    def __init__(self):
        super().__init__("reddit")
    
    def load_data(self, source_path):
        """Load Reddit data from JSONL file."""
        try:
            if os.path.exists(source_path):
                # Read JSONL file
                self.data = pd.read_json(source_path, lines=True)
                print(f"Loaded {len(self.data)} rows from Reddit data source: {source_path}")
                return True
            else:
                print(f"Reddit data file not found at {source_path}")
                return False
        except Exception as e:
            print(f"Error loading Reddit data: {str(e)}")
            return False
    
    def normalize_data(self):
        """Normalize Reddit data to standard format."""
        if self.data is None:
            return False
        
        try:
            # Normalize the nested JSON structure
            self.data = pd.json_normalize(self.data['data'])
            
            # Convert created_utc to datetime
            self.data['created_utc'] = pd.to_datetime(self.data['created_utc'], unit='s')
            
            # Add platform identifier
            self.data['platform'] = 'reddit'
            
            # Ensure standard column names
            # Map: content_text (post content), title, author, created_at, platform, engagement_count, etc.
            self.data['content_text'] = self.data['selftext'].fillna('')
            self.data['engagement_count'] = self.data['num_comments']
            self.data['created_at'] = self.data['created_utc']
            self.data['post_id'] = self.data['id']
            self.data['community'] = self.data['subreddit']
            
            print(f"Normalized Reddit data: {len(self.data)} rows")
            return True
        except Exception as e:
            print(f"Error normalizing Reddit data: {str(e)}")
            return False

class TwitterConnector(SocialMediaConnector):
    """Connector for Twitter/X data in JSONL format."""
    
    def __init__(self):
        super().__init__("twitter")
    
    def load_data(self, source_path):
        """Load Twitter data from JSONL file."""
        try:
            if os.path.exists(source_path):
                # Read JSONL file
                self.data = pd.read_json(source_path, lines=True)
                print(f"Loaded {len(self.data)} rows from Twitter data source: {source_path}")
                return True
            else:
                print(f"Twitter data file not found at {source_path}")
                return False
        except Exception as e:
            print(f"Error loading Twitter data: {str(e)}")
            return False
    
    def normalize_data(self):
        """Normalize Twitter data to standard format."""
        if self.data is None:
            return False
        
        try:
            # Add platform identifier
            self.data['platform'] = 'twitter'
            
            # Standardize column names - assuming Twitter data structure
            if 'full_text' in self.data.columns:
                self.data['content_text'] = self.data['full_text'].fillna('')
            else:
                self.data['content_text'] = self.data['text'].fillna('')
            
            if 'created_at' in self.data.columns:
                if isinstance(self.data['created_at'].iloc[0], str):
                    self.data['created_at'] = pd.to_datetime(self.data['created_at'])
            
            # Map standard fields
            self.data['title'] = ''  # Twitter doesn't have titles
            self.data['engagement_count'] = self.data.get('retweet_count', 0) + self.data.get('favorite_count', 0)
            self.data['post_id'] = self.data['id_str'] if 'id_str' in self.data.columns else self.data['id']
            self.data['community'] = ''  # Twitter doesn't have direct community equivalent
            
            print(f"Normalized Twitter data: {len(self.data)} rows")
            return True
        except Exception as e:
            print(f"Error normalizing Twitter data: {str(e)}")
            return False

class PlatformDataManager:
    """Manages data from multiple social media platforms."""
    
    def __init__(self):
        self.connectors = {}
        self.integrated_data = None
        self.platform_data = {}
    
    def add_connector(self, connector):
        """Add a platform connector."""
        self.connectors[connector.platform_name] = connector
    
    def load_platform_data(self, platform, source_path):
        """Load and normalize data for a specific platform."""
        if platform not in self.connectors:
            print(f"No connector available for platform: {platform}")
            return False
        
        connector = self.connectors[platform]
        if connector.load_data(source_path):
            if connector.normalize_data():
                self.platform_data[platform] = connector.get_data()
                return True
        return False
    
    def integrate_data(self):
        """Combine data from all platforms into a unified dataset."""
        if not self.platform_data:
            print("No platform data loaded")
            return False
        
        try:
            # Combine all dataframes
            dataframes = []
            for platform, df in self.platform_data.items():
                if not df.empty:
                    dataframes.append(df)
            
            if dataframes:
                self.integrated_data = pd.concat(dataframes, ignore_index=True)
                print(f"Integrated data created with {len(self.integrated_data)} total rows")
                return True
            else:
                print("No valid dataframes to integrate")
                return False
        except Exception as e:
            print(f"Error integrating data: {str(e)}")
            return False
    
    def get_platform_data(self, platform=None):
        """Get data for a specific platform or all integrated data."""
        if platform:
            return self.platform_data.get(platform)
        return self.integrated_data
    
    def get_available_platforms(self):
        """Get list of platforms with loaded data."""
        return list(self.platform_data.keys())
    
    def filter_data(self, query, platform=None):
        """Filter data based on query and optional platform."""
        if platform:
            if platform not in self.platform_data:
                return pd.DataFrame()
            df = self.platform_data[platform]
            return df[
                df['content_text'].str.contains(query, case=False, na=False) | 
                df['title'].str.contains(query, case=False, na=False)
            ]
        else:
            if self.integrated_data is None:
                return pd.DataFrame()
            return self.integrated_data[
                self.integrated_data['content_text'].str.contains(query, case=False, na=False) | 
                self.integrated_data['title'].str.contains(query, case=False, na=False)
            ]
    
    def discover_data_files(self):
        """Automatically discover data files for different platforms."""
        discovered_files = {}
        data_dir = "./data"
        
        # Look for Reddit data files
        reddit_files = glob.glob(f"{data_dir}/*reddit*.jsonl")
        if reddit_files:
            discovered_files["reddit"] = reddit_files[0]
        
        # Look for Twitter data files
        twitter_files = glob.glob(f"{data_dir}/*twitter*.jsonl") + glob.glob(f"{data_dir}/*tweet*.jsonl")
        if twitter_files:
            discovered_files["twitter"] = twitter_files[0]
        
        # Default to data.jsonl as Reddit if no specific files found
        if "reddit" not in discovered_files and os.path.exists(f"{data_dir}/data.jsonl"):
            discovered_files["reddit"] = f"{data_dir}/data.jsonl"
        
        return discovered_files

# Load environment variables from .env file if it exists
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variable to store the loaded data
data = None
# Path to the dataset file
DATASET_PATH = "./data/data.jsonl"

# Create the platform data manager
platform_manager = PlatformDataManager()

# Global variables for model caching
model_cache = {}
MAX_CACHE_SIZE = 2  # Maximum number of models to keep in cache

def load_model_with_cache(model_name, model_loader):
    """Load model with caching"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    # Load the model
    model = model_loader()
    
    # Cache management
    if len(model_cache) >= MAX_CACHE_SIZE:
        # Remove oldest model from cache
        oldest_key = next(iter(model_cache))
        del model_cache[oldest_key]
    
    model_cache[model_name] = model
    return model

# Initialize tokenizer and model for summarization
try:
    # Keep the existing flan-t5 model as fallback
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Initialize sentence transformer for embeddings
    semantic_model = None  # Will be loaded on demand to save memory
    
    # Check for Groq API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    has_groq = GROQ_API_KEY is not None and GROQ_API_KEY != ""
    if has_groq:
        print("Groq API key found. Enhanced LLM insights will be available.")
    else:
        print("No Groq API key found. Set GROQ_API_KEY in environment or .env file for enhanced insights.")
    
    # Check for Google Gemini API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    has_gemini = GEMINI_API_KEY is not None and GEMINI_API_KEY != ""
    if has_gemini:
        try:
            print("Gemini API key found. Enhanced chatbot functionality will be available.")
            # Configure Gemini API
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            has_gemini = False
    else:
        print("No Gemini API key found. Set GEMINI_API_KEY in environment or .env file for enhanced chatbot functionality.")
except Exception as e:
    print(f"Error initializing models: {e}")
    t5_tokenizer = None
    t5_model = None
    semantic_model = None
    has_groq = False
    has_gemini = False

# Load dataset on startup
def load_dataset():

    global data
    try:
        if os.path.exists(DATASET_PATH):
            # Read JSONL file
            data = pd.read_json(DATASET_PATH, lines=True)
            # Normalize the nested JSON structure
            data = pd.json_normalize(data['data'])
            # Convert created_utc to datetime
            data['created_utc'] = pd.to_datetime(data['created_utc'], unit='s')
            print(f"Dataset loaded successfully: {len(data)} rows")
            return True
        else:
            print(f"Dataset file not found at {DATASET_PATH}")
            return False
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return False

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/api/timeseries', methods=['GET'])
def get_timeseries():
    """
    Generates time series data for posts matching a query within a date range.
    
    This endpoint:
    1. Filters data based on the search query and optional date range
    2. Groups posts by date
    3. Counts posts per date to create the time series
    
    Query Parameters:
        query (str): Search term to filter posts
        start_date (str, optional): Start date for filtering (YYYY-MM-DD)
        end_date (str, optional): End date for filtering (YYYY-MM-DD)
    
    Returns:
        JSON: Array of objects containing date and post count
    """
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Filter data based on query and date range
    filtered_data = data[data['selftext'].str.contains(query, case=False, na=False) | 
                          data['title'].str.contains(query, case=False, na=False)]
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_data = filtered_data[
            (filtered_data['created_utc'] >= start_date) & 
            (filtered_data['created_utc'] <= end_date)
        ]
    
    # Group by date and count posts
    timeseries = filtered_data.groupby(filtered_data['created_utc'].dt.date).size().reset_index()
    timeseries.columns = ['date', 'count']
    
    return jsonify(timeseries.to_dict('records'))

@app.route('/api/top_contributors', methods=['GET'])
def get_top_contributors():
 
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 10))
    
    filtered_data = data[data['selftext'].str.contains(query, case=False, na=False) | 
                          data['title'].str.contains(query, case=False, na=False)]
    top_users = filtered_data['author'].value_counts().head(limit).reset_index()
    top_users.columns = ['author', 'count']
    
    return jsonify(top_users.to_dict('records'))

@app.route('/api/network', methods=['GET'])
def get_network():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    network_type = request.args.get('network_type', 'interaction')
    content_type = request.args.get('content_type', 'all')
    min_similarity = float(request.args.get('min_similarity', 0.2))
    
    # Filter data
    filtered_data = data[
        data['selftext'].str.contains(query, case=False, na=False) |
        data['title'].str.contains(query, case=False, na=False)
    ]
    
    if len(filtered_data) == 0:
        return jsonify({'nodes': [], 'links': []})
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for all authors
    author_counts = filtered_data['author'].value_counts()
    for author, count in author_counts.items():
        G.add_node(author, size=min(count*3, 30), posts=int(count))
    
    if network_type == 'interaction':
        # Traditional interaction network (unchanged)
        # Add edges based on interactions (comments)
        comment_edges = []
        for idx, row in filtered_data.iterrows():
            if pd.notna(row.get('parent_id')) and row.get('parent_id', '').startswith('t3_'):
                parent_post = filtered_data[filtered_data['id'] == row.get('parent_id')[3:]]
                if not parent_post.empty:
                    parent_author = parent_post.iloc[0]['author']
                    if parent_author != row['author']:  # Don't count self-interactions
                        comment_edges.append((row['author'], parent_author))
        
        # Count edge weights
        edge_weights = {}
        for source, target in comment_edges:
            if (source, target) not in edge_weights:
                edge_weights[(source, target)] = 0
            edge_weights[(source, target)] += 1
        
        # Add weighted edges
        for (source, target), weight in edge_weights.items():
            G.add_edge(source, target, weight=weight)
    
    else:
        # Content-based network
        # Extract shared content between authors
        import re
        from collections import defaultdict
        
        # Functions to extract different content types
        def extract_keywords(text):
            if not isinstance(text, str):
                return []
            # Simple keyword extraction - could be improved with NLP
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            # Filter out common words
            common_words = {'about', 'after', 'again', 'also', 'around', 'before', 'being', 'between',
                           'could', 'every', 'from', 'have', 'here', 'most', 'need', 'other', 'should',
                           'since', 'there', 'these', 'they', 'this', 'those', 'through', 'using',
                           'very', 'what', 'when', 'where', 'which', 'while', 'would', 'your'}
            return [w for w in words if w not in common_words]
        
        def extract_hashtags(text):
            if not isinstance(text, str):
                return []
            return re.findall(r'#[a-zA-Z0-9_]+', text.lower())
        
        def extract_urls(text):
            if not isinstance(text, str):
                return []
            return re.findall(r'https?://\S+', text.lower())
        
        # Extract content for each author
        author_content = defaultdict(lambda: {'keywords': set(), 'hashtags': set(), 'urls': set()})
        
        for _, row in filtered_data.iterrows():
            # Combine title and selftext
            full_text = f"{row['title']} {row.get('selftext', '')}"
            author = row['author']
            
            # Extract content based on requested type
            if content_type in ['all', 'keywords']:
                author_content[author]['keywords'].update(extract_keywords(full_text))
            
            if content_type in ['all', 'hashtags']:
                author_content[author]['hashtags'].update(extract_hashtags(full_text))
            
            if content_type in ['all', 'urls']:
                author_content[author]['urls'].update(extract_urls(full_text))
        
        # Find shared content between authors
        authors = list(author_content.keys())
        content_edges = []
        
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                author1 = authors[i]
                author2 = authors[j]
                
                shared_content = {
                    'keywords': author_content[author1]['keywords'].intersection(author_content[author2]['keywords']),
                    'hashtags': author_content[author1]['hashtags'].intersection(author_content[author2]['hashtags']),
                    'urls': author_content[author1]['urls'].intersection(author_content[author2]['urls'])
                }
                
                # Calculate similarity score based on shared content
                total_shared = len(shared_content['keywords']) + len(shared_content['hashtags']) + len(shared_content['urls'])
                
                # Only create edges if there's meaningful shared content
                if total_shared > 0:
                    # Calculate similarity score 
                    author1_total = sum(len(author_content[author1][ct]) for ct in ['keywords', 'hashtags', 'urls'])
                    author2_total = sum(len(author_content[author2][ct]) for ct in ['keywords', 'hashtags', 'urls'])
                    
                    if author1_total > 0 and author2_total > 0:
                        # Jaccard similarity: intersection / union
                        similarity = total_shared / (author1_total + author2_total - total_shared)
                        
                        if similarity >= min_similarity:
                            # Create edge with shared content metadata
                            content_edges.append((
                                author1, 
                                author2, 
                                {
                                    'weight': total_shared,
                                    'similarity': similarity,
                                    'shared_keywords': list(shared_content['keywords'])[:10],  # Limit to top 10
                                    'shared_hashtags': list(shared_content['hashtags']),
                                    'shared_urls': list(shared_content['urls']),
                                    'total_shared': total_shared
                                }
                            ))
        
        # Add content-based edges to graph
        for source, target, attrs in content_edges:
            G.add_edge(source, target, **attrs)
            # Make the graph undirected for content sharing
            G.add_edge(target, source, **attrs)
    
    # Find communities using Louvain method
    if len(G.nodes()) > 0:
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G.to_undirected())
            nx.set_node_attributes(G, partition, 'group')
        except Exception as e:
            print(f"Community detection error: {str(e)}")
            # Fallback if community detection fails
            for node in G.nodes():
                G.nodes[node]['group'] = 0
    
    # Enhance node metadata with content info
    if network_type != 'interaction':
        for node in G.nodes():
            # Add content statistics to nodes
            if node in author_content:
                G.nodes[node]['keyword_count'] = len(author_content[node]['keywords'])
                G.nodes[node]['hashtag_count'] = len(author_content[node]['hashtags'])
                G.nodes[node]['url_count'] = len(author_content[node]['urls'])
                
                # Add top keywords to nodes (limited to 5)
                G.nodes[node]['top_keywords'] = list(author_content[node]['keywords'])[:5]
                G.nodes[node]['top_hashtags'] = list(author_content[node]['hashtags'])[:5]
    
    # Convert to D3.js format
    nodes = []
    for node in G.nodes():
        # Basic node info
        node_data = {
            'id': node, 
            'size': G.nodes[node].get('size', 10),
            'group': G.nodes[node].get('group', 0),
            'posts': G.nodes[node].get('posts', 1)
        }
        
        # Add content metadata if available
        if network_type != 'interaction':
            node_data.update({
                'keyword_count': G.nodes[node].get('keyword_count', 0),
                'hashtag_count': G.nodes[node].get('hashtag_count', 0),
                'url_count': G.nodes[node].get('url_count', 0),
                'top_keywords': G.nodes[node].get('top_keywords', []),
                'top_hashtags': G.nodes[node].get('top_hashtags', [])
            })
        
        nodes.append(node_data)
    
    # Convert links with enhanced metadata
    links = []
    for source, target in G.edges():
        link_data = {
            'source': source, 
            'target': target,
            'weight': G.edges[source, target].get('weight', 1)
        }
        
        # Add shared content info if available
        if network_type != 'interaction':
            link_data.update({
                'similarity': G.edges[source, target].get('similarity', 0),
                'shared_keywords': G.edges[source, target].get('shared_keywords', []),
                'shared_hashtags': G.edges[source, target].get('shared_hashtags', []),
                'shared_urls': G.edges[source, target].get('shared_urls', []),
                'total_shared': G.edges[source, target].get('total_shared', 0)
            })
        
        links.append(link_data)
    
    # Calculate network metrics
    metrics = {
        'node_count': len(nodes),
        'edge_count': len(links),
        'network_type': network_type,
        'content_type': content_type if network_type != 'interaction' else None,
        'density': nx.density(G) if len(G.nodes()) > 1 else 0,
        'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
    }
    
    return jsonify({
        'nodes': nodes, 
        'links': links,
        'metrics': metrics,
        'network_type': network_type
    })

@app.route('/api/topics', methods=['GET'])
def get_topics():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    n_topics = int(request.args.get('n_topics', 5))
    query = request.args.get('query', '')
    
    # Filter data if query is provided
    filtered_data = data
    if query:
        filtered_data = data[
            data['selftext'].str.contains(query, case=False, na=False) |
            data['title'].str.contains(query, case=False, na=False)
        ]
    
    if len(filtered_data) == 0:
        return jsonify([])
    
    # Prepare text data - combine title and selftext for better topic detection
    filtered_data['combined_text'] = filtered_data['title'] + ' ' + filtered_data['selftext'].fillna('')
    
    # Prepare text data
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(filtered_data['combined_text'])
    
    # Apply LDA with improved parameters
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=42,
        learning_method='online',
        max_iter=50,
        learning_decay=0.7,
        evaluate_every=10
    )
    
    # Fit the model and transform the data to get document-topic distributions
    doc_topic_dists = lda.fit_transform(X)
    
    # Get top words for each topic with relevance scores
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        # Get the top words with their weights
        sorted_indices = topic.argsort()[:-20-1:-1]
        top_words = [feature_names[i] for i in sorted_indices]
        top_weights = [float(topic[i]) for i in sorted_indices]
        
        # Normalize weights to percentages for easier interpretation
        total_weight = sum(top_weights)
        top_weights_normalized = [round((w / total_weight) * 100, 2) for w in top_weights]
        
        # Create topic object with enhanced metadata
        topic_obj = {
            'topic_id': topic_idx,
            'top_words': top_words[:15],  # Top 15 words
            'weights': top_weights_normalized[:15],  # Corresponding weights
            'word_weight_pairs': [{'word': w, 'weight': round(wt, 2)} 
                                  for w, wt in zip(top_words[:15], top_weights_normalized[:15])]
        }
        
        # Find representative documents for this topic
        topic_docs = []
        for i, dist in enumerate(doc_topic_dists):
            if np.argmax(dist) == topic_idx and dist[topic_idx] > 0.5:  # Strong topic alignment
                if len(topic_docs) < 3:  # Limit to 3 examples
                    doc = filtered_data.iloc[i]
                    topic_docs.append({
                        'title': doc['title'],
                        'author': doc['author'],
                        'subreddit': doc.get('subreddit', ''),
                        'created_utc': doc['created_utc'].isoformat(),
                        'topic_probability': float(dist[topic_idx])
                    })
        
        topic_obj['representative_docs'] = topic_docs
        topics.append(topic_obj)
    
    # Time-based topic distribution (how topics evolve over time)
    try:
        # Add topic assignments to the data
        filtered_data['dominant_topic'] = np.argmax(doc_topic_dists, axis=1)
        
        # Group by date and topic
        filtered_data['date'] = filtered_data['created_utc'].dt.date
        topic_evolution = {}
        
        # For each topic, get its frequency over time
        for topic_idx in range(n_topics):
            topic_docs = filtered_data[filtered_data['dominant_topic'] == topic_idx]
            if not topic_docs.empty:
                time_dist = topic_docs.groupby('date').size()
                topic_evolution[f'topic_{topic_idx}'] = {
                    str(date): int(count) for date, count in time_dist.items()
                }
        
        # Calculate overall coherence score
        coherence_score = sum(np.max(doc_topic_dists, axis=1)) / len(doc_topic_dists)
        
        return jsonify({
            'topics': topics,
            'topic_evolution': topic_evolution,
            'coherence_score': float(coherence_score),
            'n_docs_analyzed': len(filtered_data)
        })
    except Exception as e:
        # If time-based analysis fails, return just the topics
        return jsonify({
            'topics': topics,
            'error': str(e)
        })

@app.route('/api/coordinated', methods=['GET'])
def get_coordinated_behavior():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    time_window = int(request.args.get('time_window', 3600))  # Default to 1 hour in seconds
    similarity_threshold = float(request.args.get('similarity_threshold', 0.7))
    query = request.args.get('query', '')
    
    # Filter by query if specified
    filtered_data = data
    if query:
        filtered_data = data[
            data['selftext'].str.contains(query, case=False, na=False) |
            data['title'].str.contains(query, case=False, na=False)
        ]
    
    # Step 1: Sort data by timestamp
    sorted_data = filtered_data.sort_values('created_utc')
    
    # Step 2: Find posts with similar content in close time periods using improved similarity metrics
    coordinated_groups = []
    processed_indices = set()
    
    # Create a TF-IDF vectorizer for better similarity comparison
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        ngram_range=(1, 2)  # Include bigrams for better context
    )
    
    try:
        # Combine all available text for similarity analysis
        sorted_data['analysis_text'] = sorted_data['title']
        if 'selftext' in sorted_data.columns:
            sorted_data['analysis_text'] += ' ' + sorted_data['selftext'].fillna('')
        
        # Create matrix of TF-IDF features (might be sparse for large datasets)
        tfidf_matrix = tfidf_vectorizer.fit_transform(sorted_data['analysis_text'])
        
        # Enhanced coordinated group detection using vector similarity
        for i, row1 in sorted_data.iterrows():
            if i in processed_indices:
                continue
                
            group = [{
                'author': row1['author'],
                'id': row1.get('id', ''),
                'created_utc': row1['created_utc'].isoformat(),
                'title': row1['title'],
                'selftext': row1.get('selftext', '')[:200] + '...' if len(row1.get('selftext', '') or '') > 200 else row1.get('selftext', ''),
                'url': f"https://reddit.com/{row1.get('permalink', '')}"
            }]
            
            # Find posts within the time window
            time_limit = row1['created_utc'] + pd.Timedelta(seconds=time_window)
            window_posts = sorted_data[(sorted_data['created_utc'] <= time_limit) & 
                                      (sorted_data['created_utc'] >= row1['created_utc'])]
            
            # Find posts with similar content
            row1_vector = tfidf_matrix[sorted_data.index.get_loc(i)]
            
            for j, row2 in window_posts.iterrows():
                if i == j or j in processed_indices:
                    continue
                    
                # Calculate cosine similarity using TF-IDF vectors
                row2_vector = tfidf_matrix[sorted_data.index.get_loc(j)]
                similarity = cosine_similarity(row1_vector, row2_vector)[0][0]
                
                # Check for shared links, URLs or hashtags to improve detection
                shared_links = False
                shared_hashtags = False
                
                # Extract URLs and hashtags if available
                if 'selftext' in row1 and 'selftext' in row2:
                    # Simple regex to find URLs and hashtags (could be improved)
                    import re
                    urls1 = set(re.findall(r'https?://\S+', str(row1.get('selftext', ''))))
                    urls2 = set(re.findall(r'https?://\S+', str(row2.get('selftext', ''))))
                    hashtags1 = set(re.findall(r'#\w+', str(row1.get('selftext', ''))))
                    hashtags2 = set(re.findall(r'#\w+', str(row2.get('selftext', ''))))
                    
                    # Check for overlap
                    if urls1 and urls2 and urls1.intersection(urls2):
                        shared_links = True
                        similarity += 0.1  # Boost similarity score for shared links
                    
                    if hashtags1 and hashtags2 and hashtags1.intersection(hashtags2):
                        shared_hashtags = True
                        similarity += 0.1  # Boost similarity score for shared hashtags
                
                if similarity >= similarity_threshold:
                    group.append({
                        'author': row2['author'],
                        'id': row2.get('id', ''),
                        'created_utc': row2['created_utc'].isoformat(),
                        'title': row2['title'],
                        'selftext': row2.get('selftext', '')[:200] + '...' if len(row2.get('selftext', '') or '') > 200 else row2.get('selftext', ''),
                        'url': f"https://reddit.com/{row2.get('permalink', '')}",
                        'similarity_score': round(float(similarity), 3),
                        'shared_links': shared_links,
                        'shared_hashtags': shared_hashtags
                    })
                    processed_indices.add(j)
            
            if len(group) > 1:  # Only consider groups with at least 2 posts
                # Add metadata about the group
                group_metadata = {
                    'group_id': len(coordinated_groups),
                    'size': len(group),
                    'time_span': (max([pd.to_datetime(p['created_utc']) for p in group]) - 
                                 min([pd.to_datetime(p['created_utc']) for p in group])).total_seconds(),
                    'unique_authors': len(set([p['author'] for p in group])),
                    'shared_links_count': sum(1 for p in group if p.get('shared_links', False)),
                    'shared_hashtags_count': sum(1 for p in group if p.get('shared_hashtags', False)),
                    'posts': group
                }
                coordinated_groups.append(group_metadata)
                processed_indices.add(i)
    except Exception as e:
        # Fallback to simpler method if advanced method fails
        print(f"Advanced coordination detection failed: {str(e)}")
        # (Original simpler method would go here)
    
    # Step 3: Create network of coordinated authors
    author_links = []
    author_nodes = set()
    
    for group in coordinated_groups:
        authors = [post['author'] for post in group['posts']]
        author_nodes.update(authors)
        
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                if authors[i] != authors[j]:  # Avoid self-loops
                    # Add weight based on frequency of coordination
                    author_links.append({
                        'source': authors[i], 
                        'target': authors[j],
                        'group_id': group['group_id'],
                        'weight': 1  # Could be enhanced to count multiple instances
                    })
    
    # Aggregate weights for duplicate links
    link_weights = defaultdict(int)
    for link in author_links:
        key = tuple(sorted([link['source'], link['target']]))
        link_weights[key] += link['weight']
    
    # Create final weighted links
    unique_links = [
        {'source': source, 'target': target, 'weight': weight}
        for (source, target), weight in link_weights.items()
    ]
    
    # Create nodes with metadata
    author_post_counts = filtered_data['author'].value_counts().to_dict()
    nodes = [
        {
            'id': author,
            'posts_count': author_post_counts.get(author, 0),
            'coordinated_groups_count': sum(1 for g in coordinated_groups if author in [p['author'] for p in g['posts']])
        }
        for author in author_nodes
    ]
    
    # Calculate network metrics
    network_metrics = {
        'total_groups': len(coordinated_groups),
        'total_authors': len(author_nodes),
        'total_connections': len(unique_links),
        'density': len(unique_links) / (len(author_nodes) * (len(author_nodes) - 1) / 2) if len(author_nodes) > 1 else 0,
        'avg_group_size': sum(g['size'] for g in coordinated_groups) / len(coordinated_groups) if coordinated_groups else 0,
        'authors_involved_percentage': len(author_nodes) / filtered_data['author'].nunique() * 100,
        'time_window_seconds': time_window,
        'similarity_threshold': similarity_threshold
    }
    
    return jsonify({
        'network': {'nodes': nodes, 'links': unique_links},
        'groups': coordinated_groups,
        'metrics': network_metrics
    })

@app.route('/api/ai_summary', methods=['GET'])
def get_ai_summary():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    if t5_tokenizer is None or t5_model is None:
        return jsonify({'error': 'Summarization model not available'}), 500
    
    query = request.args.get('query', '')
    
    # Filter data based on query
    filtered_data = data[
        data['selftext'].str.contains(query, case=False, na=False) |
        data['title'].str.contains(query, case=False, na=False)
    ]
    
    if len(filtered_data) == 0:
        return jsonify({'summary': f"No data found for query: {query}"})
    
    # Get time range
    min_date = filtered_data['created_utc'].min().strftime('%Y-%m-%d')
    max_date = filtered_data['created_utc'].max().strftime('%Y-%m-%d')
    
    # Get most active subreddits
    top_subreddits = filtered_data['subreddit'].value_counts().head(3).to_dict()
    
    # Get sample of titles for summarization (limit length for model)
    sample_titles = filtered_data['title'].sample(min(10, len(filtered_data))).tolist()
    titles_text = " ".join(sample_titles)
    
    # Create a summary context
    summary_context = f"From {min_date} to {max_date}, there were {len(filtered_data)} posts about '{query}'. "
    summary_context += f"The most active subreddits were {', '.join(top_subreddits.keys())}. "
    summary_context += f"Sample post titles: {titles_text[:500]}..."
    
    # Get top keywords
    try:
        vectorizer = CountVectorizer(stop_words='english', max_features=10)
        X = vectorizer.fit_transform(filtered_data['title'])
        feature_names = vectorizer.get_feature_names_out()
        freqs = X.sum(axis=0).A1
        sorted_indices = freqs.argsort()[::-1]
        top_keywords = [feature_names[i] for i in sorted_indices[:5]]
        summary_context += f" Top keywords: {', '.join(top_keywords)}."
    except:
        pass
    
    # Calculate engagement metrics
    if 'num_comments' in filtered_data.columns:
        avg_comments = filtered_data['num_comments'].mean()
        max_comments = filtered_data['num_comments'].max()
        summary_context += f" Average engagement: {avg_comments:.1f} comments per post, with the highest engagement at {max_comments} comments."
    
    # Generate summary using either Groq API or T5 model
    summary = ""
    try:
        if has_groq:
            # Use Groq API for enhanced analysis
            groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
            
            prompt = f"""
            Analyze the following social media data and provide deep insights in a well-structured format:
            
            {summary_context}
            
            FORMAT REQUIREMENTS:
            1. Structure your response using clear HTML formatting:
               - Use <h3> tags for main section headings (3-4 sections)
               - Use <h4> tags for subsection headings where needed
               - Use <p> tags for paragraphs
               - Use <ul> and <li> tags for bullet point lists
               - Use <ol> and <li> tags for numbered lists
            
            2. Include these specific sections:
               - "Key Findings" - Overall insights from the data (paragraph format)
               - "Conversation Patterns" - Trends and patterns in discussions (use bullet points)
               - "Audience Analysis" - Demographics and engagement patterns (paragraph format)
               - "Content Themes" - Major themes and narratives (use bullet points)
               
            3. Make sure each section has a clear heading and is visually distinct.
            
            4. Provide complete thoughts and don't truncate sentences.
            
            5. All content must be directly derived from the provided data - do not invent details.
            
            Return ONLY the HTML-formatted analysis without any additional explanations.
            """
            
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "qwen/qwen3-32b",  # Using LLaMA 3 model via Groq
                "messages": [
                    {"role": "system", "content": "You are an expert data analyst specializing in social media trends analysis. You provide well-structured, visually organized reports with appropriate headings and formatting."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 750  # Increased to ensure complete responses
            }
            
            response = requests.post(groq_api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                summary = result["choices"][0]["message"]["content"]
                
                # Check if the summary has proper HTML structure, if not add it
                if not summary.strip().startswith("<h3") and not summary.strip().startswith("<div"):
                    # Wrap in proper HTML structure
                    formatted_summary = "<div class='ai-summary-content'>"
                    
                    # Add default main heading if none exists
                    if "<h3" not in summary:
                        formatted_summary += f"<h3>Analysis of '{query}' Discussions</h3>"
                    
                    # Add the content, ensuring paragraphs are wrapped
                    if "<p>" not in summary:
                        # Split by double newlines and wrap in paragraph tags
                        paragraphs = summary.split("\n\n")
                        for p in paragraphs:
                            if p.strip():
                                formatted_summary += f"<p>{p.strip()}</p>"
                    else:
                        formatted_summary += summary
                    
                    formatted_summary += "</div>"
                    summary = formatted_summary
                else:
                    # If it already has HTML structure, just wrap it in a container div if needed
                    if not summary.strip().startswith("<div"):
                        summary = f"<div class='ai-summary-content'>{summary}</div>"
                
                print("Successfully generated analysis using Groq API")
            else:
                # Fallback to T5 if Groq API call fails
                print(f"Groq API call failed with status code {response.status_code}. Using T5 fallback.")
                summary = generate_structured_t5_summary(summary_context, t5_tokenizer, t5_model, query)
        else:
            # Fallback to T5 if Groq API is not available
            summary = generate_structured_t5_summary(summary_context, t5_tokenizer, t5_model, query)
    except Exception as e:
        # Fallback summary if model fails
        summary = f"""
        <div class='ai-summary-content'>
            <h3>Basic Summary of '{query}' Discussions</h3>
            <p>Found {len(filtered_data)} posts about '{query}' from {min_date} to {max_date}. 
            Most active in: {', '.join(list(top_subreddits.keys())[:3])}.</p>
            
            <h3>Key Keywords</h3>
            <ul>
        """
        if 'top_keywords' in locals():
            for keyword in top_keywords:
                summary += f"<li>{keyword}</li>"
        
        summary += """
            </ul>
            
            <h3>Engagement Statistics</h3>
            <p>
        """
        
        if 'avg_comments' in locals():
            summary += f"Average engagement: {avg_comments:.1f} comments per post."
            
        summary += "</p></div>"
    
    # Enhanced metrics for better insights
    metrics = {
        'total_posts': len(filtered_data),
        'time_range': f"{min_date} to {max_date}",
        'top_subreddits': top_subreddits,
        'unique_authors': filtered_data['author'].nunique(),
        'avg_comments': filtered_data['num_comments'].mean() if 'num_comments' in filtered_data.columns else 'N/A',
        'top_keywords': top_keywords if 'top_keywords' in locals() else [],
        'days_span': (pd.to_datetime(max_date) - pd.to_datetime(min_date)).days + 1,
        'posts_per_day': len(filtered_data) / ((pd.to_datetime(max_date) - pd.to_datetime(min_date)).days + 1),
        'top_authors': filtered_data['author'].value_counts().head(5).to_dict(),
    }
    
    # Calculate engagement trends over time if possible
    try:
        if 'num_comments' in filtered_data.columns:
            filtered_data['date'] = filtered_data['created_utc'].dt.date
            engagement_trend = filtered_data.groupby('date')['num_comments'].mean()
            # Convert date objects to strings before adding to dictionary
            engagement_trend = {str(date): float(value) for date, value in engagement_trend.items()}
            metrics['engagement_trend'] = engagement_trend
    except:
        pass
    
    return jsonify({
        'summary': summary,
        'metrics': metrics,
        'model_used': 'Groq API' if has_groq else 'Flan-T5-small'
    })

# Helper function to generate structured summaries with T5
def generate_structured_t5_summary(summary_context, tokenizer, model, query):
    """Generate a structured summary using the T5 model"""
    
    input_text = f"Analyze and summarize the following social media trends in detail: {summary_context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=500, min_length=200)
    t5_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Structure the T5 output with HTML formatting
    formatted_summary = f"""
    <div class='ai-summary-content'>
        <h3>Analysis of '{query}' Discussions</h3>
        <p>{t5_summary}</p>
        
        <h3>Key Insights</h3>
        <ul>
    """
    
    # Extract potential key points from the summary
    sentences = t5_summary.split('. ')
    key_points = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in ['significant', 'important', 'notable', 'key', 'trend', 'pattern']):
            key_points.append(sentence)
    
    # Limit to 3-5 key points
    key_points = key_points[:min(5, len(key_points))]
    if not key_points:
        key_points = sentences[:3]  # Take first 3 sentences if no key points found
    
    for point in key_points:
        if point.strip():
            formatted_summary += f"<li>{point.strip()}.</li>"
    
    formatted_summary += """
        </ul>
    </div>
    """
    
    return formatted_summary

@app.route('/api/common_words', methods=['GET'])
def get_common_words():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 50))
    
    # Filter data based on query
    filtered_data = data
    if query:
        filtered_data = data[
            data['selftext'].str.contains(query, case=False, na=False) |
            data['title'].str.contains(query, case=False, na=False)
        ]
    
    # Combine text data
    text_data = filtered_data['title'] + ' ' + filtered_data['selftext'].fillna('')
    
    # Tokenize and count words
    vectorizer = CountVectorizer(stop_words='english', max_features=limit)
    X = vectorizer.fit_transform(text_data)
    
    # Get word frequencies
    words = vectorizer.get_feature_names_out()
    freqs = X.sum(axis=0).A1
    
    # Sort by frequency
    sorted_indices = freqs.argsort()[::-1]
    result = [{'word': words[i], 'count': int(freqs[i])} for i in sorted_indices]
    
    return jsonify(result)

@app.route('/api/dynamic_description', methods=['GET'])
def get_dynamic_description():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    section = request.args.get('section', '')
    query = request.args.get('query', '')
    data_context = request.args.get('data_context', '{}')
    detail_level = request.args.get('detail_level', 'detailed')
    
    # Default response in case generation fails - now with proper HTML structure
    default_description = {
        "timeseries": """
            <div class='description-content'>
                <h4 class='section-heading'>Temporal Analysis</h4>
                <p>This visualization tracks post volume over time, revealing when discussions peaked, declined, or remained steady.</p>
                <ul>
                    <li>Identify conversation peaks and trends over time</li>
                    <li>Discover cyclical patterns in discussions</li>
                    <li>Correlate spikes with external events</li>
                </ul>
            </div>
        """,
        "network": """
            <div class='description-content'>
                <h4 class='section-heading'>User Interaction Network</h4>
                <p>This network graph maps connections between users based on their interactions. Nodes represent users and edges show interactions between them.</p>
                <ul>
                    <li>Identify central influencers and community structures</li>
                    <li>Visualize information flow patterns</li>
                    <li>Discover potential echo chambers or bridging users</li>
                </ul>
            </div>
        """,
        "topics": """
            <div class='description-content'>
                <h4 class='section-heading'>Topic Analysis</h4>
                <p>This analysis identifies distinct topics within the content using Latent Dirichlet Allocation (LDA).</p>
                <ul>
                    <li>Discover main themes and subtopics in the conversation</li>
                    <li>Analyze keyword distributions within topics</li>
                    <li>Track how topics evolve over time</li>
                </ul>
            </div>
        """,
        "coordinated": """
            <div class='description-content'>
                <h4 class='section-heading'>Coordinated Behavior Analysis</h4>
                <p>This analysis detects potentially coordinated posting behavior by identifying similar content posted within a short time window.</p>
                <ul>
                    <li>Identify patterns of synchronized content posting</li>
                    <li>Detect potential influence campaigns</li>
                    <li>Distinguish organic versus organized behavior</li>
                </ul>
            </div>
        """,
        "word_cloud": """
            <div class='description-content'>
                <h4 class='section-heading'>Term Frequency Visualization</h4>
                <p>The word cloud visualizes the most frequently occurring terms in the analyzed posts.</p>
                <ul>
                    <li>Quickly identify dominant terminology</li>
                    <li>Understand key concepts and vocabulary</li>
                    <li>Discover framing and narrative elements</li>
                </ul>
            </div>
        """,
        "contributors": """
            <div class='description-content'>
                <h4 class='section-heading'>Key Participant Analysis</h4>
                <p>This chart identifies the most active users who have posted content matching your search query.</p>
                <ul>
                    <li>Recognize dominant voices in the conversation</li>
                    <li>Assess the distribution of participation</li>
                    <li>Identify potential opinion leaders</li>
                </ul>
            </div>
        """,
        "overview": """
            <div class='description-content'>
                <h4 class='section-heading'>Comprehensive Overview</h4>
                <p>This section provides a high-level summary of the data analysis results.</p>
                <ul>
                    <li>Get quick insights into key metrics and trends</li>
                    <li>View aggregated statistics across all dimensions</li>
                    <li>Identify areas for deeper analysis</li>
                </ul>
            </div>
        """,
        "ai_insights": """
            <div class='description-content'>
                <h4 class='section-heading'>AI-Generated Insights</h4>
                <p>This section uses machine learning to generate human-readable insights from the data.</p>
                <ul>
                    <li>Receive automated analysis of complex patterns</li>
                    <li>Understand key trends without manual exploration</li>
                    <li>Discover hidden relationships in the data</li>
                </ul>
            </div>
        """,
        "data_story": """
            <div class='description-content'>
                <h4 class='section-heading'>Narrative Analysis</h4>
                <p>This synthesizes analyses into a cohesive narrative, highlighting key trends and patterns.</p>
                <ul>
                    <li>Follow the evolution of discussions over time</li>
                    <li>Connect related findings across different metrics</li>
                    <li>Understand the broader context of social media activity</li>
                </ul>
            </div>
        """,
        "semantic_map": """
            <div class='description-content'>
                <h4 class='section-heading'>Semantic Map Visualization</h4>
                <p>This visualization displays content in a 2D space where proximity represents semantic similarity. Posts with similar themes and language appear clustered together.</p>
                <ul>
                    <li>Discover thematic clusters and content relationships</li>
                    <li>Identify conceptually related posts across different authors</li>
                    <li>Visualize the semantic landscape of the conversation</li>
                    <li>Explore how different narratives relate to each other</li>
                </ul>
            </div>
        """
    }
    
    # Parse data context if provided
    context_data = {}
    try:
        import json
        context_data = json.loads(data_context)
    except:
        pass
    
    try:
        # Only proceed with Groq if API key is available
        if not has_groq or not GROQ_API_KEY:
            return jsonify({'description': default_description.get(section, """
                <div class='description-content'>
                    <h4 class='section-heading'>Data Analysis</h4>
                    <p>This visualization uncovers patterns, insights, and meaningful trends based on the conversation.</p>                                      
                </div>
            """)})
        
        # Enhanced section context with more analytical details
        section_context = {
            "timeseries": {
                "title": "time series visualization showing post frequency over time",
                "tech": "D3.js line and area charts with hoverable data points",
                "metrics": "post volume, trend direction, peak detection, temporal patterns",
                "insights": "conversation lifecycle, viral moments, correlation with external events, posting patterns"
            },
            "network": {
                "title": "network graph showing interactions between users",
                "tech": "force-directed graph with community detection via Louvain method",
                "metrics": "centrality, betweenness, clustering coefficient, modularity",
                "insights": "community structure, influence patterns, information flow, echo chambers"
            },
            "topics": {
                "title": "topic modeling analysis showing key themes in the content",
                "tech": "Latent Dirichlet Allocation (LDA) with coherence optimization",
                "metrics": "topic coherence, keyword distribution, temporal evolution, topic similarity",
                "insights": "narrative framing, emergent themes, semantic relationships, concept clusters"
            },
            "coordinated": {
                "title": "analysis of potentially coordinated posting behavior",
                "tech": "temporal-semantic clustering with TF-IDF and cosine similarity",
                "metrics": "temporal proximity, content similarity, coordination network density",
                "insights": "organized behavior patterns, information campaigns, authentic vs. inauthentic behavior"
            },
            "word_cloud": {
                "title": "word cloud visualization of frequent terms",
                "tech": "frequency-weighted layout with visual encoding of prominence",
                "metrics": "term frequency, inverse document frequency, relative prominence",
                "insights": "terminology patterns, vocabulary choices, discourse framing, key concepts"
            },
            "contributors": {
                "title": "chart of top contributors/authors",
                "tech": "comparative visualization of posting frequency by author",
                "metrics": "post volume, author distribution, participation inequality",
                "insights": "voice dominance, conversation drivers, participation patterns"
            },
            "overview": {
                "title": "dashboard overview section with key metrics",
                "tech": "multi-dimensional summary statistics with visual highlighting",
                "metrics": "post volume, unique authors, engagement, temporal span",
                "insights": "conversation scale, community engagement, topic resonance"
            },
            "ai_insights": {
                "title": "AI-generated summary of insights",
                "tech": "large language model analysis of aggregated metrics and content",
                "metrics": "semantic patterns, trend analysis, anomaly detection",
                "insights": "narrative interpretation, hidden patterns, context from broader knowledge"
            },
            "data_story": {
                "title": "narrative that connects different analyses into a cohesive story",
                "tech": "multi-faceted data synthesis with temporal and thematic organization",
                "metrics": "pattern correlation, event sequencing, thematic connection",
                "insights": "holistic understanding, causal relationships, narrative arc"
            },
            "semantic_map": {
                "title": "semantic map visualization of content",
                "tech": "dimensionality reduction (UMAP) and clustering (HDBSCAN)",
                "metrics": "semantic similarity, thematic clusters, content relationships",
                "insights": "conceptual connections, narrative landscapes, thematic exploration"
            }
        }
        
        section_info = section_context.get(section, {
            "title": "a data visualization",
            "tech": "interactive data visualization",
            "metrics": "relevant statistical measures",
            "insights": "patterns and relationships in the data"
        })
        
        # Adjust the verbosity based on detail level
        detail_settings = {
            "basic": {
                "description": "Write a brief explanation (2-3 sections)",
                "max_tokens": 750,
                "sections": 1
            },
            "detailed": {
                "description": "Write a comprehensive explanation (4-5 sections with specific details)",
                "max_tokens": 1200,
                "sections": 2
            },
            "expert": {
                "description": "Write an in-depth analytical explanation (5+ sections with technical details)",
                "max_tokens": 1800,
                "sections": 3
            }
        }
        
        detail_config = detail_settings.get(detail_level, detail_settings["detailed"])
        
        # Build enhanced contextual data based on the section
        enhanced_context = ""
        if context_data:
            if section == "timeseries" and "dataPoints" in context_data:
                filtered_data = data[data['selftext'].str.contains(query, case=False, na=False) | 
                                    data['title'].str.contains(query, case=False, na=False)]
                if len(filtered_data) > 0:
                    date_range = f"{filtered_data['created_utc'].min().strftime('%Y-%m-%d')} to {filtered_data['created_utc'].max().strftime('%Y-%m-%d')}"
                    peak_day = filtered_data.groupby(filtered_data['created_utc'].dt.date).size().idxmax()
                    peak_count = filtered_data.groupby(filtered_data['created_utc'].dt.date).size().max()
                    enhanced_context = f"The data spans from {date_range} with {len(filtered_data)} total posts. The peak day was {peak_day} with {peak_count} posts."
            
            elif section == "topics" and "topicCount" in context_data:
                topic_count = context_data.get("topicCount", 5)
                # Try to get top topics from data for richer context
                try:
                    topic_data = pd.read_json(f"/api/topics?n_topics={topic_count}&query={query}")
                    if 'topics' in topic_data and len(topic_data['topics']) > 0:
                        top_topic_words = ', '.join(topic_data['topics'][0]['top_words'][:5])
                        enhanced_context = f"Analysis found {topic_count} distinct topics. The most prominent topic contains these key terms: {top_topic_words}."
                except:
                    enhanced_context = f"Analysis is configured to find {topic_count} distinct topics in the content."
            
            elif section == "network":
                # Try to enrich with network metrics
                try:
                    node_count = context_data.get("nodeCount", 0)
                    filtered_data = data[data['selftext'].str.contains(query, case=False, na=False) | 
                                        data['title'].str.contains(query, case=False, na=False)]
                    author_count = filtered_data['author'].nunique()
                    enhanced_context = f"The network visualization shows interactions between {node_count} users out of {author_count} total authors in the dataset."
                except:
                    pass
            
            elif section == "semantic_map":
                # Add context for semantic map with comprehensive details
                try:
                                        # Get detailed information about the visualization parameters and data                    # Extract actual data from context_data with proper key paths                    total_posts = context_data.get("points", [])                    total_posts = len(total_posts) if isinstance(total_posts, list) else 0                                        # Get cluster information                    topics = context_data.get("topics", [])                    cluster_count = len(topics) if isinstance(topics, list) else 0                                        # Get UMAP parameters                    umap_params = context_data.get("umap_params", {})                    n_neighbors = umap_params.get("n_neighbors", 15) if isinstance(umap_params, dict) else 15                    min_dist = umap_params.get("min_dist", 0.1) if isinstance(umap_params, dict) else 0.1                                        # Get max_points used for visualization (from request args or default)                    max_points = context_data.get("max_points", 500)
                    
                    # Get data related to the user's query to provide specific context
                    filtered_data = data[data['selftext'].str.contains(query, case=False, na=False) | 
                                       data['title'].str.contains(query, case=False, na=False)]
                    
                    # Count unique authors and communities (subreddits) if available
                    unique_authors = filtered_data['author'].nunique() if 'author' in filtered_data.columns else 0
                    
                    # Get information about communities if available
                    community_info = ""
                    if 'subreddit' in filtered_data.columns:
                        top_communities = filtered_data['subreddit'].value_counts().head(3)
                        if not top_communities.empty:
                            communities_list = ", ".join([f"{name} ({count} posts)" for name, count in top_communities.items()])
                            community_info = f" Content is primarily from these communities: {communities_list}."
                    
                    # Extract key topics if available in context_data
                    topics_info = ""
                    if "topics" in context_data and context_data["topics"]:
                        topics = context_data["topics"]
                        if isinstance(topics, list) and len(topics) > 0:
                            top_terms = []
                            for topic in topics[:3]:  # Get top 3 topics
                                if "terms" in topic and topic["terms"]:
                                    top_terms.append(", ".join(topic["terms"][:5]))  # Top 5 terms per topic
                            
                            if top_terms:
                                topics_str = "; ".join([f"Topic {i+1}: {terms}" for i, terms in enumerate(top_terms)])
                                topics_info = f" Major thematic clusters include: {topics_str}."
                    
                    # Build comprehensive context with all collected information
                    enhanced_context = f"The semantic map visualizes {total_posts} posts about '{query}' from the dataset, grouped into {cluster_count} thematic clusters based on their semantic similarity. Posts that discuss similar themes appear closer together in the 2D space.{community_info}{topics_info} The visualization uses UMAP dimensionality reduction (n_neighbors={n_neighbors}, min_dist={min_dist}) to represent high-dimensional text relationships in two dimensions while preserving semantic relationships."
                    
                    if unique_authors > 0:
                        enhanced_context += f" The displayed content was created by {unique_authors} unique authors."
                        
                except Exception as e:
                    # If there was an error, provide more general but still informative context
                    enhanced_context = f"The semantic map visualizes posts about '{query}' in a 2D space where proximity indicates semantic similarity. Posts with similar themes and language appear clustered together, allowing you to explore the conceptual landscape of the conversation."
            
            # Add general context if no specific enhancement was added
            if not enhanced_context and context_data:
                enhanced_context = f"The visualization is based on these metrics: {json.dumps(context_data)}. "
        
        # Build prompt with enhanced context and more detailed requirements
        section_type = section_info["title"]
        
        prompt = f"""
        {detail_config["description"]} for {section_type} in a social media analysis dashboard.
        
        QUERY CONTEXT: The user is exploring data about "{query}".
        
        DATA CONTEXT: {enhanced_context}
        
        TECHNICAL DETAILS: This visualization uses {section_info["tech"]} to analyze {section_info["metrics"]}.
        
        RESPONSE FORMAT REQUIREMENTS:
        - Format your response in HTML with proper structure:
          - Include a main heading (<h4>) for the visualization
          - Use subheadings (<h5>) for each major section
          - Use paragraphs (<p>) for explanatory text
          - Use bullet lists (<ul>/<li>) or numbered lists (<ol>/<li>) as appropriate
          - Wrap everything in a <div class='description-content'> container
        - Include 3-4 distinct sections with meaningful headings (not generic "Key Insights")
        - First section should be an informative introduction to what the visualization shows
        - Include at least one section with bullet points highlighting key patterns
        - If relevant, include a section called "Interpretation Guide" with tips for reading the visualization
        
        Your description should:
        1. Explain what this visualization shows specifically for the query "{query}"
        2. Highlight the key metrics and what patterns they might reveal
        3. Discuss potential insights that could be derived from this visualization
        4. Explain why these insights are valuable for understanding conversations about "{query}"
        5. Use concrete details where possible rather than generic statements
        
        For expert level descriptions, include technical interpretation guidance and explain analytical considerations.
        
        Return ONLY the HTML content without any markdown formatting or meta-commentary.
        """
        
        # Call Groq API for the description with increased token limit
        groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen/qwen3-32b",
            "messages": [
                {"role": "system", "content": "You are a data visualization expert specializing in social media analytics. You explain complex data patterns in clear, insightful language that highlights meaningful insights. Output in HTML format with proper structure."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": detail_config["max_tokens"]
        }
        
        response = requests.post(groq_api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            description = result["choices"][0]["message"]["content"].strip()
            
            # Check if description contains HTML structure
            if not description.strip().startswith("<div") and not description.strip().startswith("<h"):
                # If not properly formatted as HTML, wrap it in default structure
                description = f"""
                <div class='description-content'>
                    <h4 class='section-heading'>Analysis Results</h4>
                    <p>{description}</p>
                </div>
                """
            
            return jsonify({'description': description})
        else:
            # Fallback to default descriptions
            return jsonify({'description': default_description.get(section, """
                <div class='description-content'>
                    <h4 class='section-heading'>Data Analysis</h4>
                    <p>This visualization uncovers patterns, insights, and meaningful trends based on the conversation.</p>
                </div>
            """)})
            
    except Exception as e:
        print(f"Error generating dynamic description: {str(e)}")
        return jsonify({
            'description': default_description.get(section, """
                <div class='description-content'>
                    <h4 class='section-heading'>Data Analysis</h4>
                    <p>This visualization uncovers patterns, insights, and meaningful trends based on the conversation.</p>
                </div>
            """),
            'error': str(e)
        })

@app.route('/api/semantic_map', methods=['GET'])
def get_semantic_map():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    max_points = min(int(request.args.get('max_points', 500)), 2000)  # Cap at 2000 for performance
    n_neighbors = int(request.args.get('n_neighbors', 15))
    min_dist = float(request.args.get('min_dist', 0.1))
    
    try:
        # Filter data based on query
        filtered_data = data
        if query:
            filtered_data = data[
                data['selftext'].str.contains(query, case=False, na=False) |
                data['title'].str.contains(query, case=False, na=False)
            ]
        
        if len(filtered_data) == 0:
            return jsonify({'error': 'No data found matching the query'}), 404
        
        # If we have too many posts, sample to improve performance
        if len(filtered_data) > max_points:
            filtered_data = filtered_data.sample(max_points, random_state=42)
        
        # Prepare text for embedding - combine title and selftext
        text_data = []
        for idx, row in filtered_data.iterrows():
            title = row['title'] if isinstance(row['title'], str) else ""
            selftext = row['selftext'] if isinstance(row['selftext'], str) and pd.notna(row['selftext']) else ""
            # Limit text length to avoid extremely long documents
            combined_text = (title + " " + selftext[:500]).strip()
            text_data.append(combined_text)
        
        # Load the sentence transformer model on demand
        global semantic_model
        if semantic_model is None:
            try:
                # Use a smaller, faster model for embeddings
                semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded semantic embedding model successfully")
            except Exception as e:
                print(f"Error loading semantic model: {e}")
                return jsonify({'error': 'Failed to load semantic model'}), 500
        
        # Generate embeddings
        embeddings = semantic_model.encode(text_data, show_progress_bar=True)
        
        # Apply UMAP for dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Prepare result points with metadata
        points = []
        for i, (idx, row) in enumerate(filtered_data.iterrows()):
            points.append({
                'x': float(reduced_embeddings[i, 0]),
                'y': float(reduced_embeddings[i, 1]),
                'id': str(i),
                'title': row['title'],
                'author': row['author'],
                'subreddit': row.get('subreddit', ''),
                'created_utc': row['created_utc'].isoformat(),
                'num_comments': int(row.get('num_comments', 0)),
                'score': int(row.get('score', 0)),
                'preview_text': (row.get('selftext', '')[:100] + '...' if len(row.get('selftext', '') or '') > 100 
                                else row.get('selftext', ''))
            })
        
        # Try to extract topics from clusters of points
        topics = []
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Use K-means to find clusters in the embedding space
            n_clusters = min(10, max(3, len(points) // 50))  # Dynamic number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(reduced_embeddings)
            
            # Add cluster assignments to points
            for i, point in enumerate(points):
                point['cluster'] = int(clusters[i])
            
            # Extract keywords for each cluster
            for cluster_id in range(n_clusters):
                cluster_texts = [text_data[i] for i in range(len(text_data)) if clusters[i] == cluster_id]
                
                if cluster_texts:
                    # Use TF-IDF to find distinctive words for this cluster
                    tfidf = TfidfVectorizer(max_features=200, stop_words='english')
                    try:
                        cluster_tfidf = tfidf.fit_transform(cluster_texts)
                        
                        # Get top terms
                        feature_names = tfidf.get_feature_names_out()
                        top_indices = np.argsort(np.asarray(cluster_tfidf.mean(axis=0)).flatten())[-7:]
                        top_terms = [feature_names[i] for i in top_indices]
                        
                        # Calculate cluster center
                        cluster_points = reduced_embeddings[clusters == cluster_id]
                        center_x = float(np.mean(cluster_points[:, 0]))
                        center_y = float(np.mean(cluster_points[:, 1]))
                        
                        topics.append({
                            'id': int(cluster_id),
                            'terms': top_terms,
                            'size': int(np.sum(clusters == cluster_id)),
                            'center_x': center_x,
                            'center_y': center_y
                        })
                    except:
                        # Skip if TF-IDF fails for some reason
                        pass
        except Exception as e:
            print(f"Error extracting topics: {e}")
            # Continue without topics if clustering fails
        
        return jsonify({
            'points': points,
            'topics': topics,
            'total_posts': len(points),
            'umap_params': {
                'n_neighbors': n_neighbors,
                'min_dist': min_dist
            }
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error generating semantic map: {str(e)}'}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot_response():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    # Get request data
    request_data = request.get_json()
    if not request_data or 'query' not in request_data:
        return jsonify({'error': 'Missing query parameter'}), 400
    
    user_query = request_data['query']
    chat_history = request_data.get('history', [])
    
    try:
        # Process the user query to determine intent and extract key terms
        query_keywords = []
        intent = "general"
        
        # Basic keyword extraction and intent classification
        lower_query = user_query.lower()
        
        # Check for trend-related queries
        if any(word in lower_query for word in ['trend', 'change', 'over time', 'evolve', 'increase', 'decrease', 'growth', 'decline', 'pattern']):
            intent = "trend"
        # Check for topic-related queries
        elif any(word in lower_query for word in ['topic', 'theme', 'about', 'discuss', 'talking about', 'conversation', 'subject', 'matter']):
            intent = "topic"
        # Check for community-related queries
        elif any(word in lower_query for word in ['community', 'subreddit', 'group', 'people', 'user', 'member', 'author', 'contributor']):
            intent = "community"
        # Check for comparison queries
        elif any(word in lower_query for word in ['compare', 'difference', 'versus', 'vs', 'similarities', 'contrast', 'different', 'same']):
            intent = "comparison"
        # Check for network analysis queries
        elif any(word in lower_query for word in ['network', 'connections', 'relationship', 'interact']):
            intent = "network"
        # Check for insight-related queries
        elif any(word in lower_query for word in ['insight', 'understand', 'analyze', 'summary']):
            intent = "insights"
        # Check for coordination-related queries
        elif any(word in lower_query for word in ['coordinate', 'coordinated', 'campaign', 'organized']):
            intent = "coordinated"
        
        # Extract potential search terms - this is a basic implementation
        # For a more robust solution, consider using NLP libraries like spaCy
        import re
        # Look for quoted text as exact search terms
        quoted_terms = re.findall(r'"([^"]*)"', user_query)
        # Remove quoted text from query
        clean_query = re.sub(r'"[^"]*"', '', user_query)
        # Extract remaining potential keywords (words of 4+ chars)
        additional_keywords = [word for word in re.findall(r'\b[a-zA-Z]{4,}\b', clean_query) 
                            if word.lower() not in ['what', 'when', 'where', 'which', 'about', 'tell', 
                                                 'show', 'give', 'find', 'many', 'much', 'most',
                                                'least', 'more', 'less', 'data', 'query', 'information']]
        
        # Combine quoted terms and additional keywords
        query_keywords = quoted_terms + additional_keywords
        
        # Use the extracted keywords for searching the dataset
        search_query = ' '.join(query_keywords) if query_keywords else user_query
        
        # Filter data based on extracted keywords
        filtered_data = data
        if query_keywords:
            query_pattern = '|'.join(query_keywords)
            filtered_data = data[
                data['selftext'].str.contains(query_pattern, case=False, na=False) |
                data['title'].str.contains(query_pattern, case=False, na=False)
            ]
        
        # If no data matched the keywords, use a broader approach
        if len(filtered_data) < 5 and query_keywords:
            # Try with just the first keyword for broader results
            if query_keywords:
                filtered_data = data[
                    data['selftext'].str.contains(query_keywords[0], case=False, na=False) |
                    data['title'].str.contains(query_keywords[0], case=False, na=False)
                ]
        
        # If still no substantial results, return a message about insufficient data
        if len(filtered_data) < 3:
            html_response = f"""
            <div class='chatbot-response'>
                <h3>Search Results</h3>
                <p>I couldn't find enough data about <strong>'{search_query}'</strong> in the dataset. This could be because:</p>
                <ul>
                    <li>The topic might not be extensively discussed in this dataset</li>
                    <li>You might be using terms that differ from how people discuss this topic</li>
                    <li>The topic may be referenced using different terminology</li>
                </ul>
                
                <h4>Suggestions</h4>
                <p>Try these approaches:</p>
                <ol>
                    <li>Use broader or more general search terms</li>
                    <li>Check spelling of specific terms or names</li>
                    <li>Try related topics or synonyms</li>
                    <li>Remove specific qualifiers that might be limiting results</li>
                </ol>
            </div>
            """
            
            return jsonify({
                'response': html_response,
                'data_points': 0,
                'suggestions': ['Try a broader topic', 'Check your spelling', 'Use fewer specific terms']
            })
        
        # Gather metrics and insights based on the filtered data
        metrics = {
            'total_posts': len(filtered_data),
            'unique_authors': filtered_data['author'].nunique(),
            'time_range': {
                'start': filtered_data['created_utc'].min().isoformat(),
                'end': filtered_data['created_utc'].max().isoformat()
            },
            'top_subreddits': filtered_data['subreddit'].value_counts().head(3).to_dict()
        }
        
        # Add engagement metrics if available
        if 'num_comments' in filtered_data.columns:
            metrics['avg_comments'] = float(filtered_data['num_comments'].mean())
            metrics['max_comments'] = int(filtered_data['num_comments'].max())
        
        # Extract top keywords for context
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            text_data = filtered_data['title'] + ' ' + filtered_data['selftext'].fillna('')
            vectorizer = CountVectorizer(stop_words='english', max_features=10)
            X = vectorizer.fit_transform(text_data)
            feature_names = vectorizer.get_feature_names_out()
            freqs = X.sum(axis=0).A1
            sorted_indices = freqs.argsort()[::-1]
            metrics['top_keywords'] = [feature_names[i] for i in sorted_indices[:7]]
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            metrics['top_keywords'] = []
        
        # Prepare for trend analysis if needed
        if intent == "trend":
            try:
                # Create a copy of the dataframe to avoid SettingWithCopyWarning
                trend_df = filtered_data.copy()
                
                # Import numpy here to ensure it's available in this scope
                import numpy as np
                
                # Group by date and count posts
                trend_df['date'] = trend_df['created_utc'].dt.date
                time_series = trend_df.groupby('date').size()
                
                # Find peaks and trends
                dates = [str(date) for date in time_series.index]
                counts = time_series.values.tolist()
                
                if len(dates) > 0 and len(counts) > 0:
                    # Add time series data to metrics
                    metrics['time_series'] = {
                        'dates': dates,
                        'counts': counts
                    }
                    
                    # Identify peak date
                    peak_idx = np.argmax(counts)
                    metrics['peak_date'] = dates[peak_idx]
                    metrics['peak_count'] = int(counts[peak_idx])
                    
                    # Calculate trend (simple linear regression)
                    if len(counts) > 3:
                        from scipy import stats
                        x = np.arange(len(counts))
                        slope, _, _, _, _ = stats.linregress(x, counts)
                        metrics['trend'] = {
                            'direction': 'increasing' if slope > 0 else 'decreasing',
                            'magnitude': abs(slope)
                        }
            except Exception as e:
                print(f"Error in trend analysis: {e}")
        
        # For topic intent, extract topics
        if intent == "topic":
            try:
                # Use LDA for topic modeling
                from sklearn.decomposition import LatentDirichletAllocation
                from sklearn.feature_extraction.text import CountVectorizer
                
                # Prepare text data
                text_data = filtered_data['title'] + ' ' + filtered_data['selftext'].fillna('')
                
                # Create document-term matrix
                vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
                X = vectorizer.fit_transform(text_data)
                
                # Apply LDA
                n_topics = min(5, len(filtered_data) // 10) if len(filtered_data) > 50 else 3
                lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                lda.fit(X)
                
                # Get top words for each topic
                feature_names = vectorizer.get_feature_names_out()
                topics_data = []
                
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[:-11:-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics_data.append({
                        'id': topic_idx,
                        'words': top_words
                    })
                
                metrics['topics'] = topics_data
            except Exception as e:
                print(f"Error in topic analysis: {e}")
        
        # For community intent, analyze subreddit distribution
        if intent == "community":
            try:
                communities = filtered_data['subreddit'].value_counts().head(10).to_dict()
                metrics['communities'] = communities
                
                # Calculate diversity metrics
                total = sum(communities.values())
                # Shannon entropy as a measure of diversity
                from math import log2
                entropy = -sum((count/total) * log2(count/total) for count in communities.values())
                metrics['community_diversity'] = entropy
            except Exception as e:
                print(f"Error in community analysis: {e}")
        
        # For network intent, get quick network statistics
        if intent == "network":
            try:
                # Count interactions between authors
                author_interactions = {}
                
                # Create a temporary copy for processing
                interaction_df = filtered_data.copy()
                
                # Check if we have parent_id information
                if 'parent_id' in interaction_df.columns:
                    # Count direct replies
                    reply_count = interaction_df['parent_id'].notna().sum()
                    metrics['reply_count'] = reply_count
                    
                    # Get top authors by network centrality (simplified as post count)
                    central_authors = interaction_df['author'].value_counts().head(5).to_dict()
                    metrics['central_authors'] = central_authors
            except Exception as e:
                print(f"Error in network analysis: {e}")
                
        # Generate response using Gemini API if available, otherwise use a template-based approach
        response_text = ""
        visualization_suggestions = []
        
        if has_gemini and GEMINI_API_KEY:
            # Build a comprehensive context object for the LLM
            context_obj = {
                "query": user_query,
                "search_query": search_query,
                "intent": intent,
                "metrics": metrics,
                "keywords": query_keywords
            }
            
            # Convert to string representation for the prompt
            import json
            context_str = json.dumps(context_obj, indent=2)
            
            # Build a more enhanced prompt specifically for Gemini
            gemini_prompt = f"""
            You are an expert social media data analyst. Answer the following question based on the data provided, with your response formatted in well-structured HTML.

            USER QUERY:
            "{user_query}"

            DATA CONTEXT:
            {context_str}

            REQUIREMENTS:
            1. Structure your response using proper HTML formatting:
               - Use <h3> tags for main section headings (1-2 main sections)
               - Use <h4> tags for subsection headings (2-4 subsections)
               - Use <p> tags for paragraphs with detailed explanations
               - Use <ul> and <li> tags for bullet point lists when providing multiple insights
               - Use <ol> and <li> tags for numbered lists when sequence matters
               - Use <strong> tags to highlight key numbers and important findings
               - Wrap everything in a <div class='chatbot-response'> container

            2. Include these elements in your response:
               - A direct answer to the user's question that is comprehensive and detailed
               - At least 3-5 specific insights backed by the data metrics
               - Multiple data points and statistics from the metrics (use exact numbers) 
               - Analysis of patterns, trends, or relationships in the data
               - Comparisons between different aspects of the data when relevant

            3. Make your response detailed and informative (aim for 300-500 words)
               - Ensure your response is comprehensive and answers all aspects of the query
               - Include nuanced analysis that considers multiple perspectives
               - Be specific rather than general in your observations
            
            4. BE PRECISE: Use exact numbers from the data when available. Include dates, counts, percentages, and other metrics.
            
            Do NOT include any redirections or navigation links to other dashboard sections.
            
            IMPORTANT: Do not wrap your HTML in code blocks or markdown formatting. Provide the HTML directly without any ```html or ``` tags surrounding it.
            Do not prefix your response with any explanation or introduction. Start directly with the <div> tag.
            
            Return ONLY the HTML content without any additional explanation, markdown, or code blocks.
            """
            
            try:
                # Generate response using Gemini API
                gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                gemini_response = gemini_model.generate_content(gemini_prompt)
                
                if gemini_response:
                    response_text = gemini_response.text.strip()
                    print(f"Raw Gemini response: {response_text[:100]}...")  # Print first 100 chars for debugging
                    
                    # Clean up the response - remove any leading/trailing quotes or backticks
                    # that might be causing the raw HTML to be displayed
                    response_text = response_text.replace('```html', '').replace('```', '')
                    response_text = response_text.strip('"\'`')
                    print(f"Cleaned response: {response_text[:100]}...")  # Print cleaned response
                    
                    # Verify we got HTML and not plain text
                    if not response_text.startswith("<div") and not response_text.startswith("<h"):
                        print("Response doesn't start with HTML tags, wrapping in div")
                        # Wrap in proper HTML if the model didn't provide it
                        response_text = f"""
                        <div class='chatbot-response'>
                            <h3>Analysis Results</h3>
                            <p>{response_text}</p>
                        </div>
                        """
                else:
                    # Fallback to template response if API fails
                    print(f"Gemini API error: No response received")
                    response_text = generate_template_response(intent, metrics, search_query)
                    # Add HTML structure to template response
                    response_text = enhance_response_with_html_no_redirects(response_text, intent)
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                response_text = generate_template_response(intent, metrics, search_query)
                # Add HTML structure to template response
                response_text = enhance_response_with_html_no_redirects(response_text, intent)
        else:
            # Use template-based response if Gemini API is not available
            response_text = generate_template_response(intent, metrics, search_query)
            # Add HTML structure to template response
            response_text = enhance_response_with_html_no_redirects(response_text, intent)
        
        # Determine visualization suggestions based on intent
        if intent == "trend":
            visualization_suggestions = [
                {
                    'type': 'timeseries',
                    'title': 'Time Series Analysis',
                    'description': 'View post frequency over time to see trends and patterns'
                },
                {
                    'type': 'events',
                    'title': 'Event Correlation',
                    'description': 'See how real-world events correlate with conversation peaks'
                }
            ]
        elif intent == "topic":
            visualization_suggestions = [
                {
                    'type': 'topics',
                    'title': 'Topic Analysis',
                    'description': 'Explore the main topics and themes in the data'
                },
                {
                    'type': 'word_cloud',
                    'title': 'Word Cloud', 
                    'description': 'See the most frequent terms used in discussions'
                },
                {
                    'type': 'semantic_map',
                    'title': 'Semantic Map', 
                    'description': 'See how posts are related to each other in semantic space'
                }
            ]
        elif intent == "community":
            visualization_suggestions = [
                {
                    'type': 'contributors',
                    'title': 'Top Contributors',
                    'description': 'See which users are most active in discussions'
                },
                {
                    'type': 'network',
                    'title': 'Network Analysis',
                    'description': 'Explore connections between users discussing this topic'
                }
            ]
        elif intent == "network":
            visualization_suggestions = [
                {
                    'type': 'network',
                    'title': 'Network Analysis',
                    'description': 'View the interaction network between authors'
                },
                {
                    'type': 'coordinated',
                    'title': 'Coordinated Behavior',
                    'description': 'Detect potentially coordinated posting patterns'
                }
            ]
        elif intent == "insights":
            visualization_suggestions = [
                {
                    'type': 'ai_insights',
                    'title': 'AI Insights',
                    'description': 'Get AI-powered analysis of the conversation'
                },
                {
                    'type': 'data_story',
                    'title': 'Data Story',
                    'description': 'View a narrative analysis of the discussion'
                }
            ]
        elif intent == "coordinated":
            visualization_suggestions = [
                {
                    'type': 'coordinated',
                    'title': 'Coordinated Behavior',
                    'description': 'Analyze potential coordination in posting behavior'
                },
                {
                    'type': 'network',
                    'title': 'Network Analysis',
                    'description': 'See connections between potentially coordinated users'
                }
            ]
        else:
            # Generic/general intent
            visualization_suggestions = [
                {
                    'type': 'overview',
                    'title': 'Data Overview',
                    'description': 'See a comprehensive summary of all available data'
                },
                {
                    'type': 'ai_insights',
                    'title': 'AI Insights',
                    'description': 'Get AI-powered analysis of the conversation'
                }
            ]
        
        # Return the enhanced response with metrics and visualizations
        return jsonify({
            'response': response_text,
            'metrics': metrics,
            'data_points': len(filtered_data),
            'visualization_suggestions': visualization_suggestions,
            'search_terms': query_keywords,
            'intent': intent
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': f'Error processing chatbot query: {str(e)}',
            'response': 'I encountered an error while processing your question. Please try rephrasing or asking something else.'
        }), 500

def generate_template_response(intent, metrics, search_query):
    """Helper function to generate template-based responses when LLM is not available"""
    if intent == "trend":
        if 'trend' in metrics:
            trend_direction = metrics['trend']['direction']
            peak_date = metrics.get('peak_date', 'the analyzed period')
            peak_count = metrics.get('peak_count', 'multiple')
            
            return f"There is a {trend_direction} trend in discussions about '{search_query}'. The conversation peaked on {peak_date} with {peak_count} posts. Overall, I found {metrics['total_posts']} posts from {metrics['unique_authors']} unique authors across {len(metrics['top_subreddits'])} subreddits."
        else:
            return f"I analyzed {metrics['total_posts']} posts about '{search_query}' from {metrics['time_range']['start'][:10]} to {metrics['time_range']['end'][:10]}. These posts came from {metrics['unique_authors']} unique authors across subreddits like {', '.join(list(metrics['top_subreddits'].keys())[:2])}."
    
    elif intent == "topic":
        if 'topics' in metrics and metrics['topics']:
            topics_text = []
            for topic in metrics['topics'][:2]:
                topics_text.append(f"{', '.join(topic['words'][:5])}")
            
            return f"The main topics related to '{search_query}' include: {'; '.join(topics_text)}. These themes appeared across {metrics['total_posts']} posts from {metrics['unique_authors']} authors."
        else:
            return f"I found {metrics['total_posts']} posts about '{search_query}'. Key terms include {', '.join(metrics.get('top_keywords', [])[:5])}."
    
    elif intent == "community":
        if 'communities' in metrics:
            top_communities = list(metrics['communities'].keys())[:3]
            return f"The main communities discussing '{search_query}' are: {', '.join(top_communities)}. I analyzed {metrics['total_posts']} posts from {metrics['unique_authors']} unique authors."
        else:
            return f"Discussions about '{search_query}' happen primarily in {', '.join(list(metrics['top_subreddits'].keys())[:3])}. I found {metrics['total_posts']} posts from {metrics['unique_authors']} authors."
    
    # Default/general response
    return f"I analyzed {metrics['total_posts']} posts about '{search_query}' from {metrics['unique_authors']} authors. Key terms include {', '.join(metrics.get('top_keywords', [])[:5])}{'. Most discussions occurred in ' + ', '.join(list(metrics['top_subreddits'].keys())[:2]) if metrics['top_subreddits'] else ''}."

def enhance_response_with_html(text, intent):
    """Enhance a plain text response with HTML structure and relevant dashboard links"""
    
    # Create section redirections based on intent
    redirections = []
    
    if intent == "trend":
        redirections.append('<a href="#timeseries" class="dashboard-link" data-section="timeseries">View Time Series Analysis</a>')
        redirections.append('<a href="#events" class="dashboard-link" data-section="events">View Event Correlation</a>')
    elif intent == "topic":
        redirections.append('<a href="#topics" class="dashboard-link" data-section="topics">View Topic Analysis</a>')
        redirections.append('<a href="#word_cloud" class="dashboard-link" data-section="word_cloud">View Word Cloud</a>')
    elif intent == "community":
        redirections.append('<a href="#contributors" class="dashboard-link" data-section="contributors">View Top Contributors</a>')
        redirections.append('<a href="#network" class="dashboard-link" data-section="network">View Network Analysis</a>')
    elif intent == "network":
        redirections.append('<a href="#network" class="dashboard-link" data-section="network">View Network Analysis</a>')
    elif intent == "insights":
        redirections.append('<a href="#ai_insights" class="dashboard-link" data-section="ai_insights">View AI Insights</a>')
    elif intent == "coordinated":
        redirections.append('<a href="#coordinated" class="dashboard-link" data-section="coordinated">View Coordinated Behavior Analysis</a>')
    elif intent == "engagement":
        redirections.append('<a href="#contributors" class="dashboard-link" data-section="contributors">View Top Contributors</a>')
        redirections.append('<a href="#timeseries" class="dashboard-link" data-section="timeseries">View Engagement Over Time</a>')
    elif intent == "sentiment":
        redirections.append('<a href="#topics" class="dashboard-link" data-section="topics">View Topic Analysis</a>')
        redirections.append('<a href="#ai_insights" class="dashboard-link" data-section="ai_insights">View AI-Generated Insights</a>')
    elif intent == "time_specific":
        redirections.append('<a href="#timeseries" class="dashboard-link" data-section="timeseries">View Time Series Analysis</a>')
        redirections.append('<a href="#events" class="dashboard-link" data-section="events">View Event Correlation</a>')
    else:
        # General/default redirections
        redirections.append('<a href="#overview" class="dashboard-link" data-section="overview">View Data Overview</a>')
        redirections.append('<a href="#ai_insights" class="dashboard-link" data-section="ai_insights">View AI Insights</a>')
    
    # Build HTML response
    html = f"""
    <div class='chatbot-response'>
        <h3>Analysis Results</h3>
        <p>{text}</p>
        
        <h4>Explore Further</h4>
        <ul>
            <li>{redirections[0]}</li>
    """
    
    # Add additional redirections
    for link in redirections[1:]:
        html += f"<li>{link}</li>"
    
    html += """
        </ul>
    </div>
    """
    
    return html

def enhance_response_with_html_no_redirects(text, intent):
    """Enhance a plain text response with HTML structure without any redirections or links"""
    
    # Build HTML response without any redirections
    html = f"""
    <div class='chatbot-response'>
        <h3>Analysis Results</h3>
        <p>{text}</p>
    </div>
    """
    
    return html

# Event-related functionality
# Dictionary to store manually curated event data
event_database = {
    "ukraine russia war": [
        {"date": "2022-02-24", "title": "Russian invasion of Ukraine begins", "description": "Russia launches a full-scale military invasion of Ukraine", "source": "Wikipedia"},
        {"date": "2022-03-02", "title": "Kherson falls to Russian forces", "description": "The city of Kherson becomes the first major Ukrainian city to fall to Russian forces", "source": "Wikipedia"},
        {"date": "2022-04-03", "title": "Bucha massacre discovered", "description": "Evidence of war crimes discovered after Russian withdrawal from Bucha", "source": "Wikipedia"},
        {"date": "2022-05-17", "title": "Azovstal defenders surrender", "description": "Ukrainian defenders at Azovstal steel plant in Mariupol surrender after weeks-long siege", "source": "Wikipedia"},
        {"date": "2022-09-30", "title": "Russian annexation of occupied territories", "description": "Russia formally annexes four partially occupied Ukrainian regions", "source": "Wikipedia"},
        {"date": "2022-11-11", "title": "Ukrainian forces recapture Kherson", "description": "Ukrainian forces liberate the city of Kherson after Russian withdrawal", "source": "Wikipedia"},
    ],
    "covid vaccine": [
        {"date": "2020-12-11", "title": "FDA authorizes Pfizer vaccine", "description": "FDA issues first emergency use authorization for Pfizer-BioNTech COVID-19 vaccine", "source": "FDA"},
        {"date": "2020-12-18", "title": "FDA authorizes Moderna vaccine", "description": "FDA issues emergency use authorization for Moderna COVID-19 vaccine", "source": "FDA"},
        {"date": "2021-02-27", "title": "Johnson & Johnson vaccine authorized", "description": "FDA issues emergency use authorization for single-dose Johnson & Johnson COVID-19 vaccine", "source": "FDA"},
        {"date": "2021-05-10", "title": "Pfizer approved for adolescents", "description": "FDA expands Pfizer vaccine authorization to include adolescents 12-15 years old", "source": "FDA"},
        {"date": "2021-08-23", "title": "Pfizer receives full FDA approval", "description": "Pfizer-BioNTech COVID-19 vaccine receives full FDA approval for individuals 16 years and older", "source": "FDA"},
        {"date": "2021-10-29", "title": "Pfizer approved for children 5-11", "description": "FDA authorizes Pfizer vaccine for emergency use in children 5-11 years old", "source": "FDA"},
    ],
    "climate change": [
        {"date": "2021-02-19", "title": "US rejoins Paris Climate Agreement", "description": "The United States officially rejoins the Paris Climate Agreement", "source": "UN"},
        {"date": "2021-08-09", "title": "IPCC Sixth Assessment Report", "description": "IPCC releases landmark report showing 'code red for humanity' on climate change", "source": "IPCC"},
        {"date": "2021-11-13", "title": "COP26 concludes with Glasgow Climate Pact", "description": "UN Climate Change Conference concludes with new global agreement", "source": "UNFCCC"},
        {"date": "2022-03-21", "title": "Record Antarctic heat wave", "description": "Antarctica experiences unprecedented heat wave with temperatures 40°C above normal", "source": "NOAA"},
        {"date": "2022-08-16", "title": "US Inflation Reduction Act signed", "description": "US passes major climate legislation with $369 billion for climate action", "source": "US Government"},
    ],
    "cryptocurrency": [
        {"date": "2021-02-08", "title": "Tesla invests $1.5B in Bitcoin", "description": "Tesla announces $1.5 billion investment in Bitcoin and plans to accept it as payment", "source": "SEC filings"},
        {"date": "2021-04-14", "title": "Coinbase goes public", "description": "Cryptocurrency exchange Coinbase begins trading on Nasdaq", "source": "Nasdaq"},
        {"date": "2021-05-12", "title": "Tesla suspends Bitcoin payments", "description": "Tesla suspends vehicle purchases using Bitcoin, citing environmental concerns", "source": "Twitter"},
        {"date": "2021-09-07", "title": "El Salvador adopts Bitcoin", "description": "El Salvador becomes first country to adopt Bitcoin as legal tender", "source": "Government of El Salvador"},
        {"date": "2022-05-12", "title": "Terra Luna collapse", "description": "Cryptocurrency Terra Luna collapses, wiping out $40 billion in market value", "source": "CoinMarketCap"},
        {"date": "2022-11-11", "title": "FTX files for bankruptcy", "description": "Major cryptocurrency exchange FTX files for bankruptcy after liquidity crisis", "source": "FTX"},
    ],
    "artificial intelligence": [
        {"date": "2022-01-27", "title": "Google's AI system for chip design", "description": "Google announces using AI system for chip floorplanning design that outperforms humans", "source": "Google Research"},
        {"date": "2022-04-13", "title": "EU AI Act proposed", "description": "European Union proposes comprehensive AI regulations", "source": "European Commission"},
        {"date": "2022-11-30", "title": "ChatGPT released", "description": "OpenAI releases ChatGPT, triggering massive public interest in AI", "source": "OpenAI"},
        {"date": "2023-03-14", "title": "GPT-4 announced", "description": "OpenAI announces GPT-4, a multimodal large language model", "source": "OpenAI"},
        {"date": "2023-05-22", "title": "AI Safety Summit announced", "description": "UK announces plans to host first global AI Safety Summit", "source": "UK Government"},
    ]
}

@app.route('/api/events', methods=['GET'])
def get_historical_events():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        # Find events related to the query
        # First, look for exact matches in our event database
        matching_events = []
        
        # Try to find the most relevant event category based on the query
        best_match = None
        highest_match_score = 0
        
        for event_category in event_database.keys():
            # Calculate simple term overlap for matching
            category_terms = set(event_category.lower().split())
            query_terms = set(query.lower().split())
            common_terms = category_terms.intersection(query_terms)
            
            # Score based on proportion of matching terms
            if len(category_terms) > 0:
                match_score = len(common_terms) / len(category_terms)
                
                # Check if this is the best match so far
                if match_score > highest_match_score:
                    highest_match_score = match_score
                    best_match = event_category
        
        # Use best matching category if score is above threshold
        if highest_match_score >= 0.3 and best_match:
            matching_events = event_database[best_match]
        
        # If no matches, try to look for partial matches or use Groq to generate events
        if not matching_events and has_groq and GROQ_API_KEY:
            # Use Groq API to generate potential events
            groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Construct a prompt to generate historical events
            prompt = f"""
            I need a list of 3-5 major historical events related to "{query}".
            
            For each event, provide:
            1. The date (in YYYY-MM-DD format)
            2. A short title (5-7 words)
            3. A brief description (15-20 words)
            4. A reliable source
            
            Format the response as a JSON list of objects with fields: date, title, description, source.
            Do not include explanations or any text outside the JSON structure.
            """
            
            payload = {
                "model": "qwen/qwen3-32b",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate historical information."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 500
            }
            
            try:
                response = requests.post(groq_api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    events_text = result["choices"][0]["message"]["content"].strip()
                    
                    # Extract JSON from response (it might be wrapped in markdown code blocks)
                    import re
                    json_match = re.search(r'```json(.*?)```', events_text, re.DOTALL)
                    if json_match:
                        events_text = json_match.group(1).strip()
                    
                    # Clean up any remaining markdown or text
                    events_text = re.sub(r'```.*?```', '', events_text, flags=re.DOTALL)
                    events_text = events_text.strip()
                    
                    # Try to parse the JSON
                    try:
                        import json
                        generated_events = json.loads(events_text)
                        matching_events = generated_events
                    except json.JSONDecodeError:
                        print(f"Error parsing generated events: {events_text}")
            except Exception as e:
                print(f"Error generating events with Groq: {e}")
        
        # If still no events, return an informative message
        if not matching_events:
            return jsonify({
                'events': [],
                'message': f"No historical events found for '{query}'. Try a more specific query related to major news events."
            })
        
        # Filter events by date range if provided
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            matching_events = [
                event for event in matching_events 
                if start <= pd.to_datetime(event['date']) <= end
            ]
        
        # If there are no events in the specified range, return appropriate message
        if not matching_events:
            return jsonify({
                'events': [],
                'message': f"No events found for '{query}' in the specified date range. Try expanding the date range or using a different query."
            })
        
        # Correlate events with social media activity
        # Filter data based on query for time series
        filtered_data = data[
            data['selftext'].str.contains(query, case=False, na=False) |
            data['title'].str.contains(query, case=False, na=False)
        ]
        
        # Group by date to get post counts
        filtered_data['date'] = filtered_data['created_utc'].dt.date
        post_counts = filtered_data.groupby('date').size()
        
        # Calculate rolling average for smoothing (7-day window)
        rolling_avg = post_counts.rolling(window=7, min_periods=1).mean()
        
        # Calculate standard deviation for detecting peaks
        std_dev = post_counts.std()
        mean_posts = post_counts.mean()
        
        # Find peaks (days with significantly higher activity)
        peak_dates = post_counts[post_counts > (mean_posts + 1.5 * std_dev)].index
        
        # Map events to their nearby activity and identify correlations
        correlated_events = []
        for event in matching_events:
            event_date = pd.to_datetime(event['date']).date()
            
            # Check if event data is in the post counts index
            if event_date in post_counts.index:
                posts_on_day = int(post_counts.loc[event_date])
                rolling_avg_on_day = float(rolling_avg.loc[event_date])
            else:
                # Find the closest date in the data
                closest_dates = post_counts.index.astype('datetime64[ns]').astype(object)
                closest_dates = [date for date in closest_dates]
                
                if not closest_dates:
                    # No post data available for comparison
                    posts_on_day = 0
                    rolling_avg_on_day = 0
                else:
                    # Find the date closest to the event date
                    closest_date = min(closest_dates, key=lambda x: abs(x - event_date))
                    posts_on_day = int(post_counts.loc[closest_date])
                    rolling_avg_on_day = float(rolling_avg.loc[closest_date])
            
            # Calculate days to nearest peak
            if len(peak_dates) > 0:
                days_to_nearest_peak = min([abs((event_date - peak_date).days) for peak_date in peak_dates])
            else:
                days_to_nearest_peak = None
            
            # Determine correlation type
            if days_to_nearest_peak is not None and days_to_nearest_peak <= 2:
                correlation = "strong"  # Event coincides with peak
            elif days_to_nearest_peak is not None and days_to_nearest_peak <= 7:
                correlation = "moderate"  # Event is close to peak
            else:
                correlation = "weak"  # No clear correlation
            
            # Calculate relative activity compared to average (how many times above average)
            if rolling_avg_on_day > 0:
                activity_ratio = posts_on_day / rolling_avg_on_day
            else:
                activity_ratio = 0
            
            # Add correlation data to the event
            correlated_event = {
                **event,  # Keep original event data
                'posts_on_day': posts_on_day,
                'average_posts': round(rolling_avg_on_day, 2),
                'activity_ratio': round(activity_ratio, 2),
                'correlation': correlation,
                'days_to_nearest_peak': days_to_nearest_peak
            }
            
            # Add user-friendly insights about correlation
            if correlation == "strong":
                correlated_event['insight'] = f"This event coincides with a significant spike in online discussions, with {posts_on_day} posts (about {round(activity_ratio, 1)}x the average)."
            elif correlation == "moderate":
                correlated_event['insight'] = f"This event is temporally close to increased online activity, with {posts_on_day} posts around this time."
            else:
                correlated_event['insight'] = f"This event didn't correspond with unusual online activity, with {posts_on_day} posts on this day."
            
            correlated_events.append(correlated_event)
        
        # Create a time series summary for context
        time_series_data = []
        for date, count in post_counts.items():
            time_series_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'count': int(count),
                'is_peak': date in peak_dates
            })
        
        # Return the correlated events with context
        return jsonify({
            'events': correlated_events,
            'time_series': time_series_data,
            'query': query,
            'total_posts': len(filtered_data),
            'peak_dates': [date.strftime('%Y-%m-%d') for date in peak_dates],
            'event_category': best_match if highest_match_score >= 0.3 else query
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error retrieving events: {str(e)}'}), 500

@app.route('/api/semantic_search', methods=['GET'])
def semantic_search():

    global platform_manager, data
    
    # Check if data is available
    if (platform_manager.integrated_data is None) and (data is None):
        return jsonify({'error': 'No data loaded'}), 400
    
    url = request.args.get('url', '')
    query = request.args.get('query', '')
    platform = request.args.get('platform', None)
    max_results = min(int(request.args.get('max_results', 20)), 100)  # Limit to max 100 for performance
    
    if not url:
        return jsonify({'error': 'URL parameter is required'}), 400
    
    try:
        # Step 1: Find posts containing the URL
        # Use platform manager if available
        if platform_manager.integrated_data is not None:
            # Find posts containing the URL across platforms or in a specific platform
            all_data = platform_manager.get_platform_data(platform)
            
            # Use the normalized column names
            content_column = 'content_text'
            title_column = 'title'
        else:
            # Fallback to global data variable (backward compatibility)
            all_data = data
            content_column = 'selftext'
            title_column = 'title'
        
        # Filter posts containing the URL
        url_pattern = re.escape(url)
        url_matches = all_data[
            all_data[content_column].str.contains(url_pattern, case=False, na=False, regex=True) |
            all_data[title_column].str.contains(url_pattern, case=False, na=False, regex=True)
        ]
        
        if len(url_matches) == 0:
            return jsonify({
                'results': [],
                'message': f"No posts found containing URL: {url}"
            })
        
        # Step 2: If no query provided, return the URL matches directly
        if not query:
            # Sort by date (most recent first)
            url_matches = url_matches.sort_values(by='created_at' if 'created_at' in url_matches.columns else 'created_utc', ascending=False)
            
            # Format the results
            results = []
            for _, post in url_matches.head(max_results).iterrows():
                result = {
                    'title': post.get(title_column, ''),
                    'content': post.get(content_column, '')[:300] + '...' if len(post.get(content_column, '') or '') > 300 else post.get(content_column, ''),
                    'author': post.get('author', ''),
                    'created_at': post.get('created_at', post.get('created_utc', '')).isoformat(),
                    'platform': post.get('platform', 'reddit'),
                    'similarity': 1.0,  # Direct URL match
                    'url_match': True
                }
                if 'community' in post:
                    result['community'] = post['community']
                elif 'subreddit' in post:
                    result['community'] = post['subreddit']
                
                results.append(result)
            
            return jsonify({
                'results': results,
                'total_matches': len(url_matches),
                'message': f"Found {len(url_matches)} posts containing the URL"
            })
        
        # Step 3: Perform semantic search with the query
        # Load the sentence transformer model on demand
        global semantic_model
        if semantic_model is None:
            try:
                # Use a smaller, faster model for embeddings
                semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded semantic embedding model successfully")
            except Exception as e:
                print(f"Error loading semantic model: {e}")
                return jsonify({'error': 'Failed to load semantic model'}), 500
        
        # Step 4: Generate embeddings for the query and posts
        # Create the query embedding
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)
        
        # Prepare texts from matched posts for embedding
        post_texts = []
        for idx, post in url_matches.iterrows():
            # Combine title and content for better semantic matching
            title = post.get(title_column, '')
            content = post.get(content_column, '')
            combined = f"{title} {content[:500]}"  # Limit length for efficiency
            post_texts.append(combined)
        
        # Generate embeddings for all posts
        post_embeddings = semantic_model.encode(post_texts, convert_to_tensor=True)
        
        # Step 5: Compute semantic similarity
        # Calculate cosine similarity between query and each post
        from torch.nn import functional as F
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), post_embeddings).cpu().numpy()
        
        # Step 6: Rank the results by similarity
        # Create a list of (index, similarity) pairs
        ranked_indices = [(i, similarities[i]) for i in range(len(similarities))]
        # Sort by similarity (highest first)
        ranked_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Step 7: Format and return top results
        results = []
        for i, similarity in ranked_indices[:max_results]:
            post = url_matches.iloc[i]
            result = {
                'title': post.get(title_column, ''),
                'content': post.get(content_column, '')[:300] + '...' if len(post.get(content_column, '') or '') > 300 else post.get(content_column, ''),
                'author': post.get('author', ''),
                'created_at': post.get('created_at', post.get('created_utc', '')).isoformat(),
                'platform': post.get('platform', 'reddit'),
                'similarity': float(similarity),
                'url_match': True
            }
            if 'community' in post:
                result['community'] = post['community']
            elif 'subreddit' in post:
                result['community'] = post['subreddit']
            
            results.append(result)
        
        return jsonify({
            'results': results,
            'total_matches': len(url_matches),
            'query': query,
            'message': f"Found {len(results)} semantically relevant posts containing the URL"
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error during semantic search: {str(e)}'}), 500

@app.route('/api/semantic_query', methods=['GET'])
def semantic_query():

    global platform_manager, data, semantic_model
    
    # Check if data is available
    if (platform_manager.integrated_data is None) and (data is None):
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    platform = request.args.get('platform', None)
    max_results = min(int(request.args.get('max_results', 20)), 100)  # Limit to max 100 for performance
    min_similarity = float(request.args.get('min_similarity', 0.5))  # Minimum similarity threshold
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    try:
        # Use platform manager if available
        if platform_manager.integrated_data is not None:
            # Get data from specific platform or all platforms
            all_data = platform_manager.get_platform_data(platform)
            
            # Use the normalized column names
            content_column = 'content_text'
            title_column = 'title'
        else:
            # Fallback to global data variable (backward compatibility)
            all_data = data
            content_column = 'selftext'
            title_column = 'title'
        
        # To avoid processing too many posts, take a reasonable sample
        sample_size = 1000  # Sample size for semantic processing
        if len(all_data) > sample_size:
            # Try to get a balanced sample across platforms
            if 'platform' in all_data.columns and platform is None:
                # Try to get a balanced sample across platforms
                platforms = all_data['platform'].unique()
                per_platform = max(50, sample_size // len(platforms))
                sample_frames = []
                
                for p in platforms:
                    platform_data = all_data[all_data['platform'] == p]
                    if len(platform_data) > per_platform:
                        platform_sample = platform_data.sample(per_platform, random_state=42)
                        sample_frames.append(platform_sample)
                    else:
                        sample_frames.append(platform_data)
                
                sampled_data = pd.concat(sample_frames)
            else:
                # Simple random sample
                sampled_data = all_data.sample(sample_size, random_state=42)
        else:
            sampled_data = all_data
        
        # Load the sentence transformer model on demand
        if semantic_model is None:
            try:
                semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded semantic embedding model successfully")
            except Exception as e:
                print(f"Error loading semantic model: {e}")
                return jsonify({'error': 'Failed to load semantic model'}), 500
        
        # Generate embedding for the query
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)
        
        # Prepare texts from posts for embedding
        post_texts = []
        post_indices = []
        
        for idx, post in sampled_data.iterrows():
            # Combine title and content for better semantic matching
            title = post.get(title_column, '')
            content = post.get(content_column, '')
            combined = f"{title} {content[:500]}"  # Limit length for efficiency
            post_texts.append(combined)
            post_indices.append(idx)
        
        # Generate embeddings for all posts
        post_embeddings = semantic_model.encode(post_texts, convert_to_tensor=True)
        
        # Compute semantic similarity
        from torch.nn import functional as F
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), post_embeddings).cpu().numpy()
        
        # Create a list of (index, similarity) pairs
        ranked_indices = [(post_indices[i], similarities[i]) for i in range(len(similarities))]
        
        # Filter by minimum similarity threshold and sort
        ranked_indices = [pair for pair in ranked_indices if pair[1] >= min_similarity]
        ranked_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Format the top results
        results = []
        for idx, similarity in ranked_indices[:max_results]:
            post = all_data.loc[idx]
            result = {
                'title': post.get(title_column, ''),
                'content': post.get(content_column, '')[:300] + '...' if len(post.get(content_column, '') or '') > 300 else post.get(content_column, ''),
                'author': post.get('author', ''),
                'created_at': post.get('created_at', post.get('created_utc', '')).isoformat(),
                'platform': post.get('platform', 'reddit'),
                'similarity': float(similarity)
            }
            if 'community' in post:
                result['community'] = post['community']
            elif 'subreddit' in post:
                result['community'] = post['subreddit']
            
            results.append(result)
        
        # Extract key terms from the query for explanation
        keywords = []
        try:
            # Simple extraction of nouns and important words
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            stop_words = {'the', 'and', 'for', 'with', 'about', 'what', 'how', 'when', 'who', 'why', 'where', 'which'}
            keywords = [word for word in words if word not in stop_words][:5]  # Top 5 keywords
        except:
            pass
        
        return jsonify({
            'results': results,
            'total_results': len(results),
            'sample_size': len(sampled_data),
            'query': query,
            'key_terms': keywords,
            'message': f"Found {len(results)} semantically relevant posts matching your query"
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error during semantic query: {str(e)}'}), 500

if __name__ == '__main__':
    # Load dataset on startup
    load_success = load_dataset()
    if not load_success:
        print("WARNING: Failed to load dataset. Make sure data file exists at ./data/data.jsonl")
        # Create data directory if it doesn't exist
        os.makedirs("./data", exist_ok=True)
        print("Created data directory. Please place your data.jsonl file in the ./data folder.")
    
    app.run(debug=True, host='0.0.0.0', port=80) 