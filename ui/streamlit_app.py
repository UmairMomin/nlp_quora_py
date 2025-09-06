import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.embeddings.similarity_search import SemanticSearchEngine
from src.database.sqlite_manager import SQLiteManager
from scripts.data_pipeline import DataPipeline

st.set_page_config(
    page_title="Quora NLP Analyzer",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_search_engine():
    """Load search engine (cached)"""
    return SemanticSearchEngine()

@st.cache_resource  
def load_db_manager():
    """Load database manager (cached)"""
    return SQLiteManager()

def main():
    st.title("🔍 Quora NLP Analyzer")
    st.markdown("Semantic Analysis and Best Answer Retrieval System")
    
    # Add custom CSS for better styling (works with both light and dark themes)
    st.markdown("""
    <style>
    /* Main best answer container */
    .best-answer {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #1f77b4;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        .best-answer {
            background-color: rgba(30, 30, 30, 0.9);
            border: 1px solid rgba(255,255,255,0.1);
        }
    }
    
    /* Answer metrics styling */
    .answer-metrics {
        background-color: rgba(31, 119, 180, 0.15);
        padding: 12px 15px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid rgba(31, 119, 180, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        margin-top: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: rgba(240, 242, 246, 0.8);
        border-radius: 6px 6px 0px 0px;
        padding: 10px 15px;
        font-weight: 500;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
        border-color: #1f77b4 !important;
    }
    
    /* Text formatting */
    .answer-text {
        font-size: 16px;
        line-height: 1.7;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin-top: 15px;
        color: inherit;
    }
    
    /* Question text styling */
    .original-question {
        margin-bottom: 15px;
        font-style: italic;
        color: #666;
        font-size: 15px;
        padding: 8px 12px;
        background-color: rgba(0,0,0,0.05);
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Search", "Database Stats", "Data Pipeline", "About"
    ])
    
    if page == "Search":
        search_page()
    elif page == "Database Stats":
        stats_page()
    elif page == "Data Pipeline":
        pipeline_page()
    elif page == "About":
        about_page()

def search_page():
    """Main search interface"""
    st.header("🔍 Semantic Search")
    st.markdown("Ask any question and get the best answers from our Quora database!")
    
    # Load search engine
    try:
        search_engine = load_search_engine()
        st.success("✅ Search engine loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading search engine: {e}")
        st.info("Please run the data pipeline first to generate embeddings and indexes.")
        return
    
    # Search input with better styling
    st.markdown("### Ask Your Question")
    query = st.text_input(
        "Enter your question:", 
        placeholder="e.g., How to learn machine learning?",
        help="Type your question in natural language. The system will find the most relevant answers."
    )
    
    # Search parameters
    with st.expander("⚙️ Search Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            top_k_questions = st.slider("Number of similar questions", 1, 10, 5, help="How many similar questions to show")
        with col2:
            top_k_answers = st.slider("Number of best answers", 1, 10, 3, help="How many best answers to retrieve")
    
    # Search button
    if query and st.button("🔍 Search for Answers", type="primary", use_container_width=True):
        with st.spinner("Searching..."):
            # Find similar questions
            similar_questions = search_engine.find_similar_questions(query, top_k_questions)
            
            # Find best answers
            best_answers = search_engine.find_best_answers_for_question(query, top_k_answers)
        
        # Display results
        if similar_questions or best_answers:
            # Show best answer prominently at the top
            if best_answers:
                st.subheader("🎯 Best Answer")
                best_answer = best_answers[0]  # Get the top answer
                
                # Get the original question for this answer
                original_question = "Unknown question"
                if best_answer.get('question_id'):
                    try:
                        import sqlite3
                        with sqlite3.connect('data/databases/quora_data.db') as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT question_text FROM questions WHERE id = ?", (best_answer['question_id'],))
                            result = cursor.fetchone()
                            if result:
                                original_question = result[0]
                    except:
                        pass
                
                # Create a highlighted box for the best answer using custom CSS
                # Clean up the answer text for better display
                answer_text = best_answer['answer_text'].replace('\n', '<br>').replace('  ', ' ')
                
                st.markdown(f"""
                <div class="best-answer">
                    <div class="original-question">
                        <strong>Original Question:</strong> {original_question}
                    </div>
                    <div class="answer-metrics">
                        <strong>Quality Score:</strong> {best_answer['quality_score']:.3f} | 
                        <strong>Length:</strong> {best_answer['answer_length']} words
                        {f' | <strong>Question Match:</strong> {best_answer["question_match_score"]:.3f}' if best_answer.get('question_match_score') else ''}
                    </div>
                    <div class="answer-text">
                        {answer_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show other similar answers and questions below
            if len(best_answers) > 1 or similar_questions:
                st.subheader("📚 Additional Results")
                
                # Create tabs for similar answers and questions
                tab1, tab2 = st.tabs(["Similar Answers", "Similar Questions"])
                
                with tab1:
                    if len(best_answers) > 1:
                        st.write("**Other High-Quality Answers:**")
                        for i, ans in enumerate(best_answers[1:], 2):  # Skip the first (best) answer
                            with st.expander(f"Answer {i}: Quality Score {ans['quality_score']:.3f}"):
                                st.write(f"**Answer:** {ans['answer_text'][:500]}...")
                                st.write(f"**Quality Score:** {ans['quality_score']:.3f}")
                                st.write(f"**Answer Length:** {ans['answer_length']} words")
                                if ans.get('sentiment_score'):
                                    st.write(f"**Sentiment Score:** {ans['sentiment_score']:.3f}")
                                st.write(f"**Question Match Score:** {ans['question_match_score']:.3f}")
                    else:
                        st.info("No additional answers found")
                
                with tab2:
                    if similar_questions:
                        st.write("**Similar Questions:**")
                        for i, q in enumerate(similar_questions, 1):
                            with st.expander(f"Q{i}: {q['question_text'][:60]}..."):
                                st.write(f"**Full Question:** {q['question_text']}")
                                st.write(f"**Similarity Score:** {q['similarity_score']:.3f}")
                                st.write(f"**Total Answers:** {q['total_answers']}")
                                st.write(f"**Topic:** {q['topic_category'] or 'Not specified'}")
                    else:
                        st.info("No similar questions found")
        else:
            st.warning("No results found for your query")

def stats_page():
    """Database statistics page"""
    st.header("📊 Database Statistics")
    
    try:
        db_manager = load_db_manager()
        stats = db_manager.get_database_stats()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", stats['total_questions'])
        with col2:
            st.metric("Total Answers", stats['total_answers'])
        with col3:
            st.metric("Question Embeddings", stats['question_embeddings'])
        with col4:
            st.metric("Answer Embeddings", stats['answer_embeddings'])
        
        # Additional stats
        st.subheader("Detailed Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Average Answers per Question:** {stats['avg_answers_per_question']}")
            
            # Calculate embedding coverage
            if stats['total_questions'] > 0:
                q_embedding_coverage = (stats['question_embeddings'] / stats['total_questions']) * 100
                st.write(f"**Question Embedding Coverage:** {q_embedding_coverage:.1f}%")
            
            if stats['total_answers'] > 0:
                a_embedding_coverage = (stats['answer_embeddings'] / stats['total_answers']) * 100
                st.write(f"**Answer Embedding Coverage:** {a_embedding_coverage:.1f}%")
        
        with col2:
            # Create a simple pie chart for data distribution
            fig = px.pie(
                values=[stats['total_questions'], stats['total_answers']], 
                names=['Questions', 'Answers'],
                title="Data Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.subheader("Recent Questions")
        recent_questions = db_manager.search_questions_by_text("", limit=10)
        
        if recent_questions:
            df = pd.DataFrame(recent_questions)
            df.columns = ['ID', 'Question', 'Total Answers', 'Scraped At']
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No questions in database yet")
            
    except Exception as e:
        st.error(f"Error loading database stats: {e}")

def pipeline_page():
    """Data pipeline management page"""
    st.header("⚙️ Data Pipeline")
    
    # Pipeline status
    try:
        pipeline = DataPipeline()
        status = pipeline.get_pipeline_status()
        
        st.subheader("Pipeline Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if status['embeddings_ready']:
                st.success("✅ Embeddings Ready")
            else:
                st.warning("⚠️ Embeddings Not Generated")
        
        with col2:
            if status['indexes_ready']:
                st.success("✅ FAISS Indexes Ready")
            else:
                st.warning("⚠️ FAISS Indexes Not Built")
        
    except Exception as e:
        st.error(f"Error checking pipeline status: {e}")
    
    st.subheader("Run Pipeline Operations")
    
    # Scraping section
    with st.expander("🕷️ Data Scraping"):
        st.write("Add new questions and answers to the database")
        
        # Input for queries
        queries_input = st.text_area(
            "Enter queries (one per line):",
            placeholder="What is machine learning?\nHow to learn Python?\nBest programming practices"
        )
        
        topic_category = st.text_input("Topic Category (optional):", placeholder="Technology")
        
        if st.button("Start Scraping"):
            if queries_input.strip():
                queries = [q.strip() for q in queries_input.split('\n') if q.strip()]
                
                with st.spinner(f"Scraping {len(queries)} queries..."):
                    try:
                        pipeline = DataPipeline()
                        results = pipeline.run_scraping_phase(queries, topic_category)
                        
                        st.success("Scraping completed!")
                        st.json(results)
                        
                    except Exception as e:
                        st.error(f"Scraping failed: {e}")
            else:
                st.warning("Please enter at least one query")
    
    # Embedding generation section
    with st.expander("🧠 Generate Embeddings"):
        st.write("Generate embeddings for existing questions and answers")
        
        if st.button("Generate Embeddings"):
            with st.spinner("Generating embeddings..."):
                try:
                    pipeline = DataPipeline()
                    results = pipeline.run_embedding_phase()
                    
                    st.success("Embedding generation completed!")
                    st.json(results)
                    
                except Exception as e:
                    st.error(f"Embedding generation failed: {e}")
    
    # Index building section
    with st.expander("🏗️ Build FAISS Indexes"):
        st.write("Build or rebuild FAISS indexes for semantic search")
        
        if st.button("Build Indexes"):
            with st.spinner("Building FAISS indexes..."):
                try:
                    pipeline = DataPipeline()
                    results = pipeline.run_index_building_phase()
                    
                    st.success("Index building completed!")
                    st.json(results)
                    
                except Exception as e:
                    st.error(f"Index building failed: {e}")

def about_page():
    """About page with project information"""
    st.header("About This Project")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    **Semantic Analysis and Best Answer Retrieval for Quora Questions Using NLP**
    
    This project implements an intelligent system for analyzing Quora questions and answers using Natural Language Processing techniques.
    
    ### 🔧 Key Features:
    - **Semantic Question Matching**: Find questions with similar meaning using sentence embeddings
    - **Answer Quality Ranking**: Rank answers based on multiple quality indicators
    - **Efficient Search**: Fast semantic search using FAISS vector indexes
    - **Comprehensive NLP Pipeline**: End-to-end processing from scraping to search
    
    ### 🏗️ Architecture:
    1. **Data Scraping**: Automated collection from Quora using Selenium
    2. **Text Preprocessing**: Cleaning, tokenization, and normalization
    3. **Embedding Generation**: Sentence-BERT embeddings for semantic understanding
    4. **Vector Storage**: FAISS indexes for efficient similarity search
    5. **Database Storage**: SQLite for structured data and metadata
    
    ### 🚀 Technologies Used:
    - **NLP**: NLTK, spaCy, Sentence-Transformers
    - **ML**: scikit-learn, NumPy, Pandas
    - **Vector Search**: FAISS
    - **Database**: SQLite
    - **Web Scraping**: Selenium, BeautifulSoup
    - **UI**: Streamlit
    
    ### 👥 Team:
    - **Umair Momin** (223A1131)
    - **Fatima Mulla** (223A1132)  
    - **Mithrajeeth Yadavar** (223A1138)
    
    **Class:** BE CE-D
    """)
    
    # Show current system status
    st.subheader("🔧 System Status")
    
    try:
        pipeline = DataPipeline()
        status = pipeline.get_pipeline_status()
        
        status_df = pd.DataFrame([
            {"Component": "Database", "Status": "✅ Ready" if status['database_stats']['total_questions'] > 0 else "⚠️ Empty"},
            {"Component": "Embeddings", "Status": "✅ Ready" if status['embeddings_ready'] else "❌ Not Generated"},
            {"Component": "FAISS Indexes", "Status": "✅ Ready" if status['indexes_ready'] else "❌ Not Built"},
        ])
        
        st.dataframe(status_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error checking system status: {e}")

if __name__ == "__main__":
    main()