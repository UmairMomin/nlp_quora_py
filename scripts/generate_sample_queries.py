def generate_sample_queries():
    """Generate categorized sample queries for testing"""
    
    queries_by_category = {
        "Technology": [
            "What is machine learning and how does it work",
            "Best programming languages for beginners 2024",
            "How does artificial intelligence impact jobs",
            "What is the difference between Python and Java",
            "How to become a software engineer",
            "What is cloud computing and its benefits",
            "How does blockchain technology work",
            "Best practices for cybersecurity",
            "What is data science and analytics",
            "How to learn web development"
        ],
        
        "Science": [
            "How does quantum computing work",
            "What causes climate change and global warming",
            "How do vaccines work in the human body",
            "What is the theory of relativity explained simply",
            "How does DNA replication work",
            "What are the effects of plastic pollution",
            "How do black holes form in space",
            "What is genetic engineering and CRISPR",
            "How does the human brain process information",
            "What causes earthquakes and tsunamis"
        ],
        
        "Career_Education": [
            "How to prepare for software engineering interviews",
            "Best universities for computer science",
            "How to transition into data science career",
            "What skills are needed for digital marketing",
            "How to write a compelling resume",
            "Best strategies for online learning",
            "How to choose the right career path",
            "What is MBA and is it worth it",
            "How to develop leadership skills",
            "Best internship opportunities for students"
        ],
        
        "Health_Lifestyle": [
            "What are the benefits of meditation",
            "How to maintain a healthy diet",
            "Best exercises for weight loss",
            "How to improve sleep quality",
            "What causes stress and how to manage it",
            "Benefits of yoga for mental health",
            "How to build muscle effectively",
            "What is intermittent fasting",
            "How to improve memory and concentration",
            "Best ways to stay motivated"
        ],
        
        "Finance_Business": [
            "How to start investing in stock market",
            "What is cryptocurrency and Bitcoin",
            "How to start a successful business",
            "Best ways to save money effectively",
            "What is compound interest explained",
            "How to improve credit score",
            "What are mutual funds and ETFs",
            "How to create a budget plan",
            "What is entrepreneurship and startups",
            "How to negotiate salary effectively"
        ]
    }
    
    return queries_by_category

def save_queries_to_files():
    """Save queries to separate files by category"""
    queries = generate_sample_queries()
    
    os.makedirs("data/sample_queries", exist_ok=True)
    
    # Save all queries in one file
    all_queries = []
    for category, category_queries in queries.items():
        all_queries.extend(category_queries)
    
    with open("data/sample_queries/all_queries.txt", "w") as f:
        for query in all_queries:
            f.write(query + "\n")
    
    # Save by category
    for category, category_queries in queries.items():
        filename = f"data/sample_queries/{category.lower()}_queries.txt"
        with open(filename, "w") as f:
            for query in category_queries:
                f.write(query + "\n")
    
    print("Sample queries saved to data/sample_queries/")
    print(f"Total queries: {len(all_queries)}")
    
    return all_queries

if __name__ == "__main__":
    save_queries_to_files()