import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.database.sqlite_manager import SQLiteManager

def main():
    print("Setting up database...")
    
    # Create data directories
    os.makedirs("data/databases", exist_ok=True)
    
    # Initialize database
    db_manager = SQLiteManager()
    
    # Print initial stats
    stats = db_manager.get_database_stats()
    print("Database initialized successfully!")
    print("Initial stats:", stats)

if __name__ == "__main__":
    main()