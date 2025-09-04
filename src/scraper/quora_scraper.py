import sys
import os
import time
import random
import re
import logging
import sqlite3
from typing import List, Dict, Optional
import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Add project root to path for database manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.database.sqlite_manager import SQLiteManager


class EnhancedQuoraScraper:
    """Enhanced Quora scraper with database integration and human-like browsing"""

    def __init__(self, db_path: str = "data/databases/quora_data.db"):
        self.db_manager = SQLiteManager(db_path)
        self.driver = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    # -------------------- DRIVER SETUP --------------------
    def setup_driver(self, headless: bool = False):
        """Setup Chrome driver with optimized options"""
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        if headless:
            options.add_argument("--headless")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

    # -------------------- HUMAN-LIKE METHODS --------------------
    def human_sleep(self, min_sec=5, max_sec=12):
        """Sleep for random duration to mimic human behavior"""
        time.sleep(random.uniform(min_sec, max_sec))

    def human_scroll(self, total_scrolls=5):
        """Scroll the page in random increments"""
        for _ in range(total_scrolls):
            self.driver.execute_script(f"window.scrollBy(0, {random.randint(200, 800)})")
            self.human_sleep(1, 3)

    def check_for_captcha(self):
        """Detect if a CAPTCHA is present"""
        if "captcha" in self.driver.current_url.lower() or "sorry/index" in self.driver.current_url.lower():
            self.logger.warning("CAPTCHA detected! Please solve it manually to continue...")
            input("After solving CAPTCHA, press Enter to continue...")

    # -------------------- CLEANING & QUALITY --------------------
    def clean_text(self, text: str) -> str:
        """Clean and normalize scraped text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'Continue Reading.*$', '', text)
        text = re.sub(r'\(more\).*$', '', text)
        text = re.sub(r'See more.*$', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        return text.strip()

    def extract_answer_quality_indicators(self, answer_element) -> Dict:
        """Extract quality indicators from answer element"""
        quality_info = {'upvotes': 0, 'shares': 0, 'comments': 0, 'has_credentials': False, 'answer_length': 0}
        try:
            upvote_elements = answer_element.find_all(text=re.compile(r'\d+\s*(upvote|like)', re.I))
            if upvote_elements:
                numbers = re.findall(r'\d+', upvote_elements[0])
                if numbers:
                    quality_info['upvotes'] = int(numbers[0])
            credential_indicators = ['PhD', 'Dr.', 'Professor', 'Engineer', 'Expert', 'CEO']
            answer_text = answer_element.get_text()
            quality_info['has_credentials'] = any(cred in answer_text for cred in credential_indicators)
        except Exception as e:
            self.logger.warning(f"Error extracting quality indicators: {e}")
        return quality_info

    # -------------------- SCRAPING --------------------
    def _is_navigation_text(self, text: str) -> bool:
        nav_indicators = [
            "Sign up", "Log in", "Follow", "Upvote", "Share", "Comment",
            "More answers", "Related questions", "Spaces", "Profile",
            "Add answer", "Write answer", "Continue reading"
        ]
        text_lower = text.lower()
        return any(ind.lower() in text_lower for ind in nav_indicators)

    def extract_question_and_answers(self) -> Optional[Dict]:
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        # Extract question title
        question_text = ""
        if soup.title:
            question_text = soup.title.get_text().replace(" - Quora", "").strip()
        if not question_text or len(question_text) < 10:
            question_elements = soup.find_all(["h1", "h2"])
            for elem in question_elements:
                text = elem.get_text().strip()
                if len(text) > 10 and "?" in text:
                    question_text = text
                    break
        if not question_text:
            self.logger.warning("Could not extract question text")
            return None
        question_text = self.clean_text(question_text)

        # Extract answers
        answer_selectors = ["div[class*='answer']", "div[class*='Answer']", "div[class*='content']", ".q-box", "[data-testid*='answer']"]
        all_answers = []
        answers_found = False

        for selector in answer_selectors:
            if answers_found:
                break
            answer_blocks = soup.select(selector)
            self.logger.info(f"Trying selector '{selector}': found {len(answer_blocks)} elements")
            for block in answer_blocks:
                try:
                    answer_text = block.get_text(strip=True)
                    if len(answer_text) > 100 and not self._is_navigation_text(answer_text) and answer_text not in [ans['text'] for ans in all_answers]:
                        quality_info = self.extract_answer_quality_indicators(block)
                        cleaned_answer = self.clean_text(answer_text)
                        if cleaned_answer and len(cleaned_answer) > 50:
                            all_answers.append({'text': cleaned_answer, 'quality_indicators': quality_info})
                            answers_found = True
                except:
                    continue
        if not all_answers:
            self.logger.warning(f"No answers found for question: {question_text}")
            return None

        return {"question": question_text, "answers": all_answers[:10]}

    def search_and_scrape_question(self, query: str) -> Optional[Dict]:
        try:
            self.logger.info(f"Searching: {query}")
            self.driver.get("https://www.google.com")
            self.human_sleep(2, 4)
            self.check_for_captcha()

            # Cookie handling
            try:
                cookie_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, '//button[contains(text(),"Accept") or contains(text(),"I agree")]'))
                )
                cookie_button.click()
                self.human_sleep(1, 2)
            except:
                pass

            search_box = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.NAME, "q")))
            search_box.clear()
            search_box.send_keys(f"site:quora.com {query}")
            self.human_sleep(1, 2)
            search_box.send_keys(Keys.RETURN)
            self.human_sleep(3, 5)
            self.check_for_captcha()

            # Click first result
            try:
                first_result = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "h3")))
                first_result.click()
                self.human_sleep(5, 8)
            except Exception as e:
                self.logger.error(f"Failed to find search result for '{query}': {e}")
                return None

            self.logger.info("Loading more answers...")
            self.human_scroll(total_scrolls=4)

            # Click "Load more"
            try:
                load_more_buttons = self.driver.find_elements(By.XPATH, '//button[contains(text(),"more") or contains(text(),"More")]')
                for button in load_more_buttons:
                    if button.is_displayed():
                        button.click()
                        self.human_sleep(2, 4)
                        break
            except:
                pass

            self.check_for_captcha()
            return self.extract_question_and_answers()

        except Exception as e:
            self.logger.error(f"Error scraping query '{query}': {e}")
            return None

    # -------------------- QUALITY & DATABASE --------------------
    def estimate_answer_quality(self, answer_data: Dict) -> float:
        quality_score = 0.0
        length = len(answer_data['text'])
        if length > 200:
            quality_score += 0.3
        if length > 500:
            quality_score += 0.2
        indicators = answer_data.get('quality_indicators', {})
        upvotes = indicators.get('upvotes', 0)
        if upvotes > 0:
            quality_score += min(0.4, upvotes * 0.05)
        if indicators.get('has_credentials', False):
            quality_score += 0.1
        return min(1.0, quality_score)

    def save_scraped_data(self, scraped_data: Dict, topic_category: str = None) -> bool:
        try:
            question_id = self.db_manager.insert_question(scraped_data['question'], topic_category)
            for answer_data in scraped_data['answers']:
                answer_id = self.db_manager.insert_answer(question_id, answer_data['text'])
                if answer_id:
                    quality_score = self.estimate_answer_quality(answer_data)
                    with sqlite3.connect(self.db_manager.db_path) as conn:
                        conn.execute("UPDATE answers SET estimated_quality_score = ? WHERE id = ?", (quality_score, answer_id))
            self.logger.info(f"Saved question_id {question_id} with {len(scraped_data['answers'])} answers")
            return True
        except Exception as e:
            self.logger.error(f"Error saving scraped data: {e}")
            return False

    def scrape_multiple_queries(self, queries: List[str], topic_category: str = None, min_delay=5, max_delay=12, delay_between_queries: int = 5) -> Dict:
        if not self.driver:
            self.setup_driver(headless=False)

        results = {'successful_scrapes': 0, 'failed_scrapes': 0, 'total_questions': 0, 'total_answers': 0}

        for i, query in enumerate(queries):
            self.logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
            try:
                scraped_data = self.search_and_scrape_question(query)
                if scraped_data and scraped_data.get('answers'):
                    if self.save_scraped_data(scraped_data, topic_category):
                        results['successful_scrapes'] += 1
                        results['total_questions'] += 1
                        results['total_answers'] += len(scraped_data['answers'])
                    else:
                        results['failed_scrapes'] += 1
                else:
                    results['failed_scrapes'] += 1
                    self.logger.warning(f"No data scraped for query: {query}")
            except Exception as e:
                self.logger.error(f"Error processing query '{query}': {e}")
                results['failed_scrapes'] += 1

            if i < len(queries) - 1:
                self.human_sleep(min_delay, max_delay)

        return results

    def close(self):
        if self.driver:
            self.driver.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -------------------- CSV LOADING & SCRIPT RUN --------------------
def load_queries_from_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if "question" in df.columns:
        return df["question"].dropna().tolist()
    return df.iloc[:, 0].dropna().tolist()


if __name__ == "__main__":
    CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quora_faq_questions.csv")
    queries = load_queries_from_csv(CSV_PATH)

    with EnhancedQuoraScraper() as scraper:
        results = scraper.scrape_multiple_queries(queries, topic_category="General", min_delay=5, max_delay=12)
        print("\nScraping Summary:", results)
