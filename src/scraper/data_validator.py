class DataValidator:
    """Validates scraped data quality"""
    
    @staticmethod
    def validate_question(question_text: str) -> bool:
        """Validate question quality"""
        if not question_text or len(question_text) < 10:
            return False
        
        # Should end with question mark or be interrogative
        interrogative_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'do', 'does', 'is', 'are']
        
        return (question_text.endswith('?') or 
                any(word in question_text.lower().split()[:3] for word in interrogative_words))
    
    @staticmethod
    def validate_answer(answer_text: str) -> bool:
        """Validate answer quality"""
        if not answer_text or len(answer_text) < 50:
            return False
        
        # Check for meaningful content (not just navigation text)
        meaningful_indicators = ['.', ',', '!', '?']
        return any(indicator in answer_text for indicator in meaningful_indicators)
    
    @staticmethod
    def validate_scraped_data(data: Dict) -> Tuple[bool, List[str]]:
        """Validate complete scraped data"""
        errors = []
        
        if not DataValidator.validate_question(data.get('question', '')):
            errors.append("Invalid or low-quality question")
        
        valid_answers = [ans for ans in data.get('answers', []) 
                        if DataValidator.validate_answer(ans.get('text', ''))]
        
        if len(valid_answers) == 0:
            errors.append("No valid answers found")
        
        return len(errors) == 0, errors