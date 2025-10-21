from fix_improved_rag import FixedImprovedJobScamRAG
import re
import time  # ADD THIS IMPORT

class AccurateJobScamRAG(FixedImprovedJobScamRAG):
    def extract_detailed_red_flags_from_text(self, text):
        """More accurate red flag detection - fixed version"""
        text_lower = text.lower()
        red_flags = []
        
        # More precise patterns with context
        red_flag_patterns = {
            'sensitive_info_request': [
                r'ssn\b', r'social security\b', r'bank details\b', 
                r'bank account\b', r'routing number\b', r'credit card\b'
            ],
            'unrealistic_salary': [
                r'\$[0-9]{4,5}\s*monthly\b', r'\$[0-9]{4,}\s*per month\b',
                r'earn\s*\$\d+,\d+\s*monthly', r'\$[0-9]{5,}\s*from home'
            ],
            'personal_contact': [
                r'@gmail\.com\b', r'@yahoo\.com\b', r'@hotmail\.com\b',
                r'@aol\.com\b', r'@protonmail\.com\b'
            ],
            'urgency_tactics': [
                r'immediate (start|hiring)', r'urgent (hiring|position)',
                r'start (immediately|right away)', r'asap\b', r'quick start'
            ],
            'upfront_payment': [
                r'pay.*fee', r'payment.*required', r'deposit.*required',
                r'training materials.*\$\d+', r'background check.*\$\d+',
                r'\$\d+.*(fee|payment|deposit)'
            ],
            'no_qualifications': [
                r'no experience (required|needed)', r'no skills (required|needed)',
                r'no background (required|needed)', r'no degree (required|needed)',
                r'no certification (required|needed)'
            ],
            'pyramid_scheme': [
                r'recruit.*friends', r'multi.level', r'mlm\b',
                r'pyramid scheme', r'downline', r'recruiting.*team'
            ]
        }
        
        for flag_name, patterns in red_flag_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Additional context checks to reduce false positives
                    if flag_name == 'pyramid_scheme':
                        # Make sure it's actually about recruiting, not legitimate HR
                        if not any(word in text_lower for word in ['hr', 'human resources', 'talent acquisition']):
                            red_flags.append(flag_name)
                            break
                    elif flag_name == 'upfront_payment':
                        # Make sure it's actually requesting payment
                        if not any(word in text_lower for word in ['salary', 'compensation', 'benefits']):
                            red_flags.append(flag_name)
                            break
                    else:
                        red_flags.append(flag_name)
                        break
        
        return red_flags
    
    def analyze_job_enhanced(self, job_text):
        """Enhanced analysis with better false positive filtering"""
        start_time = time.time()
        
        # Retrieve similar patterns
        similar_patterns = self.retrieve_similar_patterns(job_text, k=5)
        
        # Calculate weighted scores with confidence boosts
        fake_score = 0
        real_score = 0
        red_flags_found = set()
        reasoning_parts = []
        
        for pattern in similar_patterns:
            weight = pattern['similarity_score'] * pattern.get('confidence_boost', 1.0)
            
            if pattern['label'] == 'fake':
                fake_score += weight
                # Add red flags from similar scams
                red_flags_found.update(pattern['red_flags'])
                if pattern['similarity_score'] > 0.4:
                    reasoning_parts.append(f"Matches scam pattern: {pattern['text'][:80]}...")
            else:
                real_score += weight
                if pattern['similarity_score'] > 0.4:
                    reasoning_parts.append(f"Matches legitimate pattern: {pattern['text'][:80]}...")
        
        # Calculate final scores
        total_score = fake_score + real_score
        if total_score > 0:
            fake_confidence = fake_score / total_score
        else:
            fake_confidence = 0.5
        
        # Enhanced red flag detection with better filtering
        detected_red_flags = self.extract_detailed_red_flags_from_text(job_text)
        
        # Filter out false positives based on context
        filtered_red_flags = []
        for flag in detected_red_flags:
            if self.is_valid_red_flag(flag, job_text):
                filtered_red_flags.append(flag)
        
        red_flags_found.update(filtered_red_flags)
        
        # Adjust confidence based on red flags (but be more conservative)
        red_flag_penalty = len(red_flags_found) * 0.05  # Reduced from 0.1
        fake_confidence = min(0.95, fake_confidence + red_flag_penalty)
        
        # Final decision with higher threshold for "fake"
        if fake_confidence > 0.7:  # Increased from 0.5
            prediction = 'fake'
        else:
            prediction = 'real'
        
        final_confidence = max(fake_confidence, 1 - fake_confidence)
        
        # Build comprehensive reasoning
        if red_flags_found:
            reasoning_parts.append(f"Detected red flags: {', '.join(list(red_flags_found)[:3])}")
        
        reasoning = " | ".join(reasoning_parts[:3]) if reasoning_parts else "No strong patterns detected"
        
        return {
            'prediction': prediction,
            'confidence': final_confidence,
            'processing_time': time.time() - start_time,
            'method': 'Accurate RAG',
            'red_flags': list(red_flags_found),
            'reasoning': reasoning,
            'patterns_matched': len([p for p in similar_patterns if p['similarity_score'] > 0.3])
        }
    
    def is_valid_red_flag(self, flag, job_text):
        """Check if a red flag is valid in context"""
        text_lower = job_text.lower()
        
        if flag == 'pyramid_scheme':
            # Don't flag legitimate HR/recruiting roles
            return not any(term in text_lower for term in 
                          ['hr ', 'human resources', 'talent acquisition', 'recruiter'])
        
        elif flag == 'upfront_payment':
            # Don't flag salary/compensation discussions
            return not any(term in text_lower for term in
                          ['salary', 'compensation', 'benefits', 'pay range'])
        
        elif flag == 'unrealistic_salary':
            # Only flag if it's clearly unrealistic
            return any(term in text_lower for term in
                      ['$8000 monthly', '$10000 monthly', '$5000 weekly'])
        
        return True

# Test the improved system
if __name__ == "__main__":
    print("🧪 TESTING ACCURATE RAG SYSTEM")
    
    accurate_rag = AccurateJobScamRAG()
    
    # Test with the problematic job description
    test_job = """Software Engineer - Middleware/API (Remote)
    Sumitomo Mitsui Banking Corporation
    $104,000 - $170,000 a year - Full-time
    Join our mission to create a completely new, 100% digital bank...
    PRINCIPAL DUTIES AND RESPONSIBILITES:
    Develop scalable, high-performance application features using modern design patterns
    Write clean code, unit tests, and participate in code reviews
    POSITION SPECIFICATIONS:
    Bachelor's degree in Computer Science or equivalent
    2+ years of experience with Java and functional programming"""
    
    print("🔍 Analyzing legitimate job...")
    result = accurate_rag.analyze_job_enhanced(test_job)
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Red flags: {result['red_flags']}")
    print(f"Reasoning: {result['reasoning']}")
    
    # Also test with an obvious scam
    scam_job = """Work from home data entry. No experience needed! 
    Earn $8,000 monthly. Send your Social Security Number to quickhire@gmail.com"""
    
    print("\n🔍 Analyzing obvious scam...")
    scam_result = accurate_rag.analyze_job_enhanced(scam_job)
    print(f"Prediction: {scam_result['prediction'].upper()}")
    print(f"Confidence: {scam_result['confidence']:.1%}")
    print(f"Red flags: {scam_result['red_flags']}")
    print(f"Reasoning: {scam_result['reasoning']}")