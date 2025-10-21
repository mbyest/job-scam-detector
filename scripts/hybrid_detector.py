from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from rag_system import rag_system
import re

class HybridJobScamDetector:
    def __init__(self, hf_model_path):
        print("Loading Hybrid Detector...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.rag = rag_system
        print("Hybrid detector ready!")
    
    def analyze_with_hf_only(self, job_text):
        """Analyze using only Hugging Face model"""
        return self._get_model_prediction(job_text)
    
    def analyze_with_rag_only(self, job_text):
        """Analyze using only RAG system"""
        similar_patterns = self.rag.retrieve_similar_patterns(job_text, k=3)
        
        # Simple RAG-based decision
        fake_similarity = 0
        real_similarity = 0
        
        for pattern in similar_patterns:
            if pattern['label'] == 'fake':
                fake_similarity += pattern['similarity_score']
            else:
                real_similarity += pattern['similarity_score']
        
        total_similarity = fake_similarity + real_similarity
        if total_similarity > 0:
            fake_confidence = fake_similarity / total_similarity
        else:
            fake_confidence = 0.5
        
        return {
            'prediction': 'fake' if fake_confidence > 0.5 else 'real',
            'confidence': max(fake_confidence, 1 - fake_confidence),
            'method': 'rag_only',
            'evidence': similar_patterns
        }
    
    def analyze_hybrid(self, job_text):
        """Analyze using hybrid RAG + HF approach"""
        # Get both predictions
        hf_result = self.analyze_with_hf_only(job_text)
        rag_result = self.analyze_with_rag_only(job_text)
        
        # Combine predictions (weighted average)
        hf_weight = 0.7  # Trust HF more since it has 93% accuracy
        rag_weight = 0.3
        
        hf_fake_prob = hf_result['probabilities']['fake']
        rag_fake_prob = rag_result['confidence'] if rag_result['prediction'] == 'fake' else 1 - rag_result['confidence']
        
        combined_fake_prob = (hf_weight * hf_fake_prob + rag_weight * rag_fake_prob)
        
        # Enhanced reasoning
        reasoning = self._build_enhanced_reasoning(hf_result, rag_result, job_text)
        
        return {
            'prediction': 'fake' if combined_fake_prob > 0.5 else 'real',
            'confidence': max(combined_fake_prob, 1 - combined_fake_prob),
            'hf_confidence': hf_result['confidence'],
            'rag_confidence': rag_result['confidence'],
            'combined_fake_probability': combined_fake_prob,
            'method': 'hybrid',
            'reasoning': reasoning,
            'hf_prediction': hf_result['prediction'],
            'rag_prediction': rag_result['prediction'],
            'evidence': rag_result['evidence'],
            'red_flags': self._detect_red_flags(job_text)
        }
    
    def _get_model_prediction(self, job_text):
        """Get prediction from fine-tuned model"""
        text = f"JOB: Unknown. COMPANY: Unknown. DESCRIPTION: {job_text}. REQUIREMENTS: Unknown."
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0][0].item(),
                'fake': probabilities[0][1].item()
            }
        }
    
    def _build_enhanced_reasoning(self, hf_result, rag_result, job_text):
        """Build enhanced reasoning combining both approaches"""
        reasoning_parts = []
        
        # HF model reasoning
        reasoning_parts.append(f"🤖 AI Model: {hf_result['prediction'].upper()} ({hf_result['confidence']:.1%} confidence)")
        
        # RAG evidence
        if rag_result['evidence']:
            reasoning_parts.append("🔍 Similar Patterns Found:")
            for i, evidence in enumerate(rag_result['evidence'][:2], 1):
                if evidence['similarity_score'] > 0.3:
                    label_icon = "🚫" if evidence['label'] == 'fake' else "✅"
                    reasoning_parts.append(f"   {label_icon} {evidence['text'][:80]}... (Similarity: {evidence['similarity_score']:.2f})")
        
        # Red flags
        red_flags = self._detect_red_flags(job_text)
        if red_flags:
            reasoning_parts.append("🚩 Detected Red Flags: " + ", ".join(red_flags))
        
        # Agreement status
        if hf_result['prediction'] == rag_result['prediction']:
            reasoning_parts.append("✅ Both methods agree on this classification")
        else:
            reasoning_parts.append("⚠️ Methods disagree - using weighted combination")
        
        return "\n".join(reasoning_parts)
    
    def _detect_red_flags(self, job_text):
        """Detect specific red flags in job text"""
        text_lower = job_text.lower()
        red_flags = []
        
        red_flag_patterns = {
            'SSN/Bank Request': ['ssn', 'social security', 'bank details', 'bank account', 'routing number'],
            'Unrealistic Salary': ['$5000', '$8000', '$10000', 'high salary', 'earn much', 'get rich'],
            'Personal Contact': ['@gmail.com', '@yahoo.com', '@hotmail.com', 'personal email'],
            'Urgency': ['immediately', 'urgent', 'quick start', 'asap', 'right away'],
            'Upfront Payment': ['payment', 'fee', 'deposit', 'background check fee']
        }
        
        for flag_name, patterns in red_flag_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                red_flags.append(flag_name)
        
        return red_flags

# Initialize hybrid detector
hybrid_detector = HybridJobScamDetector("../models/hf-job-scam-detector")
