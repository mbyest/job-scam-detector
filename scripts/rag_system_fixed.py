import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json

class JobScamRAG:
    def __init__(self, knowledge_base_path='../data/fake_job_postings.csv'):
        print("Loading RAG system...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.index = self.build_index()
        print("RAG system ready!")
    
    def safe_get_text(self, row, field, default="Unknown", max_length=100):
        """Safely extract text from row, handling NaN values"""
        value = row.get(field)
        if pd.isna(value) or value is None:
            return default
        return str(value)[:max_length]
    
    def load_knowledge_base(self, path):
        """Load known scam patterns as knowledge base"""
        df = pd.read_csv(path)
        
        # Fill NaN values with empty strings
        df = df.fillna('')
        
        # Create knowledge entries with scam patterns
        knowledge_entries = []
        
        # Add real scam examples from your dataset
        fake_jobs = df[df['fraudulent'] == 1]
        print(f"Found {len(fake_jobs)} fake jobs in dataset")
        
        for _, row in fake_jobs.iterrows():
            title = self.safe_get_text(row, 'title', 'No Title')
            company = self.safe_get_text(row, 'company_profile', 'No Company Info', 100)
            description = self.safe_get_text(row, 'description', 'No Description', 200)
            
            entry = {
                'text': f"SCAM: {title}. Company: {company}. Description: {description}",
                'red_flags': self.extract_red_flags(row),
                'source': 'Kaggle Dataset',
                'label': 'fake'
            }
            knowledge_entries.append(entry)
        
        # Add some real job examples for contrast
        real_jobs = df[df['fraudulent'] == 0]
        sample_size = min(100, len(real_jobs))
        real_samples = real_jobs.sample(sample_size, random_state=42)
        
        for _, row in real_samples.iterrows():
            title = self.safe_get_text(row, 'title', 'No Title')
            company = self.safe_get_text(row, 'company_profile', 'No Company Info', 100)
            description = self.safe_get_text(row, 'description', 'No Description', 200)
            
            entry = {
                'text': f"REAL: {title}. Company: {company}. Description: {description}",
                'red_flags': [],
                'source': 'Kaggle Dataset', 
                'label': 'real'
            }
            knowledge_entries.append(entry)
        
        # Add common scam patterns
        common_scams = [
            {
                'text': "SCAM: Request for sensitive information like SSN, bank details, or upfront payments",
                'red_flags': ["sensitive_info_request", "upfront_payment"],
                'source': 'Common Scam Pattern',
                'label': 'fake'
            },
            {
                'text': "SCAM: Unrealistically high salary for minimal work or no experience required",
                'red_flags': ["unrealistic_salary", "no_experience"],
                'source': 'Common Scam Pattern',
                'label': 'fake'
            },
            {
                'text': "SCAM: Vague company information or personal email contacts instead of company domain",
                'red_flags': ["vague_company", "personal_contact"],
                'source': 'Common Scam Pattern',
                'label': 'fake'
            },
            {
                'text': "REAL: Professional job posting with specific requirements and company details",
                'red_flags': [],
                'source': 'Common Legitimate Pattern',
                'label': 'real'
            }
        ]
        
        knowledge_entries.extend(common_scams)
        print(f"Loaded {len(knowledge_entries)} knowledge entries")
        return knowledge_entries
    
    def extract_red_flags(self, job_row):
        """Extract specific red flags from job posting"""
        red_flags = []
        
        # Safely get text fields
        company_profile = self.safe_get_text(job_row, 'company_profile', '')
        description = self.safe_get_text(job_row, 'description', '')
        requirements = self.safe_get_text(job_row, 'requirements', '')
        
        text = f"{company_profile} {description} {requirements}".lower()
        
        if any(term in text for term in ['ssn', 'social security', 'bank details', 'bank account', 'routing number']):
            red_flags.append("sensitive_info_request")
        if any(term in text for term in ['$5000', '$8000', '$10000', 'high salary', 'earn much']):
            red_flags.append("unrealistic_salary")
        if any(term in text for term in ['@gmail.com', '@yahoo.com', '@hotmail.com']):
            red_flags.append("personal_contact")
        if any(term in text for term in ['immediately', 'urgent', 'quick start', 'asap']):
            red_flags.append("urgency")
        if any(term in text for term in ['payment', 'fee', 'deposit', 'background check fee']):
            red_flags.append("upfront_payment")
        
        return red_flags
    
    def build_index(self):
        """Build FAISS index for efficient similarity search"""
        texts = [entry['text'] for entry in self.knowledge_base]
        embeddings = self.encoder.encode(texts)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index
    
    def retrieve_similar_patterns(self, query, k=3):
        """Retrieve similar patterns for a job posting"""
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar patterns
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.knowledge_base):
                results.append({
                    **self.knowledge_base[idx],
                    'similarity_score': float(score)
                })
        
        return results

# Initialize RAG system
rag_system = JobScamRAG()
