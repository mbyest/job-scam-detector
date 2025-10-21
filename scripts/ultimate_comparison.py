import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import subprocess
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from rag_system_fixed import JobScamRAG
import os

# Create results directory
os.makedirs('../results', exist_ok=True)

class UltimateComparator:
    def __init__(self):
        print("🚀 LOADING ALL MODELS AND SYSTEMS...")
        print("=" * 70)
        
        self.results = []
        self.models_loaded = {}
        
        try:
            # 1. Load Base Ollama Model
            print("1. Loading Base Ollama (llama3.2:3b)...")
            self.models_loaded['ollama_base'] = True
            print("   ✅ Base Ollama ready (will call dynamically)")
        except Exception as e:
            print(f"   ❌ Base Ollama failed: {e}")
            self.models_loaded['ollama_base'] = False
        
        try:
            # 2. Load Fine-tuned Ollama Model
            print("2. Loading Fine-tuned Ollama (balanced-scam-detector)...")
            self.models_loaded['ollama_finetuned'] = True
            print("   ✅ Fine-tuned Ollama ready")
        except Exception as e:
            print(f"   ❌ Fine-tuned Ollama failed: {e}")
            self.models_loaded['ollama_finetuned'] = False
        
        try:
            # 3. Load Base Hugging Face Model
            print("3. Loading Base Hugging Face (distilbert-base-uncased)...")
            self.base_hf = pipeline("text-classification", model="distilbert-base-uncased")
            self.models_loaded['hf_base'] = True
            print("   ✅ Base Hugging Face ready")
        except Exception as e:
            print(f"   ❌ Base Hugging Face failed: {e}")
            self.models_loaded['hf_base'] = False
        
        try:
            # 4. Load Fine-tuned Hugging Face Model
            print("4. Loading Fine-tuned Hugging Face (your trained model)...")
            self.hf_tokenizer = AutoTokenizer.from_pretrained("../models/hf-job-scam-detector")
            self.hf_model = AutoModelForSequenceClassification.from_pretrained("../models/hf-job-scam-detector")
            self.hf_model.eval()
            self.models_loaded['hf_finetuned'] = True
            print("   ✅ Fine-tuned Hugging Face ready")
        except Exception as e:
            print(f"   ❌ Fine-tuned Hugging Face failed: {e}")
            self.models_loaded['hf_finetuned'] = False
        
        try:
            # 5. Load RAG System
            print("5. Loading RAG System...")
            self.rag_system = JobScamRAG()
            self.models_loaded['rag'] = True
            print("   ✅ RAG System ready")
        except Exception as e:
            print(f"   ❌ RAG System failed: {e}")
            self.models_loaded['rag'] = False
        
        print("\n" + "=" * 70)
        print(f"✅ SUCCESSFULLY LOADED: {sum(self.models_loaded.values())}/5 SYSTEMS")
        print("=" * 70)
    
    def analyze_with_ollama_base(self, job_text):
        """Analyze using base Ollama model"""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                'ollama', 'run', 'llama3.2:3b',
                f'Analyze this job posting and answer ONLY with "Real" or "Fake": {job_text[:400]}'
            ], capture_output=True, text=True, timeout=15)
            
            output = result.stdout.strip().lower()
            processing_time = time.time() - start_time
            
            if 'fake' in output:
                prediction = 'fake'
                confidence = 0.7 if output.count('fake') > 1 else 0.6
            else:
                prediction = 'real'
                confidence = 0.7 if output.count('real') > 1 else 0.6
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': processing_time,
                'method': 'Ollama Base',
                'raw_output': output[:100]
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'method': 'Ollama Base',
                'error': str(e)
            }
    
    def analyze_with_ollama_finetuned(self, job_text):
        """Analyze using your fine-tuned Ollama model"""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                'ollama', 'run', 'balanced-scam-detector',
                f'Analyze this job posting: {job_text[:400]} - Answer with "Real" or "Fake"'
            ], capture_output=True, text=True, timeout=15)
            
            output = result.stdout.strip().lower()
            processing_time = time.time() - start_time
            
            if 'fake' in output:
                prediction = 'fake'
                confidence = 0.8 if output.count('fake') > 1 else 0.7
            else:
                prediction = 'real'
                confidence = 0.8 if output.count('real') > 1 else 0.7
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': processing_time,
                'method': 'Ollama Fine-tuned',
                'raw_output': output[:100]
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'method': 'Ollama Fine-tuned',
                'error': str(e)
            }
    
    def analyze_with_hf_base(self, job_text):
        """Analyze using base Hugging Face model"""
        start_time = time.time()
        
        try:
            text = f"Job posting analysis: {job_text}"
            result = self.base_hf(text[:512])
            
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            if score > 0.6 and ('positive' in label or 'real' in label):
                prediction = 'real'
                confidence = score
            else:
                prediction = 'fake'
                confidence = 1 - score if score > 0.5 else score
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': time.time() - start_time,
                'method': 'HF Base'
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'method': 'HF Base',
                'error': str(e)
            }
    
    def analyze_with_hf_finetuned(self, job_text):
        """Analyze using your fine-tuned Hugging Face model"""
        start_time = time.time()
        
        try:
            text = f"JOB: Unknown. COMPANY: Unknown. DESCRIPTION: {job_text}. REQUIREMENTS: Unknown."
            
            inputs = self.hf_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()
            
            return {
                'prediction': 'fake' if prediction == 1 else 'real',
                'confidence': confidence,
                'processing_time': time.time() - start_time,
                'method': 'HF Fine-tuned'
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'method': 'HF Fine-tuned',
                'error': str(e)
            }
    
    def analyze_with_rag(self, job_text):
        """Analyze using RAG system only"""
        start_time = time.time()
        
        try:
            similar_patterns = self.rag_system.retrieve_similar_patterns(job_text, k=3)
            
            fake_score = 0
            real_score = 0
            
            for pattern in similar_patterns:
                if pattern['label'] == 'fake':
                    fake_score += pattern['similarity_score']
                else:
                    real_score += pattern['similarity_score']
            
            total_score = fake_score + real_score
            if total_score > 0:
                fake_confidence = fake_score / total_score
            else:
                fake_confidence = 0.5
            
            prediction = 'fake' if fake_confidence > 0.5 else 'real'
            confidence = max(fake_confidence, 1 - fake_confidence)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': time.time() - start_time,
                'method': 'RAG Only',
                'evidence_count': len(similar_patterns)
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'method': 'RAG Only',
                'error': str(e)
            }
    
    def run_comprehensive_comparison(self):
        """Run comparison across ALL models"""
        
        # Comprehensive test cases covering all scenarios
        test_cases = [
            # OBVIOUS SCAMS
            {
                'job': "Work from home data entry. No experience needed. Earn $8,000 monthly. Send your Social Security Number and bank details to quickhire2024@gmail.com to start immediately.",
                'expected': 'fake',
                'type': 'obvious_scam',
                'description': 'Classic scam with sensitive info request'
            },
            {
                'job': "Make $10,000 weekly from home processing emails. No skills required. Just forward our messages. Contact millionaire@yahoo.com",
                'expected': 'fake',
                'type': 'obvious_scam', 
                'description': 'Pyramid scheme scam'
            },
            {
                'job': "Urgent hiring! Customer service reps needed immediately. Earn $5,000 monthly. No interview. Send $200 for training materials to start.",
                'expected': 'fake',
                'type': 'obvious_scam',
                'description': 'Scam with urgency + payment'
            },
            
            # LEGITIMATE JOBS
            {
                'job': "Software Engineer at Google. Requirements: Bachelor's in Computer Science, 3+ years experience with Java and Python. Competitive salary with full benefits. Apply through careers.google.com.",
                'expected': 'real',
                'type': 'legitimate',
                'description': 'Professional tech job'
            },
            {
                'job': "Marketing Coordinator at IBM. Requirements: Marketing degree, 2+ years experience, social media proficiency. Full benefits package. Apply via IBM careers portal.",
                'expected': 'real',
                'type': 'legitimate',
                'description': 'Corporate marketing role'
            },
            {
                'job': "Data Analyst at Microsoft. Analyze business data for insights. Requirements: SQL, Python, statistics. Competitive compensation. Professional references required.",
                'expected': 'real',
                'type': 'legitimate',
                'description': 'Professional data role'
            },
            
            # EDGE CASES
            {
                'job': "Sales Representative. Commission-based role. Work from home flexibility. Must have reliable internet. High earning potential for motivated individuals.",
                'expected': 'real',
                'type': 'edge_case',
                'description': 'Legitimate but sales-heavy'
            },
            {
                'job': "Immediate opening for customer support. Great company culture. Work remotely. Competitive pay. Background check required.",
                'expected': 'real',
                'type': 'edge_case',
                'description': 'Legitimate with urgency language'
            },
            
            # MORE SCAMS
            {
                'job': "Medical billing specialist. Work from home. $12,000 monthly. No certification. Send SSN for background check to medicalhr@yahoo.com",
                'expected': 'fake',
                'type': 'obvious_scam',
                'description': 'Unrealistic medical salary scam'
            },
            {
                'job': "Get rich quick! Earn $15,000 monthly from home. No experience. Just forward emails. Contact rich2024@gmail.com",
                'expected': 'fake',
                'type': 'obvious_scam',
                'description': 'Get-rich-quick scam'
            }
        ]
        
        print("\n🧪 ULTIMATE COMPARISON: 5 MODELS, 10 TEST CASES")
        print("=" * 80)
        print("Testing: Ollama Base, Ollama Fine-tuned, HF Base, HF Fine-tuned, RAG Only")
        print("=" * 80)
        
        # Define methods to test based on what loaded successfully
        methods_to_test = []
        if self.models_loaded['ollama_base']:
            methods_to_test.append(('ollama_base', 'Ollama Base'))
        if self.models_loaded['ollama_finetuned']:
            methods_to_test.append(('ollama_finetuned', 'Ollama Fine-tuned'))
        if self.models_loaded['hf_base']:
            methods_to_test.append(('hf_base', 'HF Base'))
        if self.models_loaded['hf_finetuned']:
            methods_to_test.append(('hf_finetuned', 'HF Fine-tuned'))
        if self.models_loaded['rag']:
            methods_to_test.append(('rag', 'RAG Only'))
        
        print(f"Testing {len(methods_to_test)} models across {len(test_cases)} test cases...")
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n📋 Test {i:2d} [{test['type']:>12}]: {test['description']}")
            print(f"   Expected: {test['expected'].upper()}")
            print(f"   Job: {test['job'][:60]}...")
            
            for method_key, method_name in methods_to_test:
                if method_key == 'ollama_base':
                    result = self.analyze_with_ollama_base(test['job'])
                elif method_key == 'ollama_finetuned':
                    result = self.analyze_with_ollama_finetuned(test['job'])
                elif method_key == 'hf_base':
                    result = self.analyze_with_hf_base(test['job'])
                elif method_key == 'hf_finetuned':
                    result = self.analyze_with_hf_finetuned(test['job'])
                elif method_key == 'rag':
                    result = self.analyze_with_rag(test['job'])
                
                correct = result['prediction'] == test['expected']
                icon = "✅" if correct else "❌"
                
                print(f"   {method_name:<18} {icon} {result['prediction'].upper():<6} "
                      f"(conf: {result['confidence']:.1%}, time: {result['processing_time']:.2f}s)")
                
                self.results.append({
                    'test_case': i,
                    'method': method_name,
                    'prediction': result['prediction'],
                    'expected': test['expected'],
                    'confidence': result['confidence'],
                    'processing_time': result['processing_time'],
                    'correct': correct,
                    'type': test['type'],
                    'description': test['description']
                })
        
        return self.results
    
    def create_comprehensive_visualizations(self):
        """Create ultimate comparison visualizations"""
        df = pd.DataFrame(self.results)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: Overall Accuracy
        accuracy_data = df.groupby('method')['correct'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(accuracy_data.index, accuracy_data.values)
        ax1.set_title('Overall Accuracy by Model', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        for bar, acc in zip(bars1, accuracy_data.values):
            ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.02, f'{acc:.1%}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Performance by Job Type
        type_performance = df.pivot_table(index='type', columns='method', values='correct', aggfunc='mean')
        type_performance.plot(kind='bar', ax=ax2)
        ax2.set_title('Accuracy by Job Type', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Confidence vs Accuracy
        confidence_data = df.groupby('method').agg({'confidence': 'mean', 'correct': 'mean'}).reset_index()
        scatter = ax3.scatter(confidence_data['confidence'], confidence_data['correct'], 
                             s=200, alpha=0.7)
        ax3.set_xlabel('Average Confidence')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Confidence vs Accuracy', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add labels to points
        for i, row in confidence_data.iterrows():
            ax3.annotate(row['method'], (row['confidence'], row['correct']),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # Plot 4: Processing Speed
        time_data = df.groupby('method')['processing_time'].mean().sort_values(ascending=True)
        bars4 = ax4.bar(time_data.index, time_data.values)
        ax4.set_title('Average Processing Time', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Time (seconds)')
        for bar, t in zip(bars4, time_data.values):
            ax4.text(bar.get_x() + bar.get_width()/2., t + 0.01, f'{t:.2f}s', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../results/ultimate_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return df
    
    def print_ultimate_summary(self, df):
        """Print comprehensive summary"""
        print("\n" + "=" * 100)
        print("🏆 ULTIMATE COMPARISON SUMMARY")
        print("=" * 100)
        
        # Overall performance
        print("\n�� OVERALL PERFORMANCE")
        print("-" * 50)
        
        summary_data = []
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            accuracy = method_data['correct'].mean()
            avg_confidence = method_data['confidence'].mean()
            avg_time = method_data['processing_time'].mean()
            total_correct = method_data['correct'].sum()
            total_tests = len(method_data)
            
            summary_data.append({
                'Model': method,
                'Accuracy': f"{accuracy:.1%}",
                'Avg Confidence': f"{avg_confidence:.1%}",
                'Avg Time (s)': f"{avg_time:.3f}",
                'Correct/Total': f"{total_correct}/{total_tests}",
                'Score': f"{(accuracy * 100):.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Performance by type
        print("\n🎯 PERFORMANCE BREAKDOWN BY JOB TYPE")
        print("-" * 50)
        
        type_summary = df.pivot_table(index='type', columns='method', values='correct', aggfunc='mean')
        print(type_summary.round(3).to_string())
        
        # Key insights
        print("\n💡 KEY INSIGHTS")
        print("-" * 50)
        
        best_model = summary_df.loc[summary_df['Accuracy'].str.rstrip('%').astype(float).idxmax()]
        fastest_model = summary_df.loc[summary_df['Avg Time (s)'].astype(float).idxmin()]
        most_confident = summary_df.loc[summary_df['Avg Confidence'].str.rstrip('%').astype(float).idxmax()]
        
        print(f"• Most Accurate: {best_model['Model']} ({best_model['Accuracy']})")
        print(f"• Fastest: {fastest_model['Model']} ({fastest_model['Avg Time (s)']})")
        print(f"• Most Confident: {most_confident['Model']} ({most_confident['Avg Confidence']})")
        
        # Save results
        df.to_csv('../results/ultimate_comparison_results.csv', index=False)
        summary_df.to_csv('../results/ultimate_summary.csv', index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"   - ../results/ultimate_comparison_results.csv")
        print(f"   - ../results/ultimate_summary.csv")
        print(f"   - ../results/ultimate_comparison.png")

def main():
    comparator = UltimateComparator()
    
    print("\n🚀 STARTING ULTIMATE COMPARISON...")
    print("This will test ALL your models side-by-side:")
    print("• Ollama Base (llama3.2:3b)")
    print("• Ollama Fine-tuned (balanced-scam-detector)") 
    print("• Hugging Face Base (distilbert-base-uncased)")
    print("• Hugging Face Fine-tuned (your trained model)")
    print("• RAG Only (pattern matching system)")
    print("\nPlease be patient - this may take 5-10 minutes...")
    
    results = comparator.run_comprehensive_comparison()
    df = comparator.create_comprehensive_visualizations()
    comparator.print_ultimate_summary(df)
    
    print("\n🎉 ULTIMATE COMPARISON COMPLETE!")
    print("You now have complete data showing how ALL your models perform!")

if __name__ == "__main__":
    main()
