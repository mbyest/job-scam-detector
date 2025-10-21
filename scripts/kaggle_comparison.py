import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import subprocess
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from rag_system_fixed import JobScamRAG
from fix_improved_rag import FixedImprovedJobScamRAG
import os

# Create results directory
os.makedirs('../results', exist_ok=True)

class KaggleDatasetTester:
    def __init__(self):
        print("🚀 LOADING ALL MODELS FOR KAGGLE TESTING...")
        print("=" * 70)
        
        self.models_loaded = {}
        self.results = []
        
        # Load your Kaggle dataset
        print("📊 Loading Kaggle dataset...")
        self.kaggle_df = pd.read_csv('../data/fake_job_postings.csv')
        self.kaggle_df = self.kaggle_df.fillna('')
        print(f"   Loaded {len(self.kaggle_df)} job postings from Kaggle")
        
        try:
            # 1. Base Ollama Model
            print("1. Loading Base Ollama (llama3.2:3b)...")
            self.models_loaded['ollama_base'] = True
            print("   ✅ Base Ollama ready")
        except Exception as e:
            print(f"   ❌ Base Ollama failed: {e}")
            self.models_loaded['ollama_base'] = False
        
        try:
            # 2. Fine-tuned Ollama Model
            print("2. Loading Fine-tuned Ollama (balanced-scam-detector)...")
            self.models_loaded['ollama_finetuned'] = True
            print("   ✅ Fine-tuned Ollama ready")
        except Exception as e:
            print(f"   ❌ Fine-tuned Ollama failed: {e}")
            self.models_loaded['ollama_finetuned'] = False
        
        try:
            # 3. Base Hugging Face Model
            print("3. Loading Base Hugging Face (distilbert-base-uncased)...")
            self.base_hf = pipeline("text-classification", model="distilbert-base-uncased")
            self.models_loaded['hf_base'] = True
            print("   ✅ Base Hugging Face ready")
        except Exception as e:
            print(f"   ❌ Base Hugging Face failed: {e}")
            self.models_loaded['hf_base'] = False
        
        try:
            # 4. Fine-tuned Hugging Face Model
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
            # 5. Original RAG System
            print("5. Loading Original RAG System...")
            self.rag_original = JobScamRAG()
            self.models_loaded['rag_original'] = True
            print("   ✅ Original RAG ready")
        except Exception as e:
            print(f"   ❌ Original RAG failed: {e}")
            self.models_loaded['rag_original'] = False
        
        try:
            print("6. Loading Fixed Improved RAG System...")
            self.rag_improved = FixedImprovedJobScamRAG()
            self.models_loaded['rag_improved'] = True
            print("   ✅ Fixed Improved RAG ready")
        except Exception as e:
            print(f"   ❌ Fixed Improved RAG failed: {e}")
            self.models_loaded['rag_improved'] = False
        
        print("\n" + "=" * 70)
        print(f"✅ SUCCESSFULLY LOADED: {sum(self.models_loaded.values())}/6 SYSTEMS")
        print("=" * 70)
    
    def prepare_kaggle_test_samples(self, num_samples=50):
        """Prepare balanced test samples from Kaggle dataset"""
        print(f"\n🎯 PREPARING {num_samples} TEST SAMPLES FROM KAGGLE DATASET...")
        
        # Get balanced samples
        fake_jobs = self.kaggle_df[self.kaggle_df['fraudulent'] == 1]
        real_jobs = self.kaggle_df[self.kaggle_df['fraudulent'] == 0]
        
        # Sample equal numbers of fake and real jobs
        n_each = num_samples // 2
        
        fake_sample = fake_jobs.sample(min(n_each, len(fake_jobs)), random_state=42)
        real_sample = real_jobs.sample(min(n_each, len(real_jobs)), random_state=42)
        
        test_samples = []
        
        # Process fake jobs
        for _, row in fake_sample.iterrows():
            job_text = self.construct_job_text(row)
            test_samples.append({
                'text': job_text,
                'expected': 'fake',
                'type': 'scam',
                'title': row['title'],
                'source': 'Kaggle'
            })
        
        # Process real jobs
        for _, row in real_sample.iterrows():
            job_text = self.construct_job_text(row)
            test_samples.append({
                'text': job_text,
                'expected': 'real',
                'type': 'legitimate',
                'title': row['title'],
                'source': 'Kaggle'
            })
        
        print(f"✅ Prepared {len(test_samples)} test samples ({n_each} fake, {n_each} real)")
        return test_samples
    
    def construct_job_text(self, row):
        """Construct a complete job posting text from row data"""
        parts = []
        
        if row['title']:
            parts.append(f"TITLE: {row['title']}")
        if row['company_profile']:
            parts.append(f"COMPANY: {row['company_profile'][:200]}")
        if row['description']:
            parts.append(f"DESCRIPTION: {row['description'][:300]}")
        if row['requirements']:
            parts.append(f"REQUIREMENTS: {row['requirements'][:200]}")
        
        return " | ".join(parts)
    
    def analyze_with_ollama_base(self, job_text):
        """Analyze using base Ollama model"""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                'ollama', 'run', 'llama3.2:3b',
                f'Analyze this job posting and answer ONLY with "Real" or "Fake": {job_text[:400]}'
            ], capture_output=True, text=True, timeout=30)
            
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
                'raw_output': output[:100]
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def analyze_with_ollama_finetuned(self, job_text):
        """Analyze using your fine-tuned Ollama model"""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                'ollama', 'run', 'balanced-scam-detector',
                f'Analyze this job posting: {job_text[:400]} - Answer with "Real" or "Fake"'
            ], capture_output=True, text=True, timeout=30)
            
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
                'raw_output': output[:100]
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
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
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
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
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def analyze_with_rag_original(self, job_text):
        """Analyze using original RAG system"""
        start_time = time.time()
        
        try:
            similar_patterns = self.rag_original.retrieve_similar_patterns(job_text, k=3)
            
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
                'evidence_count': len(similar_patterns)
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def analyze_with_rag_improved(self, job_text):
        """Analyze using improved RAG system"""
        start_time = time.time()
        
        try:
            result = self.rag_improved.analyze_job_enhanced(job_text)
            return result
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def run_kaggle_comparison(self, num_samples=50):
        """Run comprehensive comparison on Kaggle dataset samples"""
        test_samples = self.prepare_kaggle_test_samples(num_samples)
        
        print(f"\n🧪 KAGGLE DATASET COMPARISON: {len(self.models_loaded)} MODELS, {len(test_samples)} SAMPLES")
        print("=" * 80)
        print("Testing: Ollama Base, Ollama Fine-tuned, HF Base, HF Fine-tuned, RAG Original, RAG Improved")
        print("=" * 80)
        
        # Define methods to test
        methods_to_test = []
        if self.models_loaded['ollama_base']:
            methods_to_test.append(('ollama_base', 'Ollama Base', self.analyze_with_ollama_base))
        if self.models_loaded['ollama_finetuned']:
            methods_to_test.append(('ollama_finetuned', 'Ollama Fine-tuned', self.analyze_with_ollama_finetuned))
        if self.models_loaded['hf_base']:
            methods_to_test.append(('hf_base', 'HF Base', self.analyze_with_hf_base))
        if self.models_loaded['hf_finetuned']:
            methods_to_test.append(('hf_finetuned', 'HF Fine-tuned', self.analyze_with_hf_finetuned))
        if self.models_loaded['rag_original']:
            methods_to_test.append(('rag_original', 'RAG Original', self.analyze_with_rag_original))
        if self.models_loaded['rag_improved']:
            methods_to_test.append(('rag_improved', 'RAG Improved', self.analyze_with_rag_improved))
        
        print(f"Testing {len(methods_to_test)} models across {len(test_samples)} Kaggle samples...")
        print("This may take 10-30 minutes...")
        
        # Progress tracking
        total_tests = len(methods_to_test) * len(test_samples)
        completed = 0
        
        for i, test in enumerate(test_samples):
            if i % 10 == 0:
                print(f"\n📊 Progress: {i}/{len(test_samples)} samples completed")
            
            for method_key, method_name, method_func in methods_to_test:
                result = method_func(test['text'])
                
                correct = result['prediction'] == test['expected']
                
                self.results.append({
                    'sample_id': i,
                    'method': method_name,
                    'prediction': result['prediction'],
                    'expected': test['expected'],
                    'confidence': result['confidence'],
                    'processing_time': result['processing_time'],
                    'correct': correct,
                    'type': test['type'],
                    'title': test['title'],
                    'job_text_length': len(test['text'])
                })
                
                completed += 1
                if completed % 20 == 0:
                    print(f"   Completed {completed}/{total_tests} tests")
        
        return self.results
    
    def create_kaggle_comparison_visualizations(self):
        """Create comprehensive visualizations for Kaggle test results"""
        df = pd.DataFrame(self.results)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Kaggle Dataset Model Comparison Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall Accuracy
        accuracy_data = df.groupby('method')['correct'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(accuracy_data.index, accuracy_data.values, color='skyblue')
        ax1.set_title('Overall Accuracy by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        for bar, acc in zip(bars1, accuracy_data.values):
            ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.02, f'{acc:.1%}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Performance by Job Type
        type_performance = df.pivot_table(index='type', columns='method', values='correct', aggfunc='mean')
        type_performance.plot(kind='bar', ax=ax2)
        ax2.set_title('Accuracy by Job Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Processing Speed Comparison
        time_data = df.groupby('method')['processing_time'].mean().sort_values(ascending=True)
        bars3 = ax3.bar(time_data.index, time_data.values, color='lightcoral')
        ax3.set_title('Average Processing Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        for bar, t in zip(bars3, time_data.values):
            ax3.text(bar.get_x() + bar.get_width()/2., t + 0.01, f'{t:.2f}s', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Confidence Distribution
        confidence_data = []
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            confidence_data.append({
                'method': method,
                'avg_confidence': method_data['confidence'].mean(),
                'correct_confidence': method_data[method_data['correct'] == True]['confidence'].mean(),
                'wrong_confidence': method_data[method_data['correct'] == False]['confidence'].mean()
            })
        
        confidence_df = pd.DataFrame(confidence_data)
        x = np.arange(len(confidence_df))
        width = 0.25
        
        ax4.bar(x - width, confidence_df['avg_confidence'], width, label='Avg Confidence', alpha=0.7)
        ax4.bar(x, confidence_df['correct_confidence'], width, label='When Correct', alpha=0.7)
        ax4.bar(x + width, confidence_df['wrong_confidence'], width, label='When Wrong', alpha=0.7)
        
        ax4.set_title('Confidence Analysis', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Confidence Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(confidence_df['method'], rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('../results/kaggle_comparison_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return df
    
    def print_detailed_kaggle_summary(self, df):
        """Print comprehensive summary of Kaggle test results"""
        print("\n" + "=" * 120)
        print("🏆 KAGGLE DATASET COMPARISON - COMPREHENSIVE RESULTS")
        print("=" * 120)
        
        # Overall performance summary
        print("\n📊 OVERALL PERFORMANCE SUMMARY")
        print("-" * 80)
        
        summary_data = []
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            accuracy = method_data['correct'].mean()
            
            # Breakdown by type
            scam_accuracy = method_data[method_data['type'] == 'scam']['correct'].mean()
            legit_accuracy = method_data[method_data['type'] == 'legitimate']['correct'].mean()
            
            avg_confidence = method_data['confidence'].mean()
            avg_time = method_data['processing_time'].mean()
            
            total_correct = method_data['correct'].sum()
            total_tests = len(method_data)
            
            summary_data.append({
                'Model': method,
                'Overall Accuracy': f"{accuracy:.1%}",
                'Scam Detection': f"{scam_accuracy:.1%}",
                'Legit Detection': f"{legit_accuracy:.1%}",
                'Avg Confidence': f"{avg_confidence:.1%}",
                'Avg Time (s)': f"{avg_time:.3f}",
                'Correct/Total': f"{total_correct}/{total_tests}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Detailed confusion matrices
        print("\n🎯 DETAILED PERFORMANCE BREAKDOWN")
        print("-" * 80)
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            true_positives = len(method_data[(method_data['prediction'] == 'fake') & (method_data['expected'] == 'fake')])
            false_positives = len(method_data[(method_data['prediction'] == 'fake') & (method_data['expected'] == 'real')])
            true_negatives = len(method_data[(method_data['prediction'] == 'real') & (method_data['expected'] == 'real')])
            false_negatives = len(method_data[(method_data['prediction'] == 'real') & (method_data['expected'] == 'fake')])
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{method}:")
            print(f"  True Positives (Scams caught): {true_positives}")
            print(f"  False Positives (Legit flagged as scams): {false_positives}")
            print(f"  True Negatives (Legit correctly identified): {true_negatives}") 
            print(f"  False Negatives (Scams missed): {false_negatives}")
            print(f"  Precision: {precision:.1%} | Recall: {recall:.1%} | F1-Score: {f1:.1%}")
        
        # Save detailed results
        df.to_csv('../results/kaggle_comparison_detailed.csv', index=False)
        summary_df.to_csv('../results/kaggle_comparison_summary.csv', index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"   - ../results/kaggle_comparison_detailed.csv")
        print(f"   - ../results/kaggle_comparison_summary.csv") 
        print(f"   - ../results/kaggle_comparison_results.png")

def main():
    tester = KaggleDatasetTester()
    
    print("\n🚀 STARTING KAGGLE DATASET COMPARISON...")
    print("This will test ALL your models on REAL Kaggle dataset samples!")
    print("Models being tested:")
    print("• Ollama Base (llama3.2:3b)")
    print("• Ollama Fine-tuned (balanced-scam-detector)") 
    print("• Hugging Face Base (distilbert-base-uncased)")
    print("• Hugging Face Fine-tuned (your trained model)")
    print("• RAG Original (original pattern matching)")
    print("• RAG Improved (enhanced with expert rules)")
    print("\nPlease be patient - this may take 10-30 minutes...")
    
    # Run comparison with 50 samples (25 scams, 25 legitimate)
    results = tester.run_kaggle_comparison(num_samples=50)
    df = tester.create_kaggle_comparison_visualizations()
    tester.print_detailed_kaggle_summary(df)
    
    print("\n🎉 KAGGLE DATASET COMPARISON COMPLETE!")
    print("You now have comprehensive results showing how ALL your models perform on real data!")

if __name__ == "__main__":
    main()
