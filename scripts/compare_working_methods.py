from hybrid_detector import hybrid_detector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from transformers import pipeline
import os

class WorkingMethodsComparator:
    def __init__(self):
        self.hybrid_detector = hybrid_detector
        self.base_classifier = pipeline(
            "text-classification", 
            model="distilbert-base-uncased",
            tokenizer="distilbert-base-uncased"
        )
        self.results = []
        
    def analyze_with_base_model(self, job_text):
        """Analyze using base (non-fine-tuned) model"""
        start_time = time.time()
        
        try:
            # Base model analysis
            text = f"Is this job posting legitimate or a scam? {job_text}"
            result = self.base_classifier(text[:512])
            
            # Simple mapping - this won't be perfect but gives a baseline
            label = result[0]['label'].lower()
            if 'scam' in label or 'fake' in label or 'fraud' in label:
                prediction = 'fake'
            else:
                prediction = 'real'
                
            confidence = result[0]['score']
            processing_time = time.time() - start_time
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': processing_time,
                'method': 'base_model'
            }
        except Exception as e:
            print(f"Base model error: {e}")
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'method': 'base_model'
            }
    
    def run_comparison(self):
        """Run comparison across working methods"""
        
        test_cases = [
            {
                'job': "Work from home data entry. No experience needed. Earn $8,000 monthly. Send your Social Security Number and bank details to quickhire2024@gmail.com to start immediately.",
                'expected': 'fake',
                'description': 'Obvious scam with multiple red flags'
            },
            {
                'job': "Software Engineer at Google. Requirements: Bachelor's in Computer Science, 3+ years experience with Java and Python. Competitive salary with full benefits. Apply through careers.google.com.",
                'expected': 'real', 
                'description': 'Legitimate professional job'
            },
            {
                'job': "Marketing Intern at Food52. Unpaid internship. Requirements: Experience with content management systems, meticulous editing skills. Apply through professional channels.",
                'expected': 'real',
                'description': 'Legitimate unpaid internship'
            },
            {
                'job': "Immediate hiring! Customer service representatives needed. Work from anywhere. Earn $5,000 monthly. No background check required. Send resume to hiringmanager@personal.com",
                'expected': 'fake',
                'description': 'Scam with urgency and unrealistic salary'
            },
            {
                'job': "Data Analyst at IBM. Requirements: SQL, Python, statistical analysis. Competitive compensation. Apply via IBM careers portal with professional references.",
                'expected': 'real',
                'description': 'Legitimate corporate job'
            }
        ]
        
        print("🧪 COMPARISON: BASE MODEL vs FINE-TUNED vs HYBRID")
        print("=" * 80)
        
        methods = ['base_model', 'hf_only', 'hybrid']
        method_names = {
            'base_model': 'Base Model',
            'hf_only': 'Fine-tuned Only',
            'hybrid': 'Hybrid (Fine-tuned + RAG)'
        }
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test_case['description']}")
            print(f"   Job: {test_case['job'][:70]}...")
            print(f"   Expected: {test_case['expected'].upper()}")
            
            for method in methods:
                if method == 'base_model':
                    result = self.analyze_with_base_model(test_case['job'])
                elif method == 'hf_only':
                    result = self.hybrid_detector.analyze_with_hf_only(test_case['job'])
                    result['method'] = 'hf_only'
                else:  # hybrid
                    result = self.hybrid_detector.analyze_hybrid(test_case['job'])
                    result['method'] = 'hybrid'
                
                correct = result['prediction'] == test_case['expected']
                icon = "✅" if correct else "❌"
                
                print(f"   {method_names[method]:<20} {icon} {result['prediction'].upper():<6} "
                      f"(conf: {result['confidence']:.1%}, time: {result.get('processing_time', 0.2):.2f}s)")
                
                self.results.append({
                    'test_case': i,
                    'method': method,
                    'method_name': method_names[method],
                    'prediction': result['prediction'],
                    'expected': test_case['expected'],
                    'confidence': result['confidence'],
                    'processing_time': result.get('processing_time', 0.2),
                    'correct': correct,
                    'description': test_case['description']
                })
        
        return self.results
    
    def create_plots(self):
        """Create comparison plots"""
        plt.style.use('default')
        sns.set_palette("husl")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        df = pd.DataFrame(self.results)
        
        # Plot 1: Accuracy
        accuracy_data = df.groupby('method_name')['correct'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(accuracy_data.index, accuracy_data.values, color=['#2E8B57', '#4169E1', '#FF6347'])
        ax1.set_title('Accuracy by Method', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        for bar, acc in zip(bars1, accuracy_data.values):
            ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.02, f'{acc:.1%}', 
                    ha='center', va='bottom', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Confidence
        confidence_data = df.groupby('method_name')['confidence'].mean().sort_values(ascending=False)
        bars2 = ax2.bar(confidence_data.index, confidence_data.values, color=['#2E8B57', '#4169E1', '#FF6347'])
        ax2.set_title('Average Confidence', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Confidence Score')
        ax2.set_ylim(0, 1.1)
        for bar, conf in zip(bars2, confidence_data.values):
            ax2.text(bar.get_x() + bar.get_width()/2., conf + 0.02, f'{conf:.1%}', 
                    ha='center', va='bottom', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Speed
        time_data = df.groupby('method_name')['processing_time'].mean().sort_values(ascending=True)
        bars3 = ax3.bar(time_data.index, time_data.values, color=['#2E8B57', '#4169E1', '#FF6347'])
        ax3.set_title('Processing Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        for bar, t in zip(bars3, time_data.values):
            ax3.text(bar.get_x() + bar.get_width()/2., t + 0.01, f'{t:.2f}s', 
                    ha='center', va='bottom', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../results/working_methods_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print summary statistics"""
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 80)
        
        summary_data = []
        for method in df['method_name'].unique():
            method_data = df[df['method_name'] == method]
            accuracy = method_data['correct'].mean()
            avg_confidence = method_data['confidence'].mean()
            avg_time = method_data['processing_time'].mean()
            
            summary_data.append({
                'Method': method,
                'Accuracy': f"{accuracy:.1%}",
                'Avg Confidence': f"{avg_confidence:.1%}",
                'Avg Time (s)': f"{avg_time:.2f}",
                'Correct/Total': f"{method_data['correct'].sum()}/{len(method_data)}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Save results
        df.to_csv('../results/working_methods_results.csv', index=False)
        print(f"\n💾 Results saved to ../results/working_methods_results.csv")

def main():
    os.makedirs('../results', exist_ok=True)
    
    comparator = WorkingMethodsComparator()
    print("🚀 Comparing Base Model vs Fine-tuned vs Hybrid...")
    results = comparator.run_comparison()
    comparator.create_plots()
    comparator.print_summary()

if __name__ == "__main__":
    main()
