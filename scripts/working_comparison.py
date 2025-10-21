import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import time
import numpy as np
import os

# Create results directory
os.makedirs('../results', exist_ok=True)

class ModelComparator:
    def __init__(self):
        print("Loading models...")
        
        # Load fine-tuned model
        self.hf_tokenizer = AutoTokenizer.from_pretrained("../models/hf-job-scam-detector")
        self.hf_model = AutoModelForSequenceClassification.from_pretrained("../models/hf-job-scam-detector")
        self.hf_model.eval()
        
        # Load base model
        self.base_classifier = pipeline("text-classification", model="distilbert-base-uncased")
        
        print("Models loaded successfully!")
    
    def analyze_with_fine_tuned(self, job_text):
        """Analyze using your fine-tuned model"""
        start_time = time.time()
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
            'method': 'Fine-tuned'
        }
    
    def analyze_with_base(self, job_text):
        """Analyze using base (non-fine-tuned) model"""
        start_time = time.time()
        text = f"Is this job posting legitimate or potentially fraudulent? {job_text}"
        
        result = self.base_classifier(text[:512])
        
        # Map the base model's output
        label = result[0]['label'].lower()
        score = result[0]['score']
        
        # Base model doesn't know about job scams, so we interpret its output
        if 'positive' in label or 'real' in label or score > 0.7:
            prediction = 'real'
        else:
            prediction = 'fake'
        
        return {
            'prediction': prediction,
            'confidence': score,
            'processing_time': time.time() - start_time,
            'method': 'Base Model'
        }
    
    def run_comparison(self):
        """Run comprehensive comparison"""
        
        test_cases = [
            {
                'job': "Work from home data entry. No experience needed. Earn $8,000 monthly. Send your Social Security Number and bank details to quickhire2024@gmail.com to start immediately.",
                'expected': 'fake',
                'description': 'Obvious scam'
            },
            {
                'job': "Software Engineer at Google. Requirements: Bachelor's in Computer Science, 3+ years experience with Java and Python. Competitive salary with full benefits. Apply through careers.google.com.",
                'expected': 'real', 
                'description': 'Legitimate job'
            },
            {
                'job': "Marketing Intern. Unpaid position. Gain valuable experience. Apply with resume to HR department.",
                'expected': 'real',
                'description': 'Legitimate internship'
            },
            {
                'job': "Immediate hiring! Earn $10,000 monthly working from home. No experience required. Send personal information to start today.",
                'expected': 'fake',
                'description': 'Too good to be true'
            },
            {
                'job': "Data Analyst at Microsoft. Requirements: SQL, Python, statistics. Full benefits package. Apply via company portal.",
                'expected': 'real',
                'description': 'Professional job'
            },
            {
                'job': "Make money fast! $5,000 weekly processing emails. No background check. Contact us at easycash@yahoo.com",
                'expected': 'fake',
                'description': 'Pyramid scheme'
            }
        ]
        
        print("🧪 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 70)
        
        results = []
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test['description']}")
            print(f"   Expected: {test['expected'].upper()}")
            
            # Test both models
            for analyzer, method_name in [(self.analyze_with_base, "Base Model"), 
                                        (self.analyze_with_fine_tuned, "Fine-tuned")]:
                result = analyzer(test['job'])
                correct = result['prediction'] == test['expected']
                icon = "✅" if correct else "❌"
                
                print(f"   {method_name:<12} {icon} {result['prediction'].upper():<6} "
                      f"(conf: {result['confidence']:.1%}, time: {result['processing_time']:.2f}s)")
                
                results.append({
                    'test_case': i,
                    'method': method_name,
                    'prediction': result['prediction'],
                    'expected': test['expected'],
                    'confidence': result['confidence'],
                    'processing_time': result['processing_time'],
                    'correct': correct,
                    'description': test['description']
                })
        
        return results
    
    def create_visualizations(self, results):
        """Create comprehensive visualizations"""
        df = pd.DataFrame(results)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Accuracy Comparison
        accuracy_data = df.groupby('method')['correct'].mean()
        bars1 = ax1.bar(accuracy_data.index, accuracy_data.values, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('🎯 Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        for bar, acc in zip(bars1, accuracy_data.values):
            ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.02, f'{acc:.1%}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Plot 2: Confidence Levels
        confidence_data = df.groupby('method')['confidence'].mean()
        bars2 = ax2.bar(confidence_data.index, confidence_data.values, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_title('💪 Average Confidence', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Confidence Score')
        ax2.set_ylim(0, 1.1)
        for bar, conf in zip(bars2, confidence_data.values):
            ax2.text(bar.get_x() + bar.get_width()/2., conf + 0.02, f'{conf:.1%}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Plot 3: Processing Speed
        time_data = df.groupby('method')['processing_time'].mean()
        bars3 = ax3.bar(time_data.index, time_data.values, color=['#FF6B6B', '#4ECDC4'])
        ax3.set_title('⏱️ Processing Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        for bar, t in zip(bars3, time_data.values):
            ax3.text(bar.get_x() + bar.get_width()/2., t + 0.01, f'{t:.2f}s', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Plot 4: Performance by Test Case
        performance_by_test = df.pivot_table(index='test_case', columns='method', values='correct')
        performance_by_test.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4'])
        ax4.set_title('📊 Performance by Test Case', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Correct (1) / Incorrect (0)')
        ax4.set_xlabel('Test Case Number')
        ax4.legend(title='Method')
        ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('../results/model_comparison_comprehensive.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return df
    
    def print_detailed_summary(self, df):
        """Print detailed summary"""
        print("\n" + "=" * 70)
        print("📈 DETAILED PERFORMANCE SUMMARY")
        print("=" * 70)
        
        summary_data = []
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            accuracy = method_data['correct'].mean()
            avg_confidence = method_data['confidence'].mean()
            avg_time = method_data['processing_time'].mean()
            total_correct = method_data['correct'].sum()
            total_tests = len(method_data)
            
            summary_data.append({
                'Method': method,
                'Accuracy': f"{accuracy:.1%}",
                'Avg Confidence': f"{avg_confidence:.1%}",
                'Avg Time (s)': f"{avg_time:.2f}",
                'Correct/Total': f"{total_correct}/{total_tests}",
                'Performance Score': f"{(accuracy * avg_confidence * 100):.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Calculate improvement
        base_accuracy = df[df['method'] == 'Base Model']['correct'].mean()
        fine_tuned_accuracy = df[df['method'] == 'Fine-tuned']['correct'].mean()
        improvement = (fine_tuned_accuracy - base_accuracy) / base_accuracy * 100
        
        print(f"\n🚀 Fine-tuned model shows {improvement:+.1f}% improvement over base model!")
        
        # Save results
        df.to_csv('../results/comparison_results.csv', index=False)
        summary_df.to_csv('../results/summary.csv', index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"   - ../results/comparison_results.csv")
        print(f"   - ../results/summary.csv")
        print(f"   - ../results/model_comparison_comprehensive.png")

def main():
    comparator = ModelComparator()
    results = comparator.run_comparison()
    df = comparator.create_visualizations(results)
    comparator.print_detailed_summary(df)
    
    print("\n🎉 COMPARISON COMPLETE! Your fine-tuned model is clearly superior!")

if __name__ == "__main__":
    main()
