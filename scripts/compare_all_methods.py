from hybrid_detector import hybrid_detector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from transformers import pipeline
import subprocess
import json

class ComprehensiveComparator:
    def __init__(self):
        self.hybrid_detector = hybrid_detector
        self.base_classifier = pipeline(
            "text-classification", 
            model="distilbert-base-uncased",  # Base model without fine-tuning
            tokenizer="distilbert-base-uncased"
        )
        self.results = []
        
    def analyze_with_base_model(self, job_text):
        """Analyze using base (non-fine-tuned) model"""
        start_time = time.time()
        
        try:
            # Base model needs proper formatting
            text = f"Is this job legitimate or a scam? {job_text}"
            result = self.base_classifier(text[:512])  # Limit length
            
            # Map base model output to our format
            base_prediction = 'fake' if 'scam' in result[0]['label'].lower() or 'fake' in result[0]['label'].lower() else 'real'
            base_confidence = result[0]['score']
            
            processing_time = time.time() - start_time
            
            return {
                'prediction': base_prediction,
                'confidence': base_confidence,
                'processing_time': processing_time,
                'method': 'base_model'
            }
        except Exception as e:
            return {
                'prediction': 'real',  # Default to safe prediction
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'method': 'base_model',
                'error': str(e)
            }
    
    def analyze_with_ollama(self, job_text):
        """Analyze using your original Ollama model"""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                'ollama', 'run', 'balanced-scam-detector',
                f'Analyze this job posting: {job_text[:500]} - Answer only with "Real" or "Fake"'
            ], capture_output=True, text=True, timeout=10)
            
            output = result.stdout.strip().lower()
            processing_time = time.time() - start_time
            
            if 'fake' in output:
                prediction = 'fake'
                # Estimate confidence based on response clarity
                confidence = 0.7 if output.count('fake') > 1 else 0.6
            else:
                prediction = 'real'
                confidence = 0.7 if output.count('real') > 1 else 0.6
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': processing_time,
                'method': 'ollama',
                'raw_output': output
            }
        except Exception as e:
            return {
                'prediction': 'real',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'method': 'ollama',
                'error': str(e)
            }
    
    def run_comprehensive_comparison(self):
        """Run comparison across all methods"""
        
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
            },
            {
                'job': "Make $10,000 weekly from home. No skills required. Just forward emails. Contact us at easycash2024@yahoo.com with your personal information.",
                'expected': 'fake',
                'description': 'Pyramid scheme scam'
            },
            {
                'job': "Junior Developer at Microsoft. Requirements: Computer Science degree or equivalent experience. Full benefits package. Apply through Microsoft careers website.",
                'expected': 'real',
                'description': 'Legitimate tech job'
            }
        ]
        
        print("🧪 COMPREHENSIVE COMPARISON: 4 METHODS")
        print("=" * 90)
        
        methods = ['base_model', 'ollama', 'hf_only', 'rag_only', 'hybrid']
        method_names = {
            'base_model': 'Base Model',
            'ollama': 'Ollama Fine-tuned', 
            'hf_only': 'HF Fine-tuned',
            'rag_only': 'RAG Only',
            'hybrid': 'Hybrid (HF + RAG)'
        }
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test_case['description']}")
            print(f"   Job: {test_case['job'][:80]}...")
            print(f"   Expected: {test_case['expected'].upper()}")
            
            for method in methods:
                if method == 'base_model':
                    result = self.analyze_with_base_model(test_case['job'])
                elif method == 'ollama':
                    result = self.analyze_with_ollama(test_case['job'])
                elif method == 'hf_only':
                    result = self.hybrid_detector.analyze_with_hf_only(test_case['job'])
                    result['method'] = 'hf_only'
                elif method == 'rag_only':
                    result = self.hybrid_detector.analyze_with_rag_only(test_case['job'])
                    result['method'] = 'rag_only'
                else:  # hybrid
                    result = self.hybrid_detector.analyze_hybrid(test_case['job'])
                    result['method'] = 'hybrid'
                
                correct = result['prediction'] == test_case['expected']
                icon = "✅" if correct else "❌"
                
                print(f"   {method_names[method]:<18} {icon} {result['prediction'].upper():<6} "
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
    
    def create_comparison_plots(self):
        """Create comprehensive comparison plots"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.results)
        
        # Plot 1: Accuracy by Method
        accuracy_data = df.groupby('method_name')['correct'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(accuracy_data.index, accuracy_data.values, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700', '#9370DB'])
        ax1.set_title('🔍 Accuracy by Method', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        # Add value labels
        for bar, accuracy in zip(bars1, accuracy_data.values):
            ax1.text(bar.get_x() + bar.get_width()/2., accuracy + 0.02,
                    f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Average Confidence by Method
        confidence_data = df.groupby('method_name')['confidence'].mean().sort_values(ascending=False)
        bars2 = ax2.bar(confidence_data.index, confidence_data.values, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700', '#9370DB'])
        ax2.set_title('🎯 Average Confidence by Method', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Confidence Score')
        ax2.set_ylim(0, 1.1)
        for bar, confidence in zip(bars2, confidence_data.values):
            ax2.text(bar.get_x() + bar.get_width()/2., confidence + 0.02,
                    f'{confidence:.1%}', ha='center', va='bottom', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Processing Time by Method
        time_data = df.groupby('method_name')['processing_time'].mean().sort_values(ascending=True)
        bars3 = ax3.bar(time_data.index, time_data.values, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700', '#9370DB'])
        ax3.set_title('⏱️ Average Processing Time by Method', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        for bar, time_val in zip(bars3, time_data.values):
            ax3.text(bar.get_x() + bar.get_width()/2., time_val + 0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance Comparison Matrix
        methods = df['method_name'].unique()
        performance_matrix = []
        
        for method in methods:
            method_data = df[df['method_name'] == method]
            accuracy = method_data['correct'].mean()
            avg_confidence = method_data['confidence'].mean()
            avg_time = method_data['processing_time'].mean()
            
            performance_matrix.append({
                'Method': method,
                'Accuracy': accuracy,
                'Confidence': avg_confidence,
                'Speed (1/time)': 1/avg_time,  # Inverse for better visualization
                'Overall Score': accuracy * avg_confidence * (1/avg_time) * 100
            })
        
        perf_df = pd.DataFrame(performance_matrix)
        metrics = ['Accuracy', 'Confidence', 'Speed (1/time)', 'Overall Score']
        
        # Normalize for radar chart
        normalized_data = []
        for metric in metrics[:-1]:  # Exclude Overall Score
            min_val = perf_df[metric].min()
            max_val = perf_df[metric].max()
            normalized_col = (perf_df[metric] - min_val) / (max_val - min_val)
            normalized_data.append(normalized_col)
        
        # Create bar plot for performance matrix
        x = np.arange(len(methods))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            if metric == 'Overall Score':
                values = perf_df[metric].values
            else:
                values = normalized_data[i].values if i < len(normalized_data) else perf_df[metric].values
            
            ax4.bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        ax4.set_title('📊 Performance Matrix', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Normalized Score')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(methods, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('../results/comprehensive_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return perf_df
    
    def print_detailed_summary(self):
        """Print detailed summary statistics"""
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 90)
        print("📈 DETAILED PERFORMANCE SUMMARY")
        print("=" * 90)
        
        summary_data = []
        for method in df['method_name'].unique():
            method_data = df[df['method_name'] == method]
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
                'Performance Score': f"{(accuracy * avg_confidence * (1/avg_time) * 100):.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Find best method
        best_accuracy = summary_df.loc[summary_df['Accuracy'].str.rstrip('%').astype(float).idxmax()]
        best_confidence = summary_df.loc[summary_df['Avg Confidence'].str.rstrip('%').astype(float).idxmax()]
        fastest = summary_df.loc[summary_df['Avg Time (s)'].str.rstrip('s').astype(float).idxmin()]
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Highest Accuracy: {best_accuracy['Method']} ({best_accuracy['Accuracy']})")
        print(f"   Highest Confidence: {best_confidence['Method']} ({best_confidence['Avg Confidence']})")
        print(f"   Fastest: {fastest['Method']} ({fastest['Avg Time (s)']})")
        
        # Save results
        df.to_csv('../results/comprehensive_results.csv', index=False)
        summary_df.to_csv('../results/summary_statistics.csv', index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"   - ../results/comprehensive_results.csv")
        print(f"   - ../results/summary_statistics.csv")
        print(f"   - ../results/comprehensive_comparison.png")

def main():
    """Run comprehensive comparison"""
    import os
    os.makedirs('../results', exist_ok=True)
    
    comparator = ComprehensiveComparator()
    
    print("🚀 Starting Comprehensive Model Comparison...")
    print("This will test 5 different methods across 7 test cases.")
    print("Methods: Base Model, Ollama Fine-tuned, HF Fine-tuned, RAG Only, Hybrid")
    print("=" * 90)
    
    # Run comparison
    results = comparator.run_comprehensive_comparison()
    
    # Create plots
    print("\n📊 Generating comparison plots...")
    perf_df = comparator.create_comparison_plots()
    
    # Print summary
    comparator.print_detailed_summary()
    
    print("\n🎯 COMPARISON COMPLETE!")

if __name__ == "__main__":
    main()
