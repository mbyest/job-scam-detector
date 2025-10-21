import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import time
import numpy as np
import os
from rag_system_fixed import JobScamRAG

# Create results directory
os.makedirs('../results', exist_ok=True)

class EnhancedComparator:
    def __init__(self):
        print("Loading all models and systems...")
        
        # Load fine-tuned model
        self.hf_tokenizer = AutoTokenizer.from_pretrained("../models/hf-job-scam-detector")
        self.hf_model = AutoModelForSequenceClassification.from_pretrained("../models/hf-job-scam-detector")
        self.hf_model.eval()
        
        # Load base model
        self.base_classifier = pipeline("text-classification", model="distilbert-base-uncased")
        
        # Load RAG system
        self.rag_system = JobScamRAG()
        
        print("All systems loaded successfully!")
    
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
            'method': 'Fine-tuned HF'
        }
    
    def analyze_with_base(self, job_text):
        """Analyze using base (non-fine-tuned) model"""
        start_time = time.time()
        text = f"Job posting analysis: {job_text}"
        
        result = self.base_classifier(text[:512])
        
        # More sophisticated mapping for base model
        label = result[0]['label'].lower()
        score = result[0]['score']
        
        # Base model doesn't know job scams, so we need to interpret carefully
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
            'method': 'Base Model'
        }
    
    def analyze_with_rag(self, job_text):
        """Analyze using RAG system only"""
        start_time = time.time()
        
        # Get similar patterns
        similar_patterns = self.rag_system.retrieve_similar_patterns(job_text, k=3)
        
        # Calculate weighted decision based on similarity
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
    
    def analyze_hybrid(self, job_text):
        """Analyze using hybrid approach"""
        start_time = time.time()
        
        # Get predictions from both systems
        hf_result = self.analyze_with_fine_tuned(job_text)
        rag_result = self.analyze_with_rag(job_text)
        
        # Combine predictions (weighted average)
        hf_weight = 0.7  # Trust fine-tuned model more
        rag_weight = 0.3
        
        hf_fake_prob = hf_result['confidence'] if hf_result['prediction'] == 'fake' else 1 - hf_result['confidence']
        rag_fake_prob = rag_result['confidence'] if rag_result['prediction'] == 'fake' else 1 - rag_result['confidence']
        
        combined_fake_prob = (hf_weight * hf_fake_prob + rag_weight * rag_fake_prob)
        
        final_prediction = 'fake' if combined_fake_prob > 0.5 else 'real'
        final_confidence = max(combined_fake_prob, 1 - combined_fake_prob)
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'processing_time': time.time() - start_time,
            'method': 'Hybrid (HF+RAG)',
            'hf_prediction': hf_result['prediction'],
            'rag_prediction': rag_result['prediction'],
            'combined_probability': combined_fake_prob
        }
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison with diverse test cases"""
        
        # More diverse and realistic test cases
        test_cases = [
            # OBVIOUS SCAMS
            {
                'job': "Work from home data entry. No experience needed. Earn $8,000 monthly. Send your Social Security Number and bank details to quickhire2024@gmail.com to start immediately.",
                'expected': 'fake',
                'type': 'obvious_scam',
                'description': 'Classic scam: sensitive info + unrealistic salary'
            },
            {
                'job': "Make $10,000 weekly processing emails from home. No skills required. Contact easycash@yahoo.com with personal information.",
                'expected': 'fake', 
                'type': 'obvious_scam',
                'description': 'Pyramid scheme: too good to be true'
            },
            {
                'job': "Urgent hiring! Customer service reps needed immediately. Earn $5,000 monthly. No interview required. Send $200 for training materials.",
                'expected': 'fake',
                'type': 'obvious_scam', 
                'description': 'Scam: urgency + upfront payment'
            },
            
            # LEGITIMATE JOBS
            {
                'job': "Software Engineer at Google. Requirements: Bachelor's in Computer Science, 3+ years experience with Java and Python. Competitive salary with full benefits. Apply through careers.google.com.",
                'expected': 'real',
                'type': 'legitimate',
                'description': 'Professional tech job at known company'
            },
            {
                'job': "Marketing Coordinator at IBM. Requirements: Marketing degree, 2+ years experience, proficiency with social media platforms. Full benefits package. Apply via IBM careers portal.",
                'expected': 'real',
                'type': 'legitimate',
                'description': 'Legitimate corporate marketing role'
            },
            {
                'job': "Data Analyst at Microsoft. Analyze business data to drive insights. Requirements: SQL, Python, statistics background. Competitive compensation. Professional references required.",
                'expected': 'real',
                'type': 'legitimate',
                'description': 'Professional data role'
            },
            
            # EDGE CASES - These are tricky!
            {
                'job': "Sales Representative. Commission-based role. Work from home flexibility. Must have reliable internet and phone. High earning potential for motivated individuals.",
                'expected': 'real', 
                'type': 'edge_case',
                'description': 'Legitimate but sounds salesy'
            },
            {
                'job': "Immediate opening for customer support. Great company culture. Work remotely. Competitive pay based on experience. Background check required.",
                'expected': 'real',
                'type': 'edge_case',
                'description': 'Legitimate but uses some urgent language'
            },
            {
                'job': "Financial advisor needed. Help clients with investments. Flexible hours. Must be self-motivated. Send resume to careers@legitfinance.com",
                'expected': 'real',
                'type': 'edge_case', 
                'description': 'Legitimate finance role'
            },
            
            # MORE OBVIOUS SCAMS
            {
                'job': "Get rich quick! Earn $15,000 monthly from home. No experience needed. Just forward our emails. Contact millionaire2024@gmail.com",
                'expected': 'fake',
                'type': 'obvious_scam',
                'description': 'Obvious get-rich-quick scam'
            },
            {
                'job': "Medical billing specialist. Work from home. $12,000 monthly. No certification required. Send SSN for background check to medicalhr@yahoo.com",
                'expected': 'fake',
                'type': 'obvious_scam',
                'description': 'Scam: unrealistic medical salary'
            }
        ]
        
        print("�� COMPREHENSIVE COMPARISON: 4 METHODS")
        print("=" * 80)
        print(f"Testing {len(test_cases)} diverse job postings...")
        print("Methods: Base Model, Fine-tuned HF, RAG Only, Hybrid (HF+RAG)")
        print("=" * 80)
        
        results = []
        methods = ['base', 'fine_tuned', 'rag', 'hybrid']
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n📋 Test {i:2d} [{test['type']:>12}]: {test['description']}")
            print(f"   Expected: {test['expected'].upper()}")
            
            for method in methods:
                if method == 'base':
                    result = self.analyze_with_base(test['job'])
                elif method == 'fine_tuned':
                    result = self.analyze_with_fine_tuned(test['job'])
                elif method == 'rag':
                    result = self.analyze_with_rag(test['job'])
                else:  # hybrid
                    result = self.analyze_hybrid(test['job'])
                
                correct = result['prediction'] == test['expected']
                icon = "✅" if correct else "❌"
                
                print(f"   {result['method']:<18} {icon} {result['prediction'].upper():<6} "
                      f"(conf: {result['confidence']:.1%}, time: {result['processing_time']:.2f}s)")
                
                results.append({
                    'test_case': i,
                    'method': result['method'],
                    'prediction': result['prediction'],
                    'expected': test['expected'],
                    'confidence': result['confidence'],
                    'processing_time': result['processing_time'],
                    'correct': correct,
                    'type': test['type'],
                    'description': test['description']
                })
        
        return results
    
    def create_detailed_visualizations(self, results):
        """Create comprehensive visualizations"""
        df = pd.DataFrame(results)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Define subplot grid
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # Plot 1: Overall Accuracy (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        accuracy_data = df.groupby('method')['correct'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(accuracy_data.index, accuracy_data.values)
        ax1.set_title('Overall Accuracy by Method', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        for bar, acc in zip(bars1, accuracy_data.values):
            ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.02, f'{acc:.1%}', 
                    ha='center', va='bottom', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Accuracy by Job Type (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        type_accuracy = df.pivot_table(index='type', columns='method', values='correct', aggfunc='mean')
        type_accuracy.plot(kind='bar', ax=ax2)
        ax2.set_title('Accuracy by Job Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.legend(title='Method')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Confidence Levels (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        confidence_data = df.groupby('method')['confidence'].mean().sort_values(ascending=False)
        bars3 = ax3.bar(confidence_data.index, confidence_data.values)
        ax3.set_title('Average Confidence by Method', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Confidence Score')
        ax3.set_ylim(0, 1.1)
        for bar, conf in zip(bars3, confidence_data.values):
            ax3.text(bar.get_x() + bar.get_width()/2., conf + 0.02, f'{conf:.1%}', 
                    ha='center', va='bottom', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Processing Time (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        time_data = df.groupby('method')['processing_time'].mean().sort_values(ascending=True)
        bars4 = ax4.bar(time_data.index, time_data.values)
        ax4.set_title('Average Processing Time', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Time (seconds)')
        for bar, t in zip(bars4, time_data.values):
            ax4.text(bar.get_x() + bar.get_width()/2., t + 0.01, f'{t:.2f}s', 
                    ha='center', va='bottom', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Performance Matrix (middle right)
        ax5 = fig.add_subplot(gs[1, 1:])
        performance_metrics = []
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            accuracy = method_data['correct'].mean()
            confidence = method_data['confidence'].mean()
            speed = 1 / method_data['processing_time'].mean()  # Inverse for better scaling
            
            performance_metrics.append({
                'Method': method,
                'Accuracy': accuracy,
                'Confidence': confidence,
                'Speed': speed
            })
        
        perf_df = pd.DataFrame(performance_metrics)
        perf_df.set_index('Method', inplace=True)
        
        # Normalize for radar chart-like visualization
        normalized_df = perf_df / perf_df.max()
        
        x = np.arange(len(normalized_df))
        width = 0.2
        
        for i, metric in enumerate(normalized_df.columns):
            ax5.bar(x + i*width, normalized_df[metric], width, label=metric, alpha=0.8)
        
        ax5.set_title('Performance Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Normalized Score')
        ax5.set_xticks(x + width)
        ax5.set_xticklabels(normalized_df.index, rotation=45)
        ax5.legend()
        
        # Plot 6: Detailed breakdown (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        
        # Create detailed performance table
        summary_data = []
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            for job_type in df['type'].unique():
                type_data = method_data[method_data['type'] == job_type]
                if len(type_data) > 0:
                    accuracy = type_data['correct'].mean()
                    summary_data.append({
                        'Method': method,
                        'Job Type': job_type,
                        'Accuracy': accuracy,
                        'Count': len(type_data)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        pivot_table = summary_df.pivot(index='Method', columns='Job Type', values='Accuracy')
        
        sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Accuracy'})
        ax6.set_title('Detailed Accuracy Breakdown by Job Type', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../results/enhanced_comparison_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return df, perf_df
    
    def print_comprehensive_summary(self, df, perf_df):
        """Print comprehensive performance summary"""
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 100)
        
        # Overall summary
        print("\n" + "�� OVERALL PERFORMANCE SUMMARY")
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
                'Method': method,
                'Accuracy': f"{accuracy:.1%}",
                'Avg Confidence': f"{avg_confidence:.1%}",
                'Avg Time (s)': f"{avg_time:.3f}",
                'Correct/Total': f"{total_correct}/{total_tests}",
                'Performance Score': f"{(accuracy * avg_confidence * 100):.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Performance by job type
        print("\n" + "🎯 PERFORMANCE BY JOB TYPE")
        print("-" * 50)
        
        type_performance = df.pivot_table(index='type', columns='method', values='correct', aggfunc='mean')
        print(type_performance.round(3).to_string())
        
        # Key insights
        print("\n" + "💡 KEY INSIGHTS")
        print("-" * 50)
        
        best_overall = summary_df.loc[summary_df['Accuracy'].str.rstrip('%').astype(float).idxmax()]
        most_confident = summary_df.loc[summary_df['Avg Confidence'].str.rstrip('%').astype(float).idxmax()]
        fastest = summary_df.loc[summary_df['Avg Time (s)'].astype(float).idxmin()]
        
        print(f"• Most Accurate: {best_overall['Method']} ({best_overall['Accuracy']})")
        print(f"• Most Confident: {most_confident['Method']} ({most_confident['Avg Confidence']})")
        print(f"• Fastest: {fastest['Method']} ({fastest['Avg Time (s)']})")
        
        # Calculate improvement over base model
        base_accuracy = df[df['method'] == 'Base Model']['correct'].mean()
        for method in ['Fine-tuned HF', 'RAG Only', 'Hybrid (HF+RAG)']:
            if method in df['method'].values:
                method_accuracy = df[df['method'] == method]['correct'].mean()
                improvement = (method_accuracy - base_accuracy) / base_accuracy * 100
                print(f"• {method}: {improvement:+.1f}% improvement over Base Model")
        
        # Save detailed results
        df.to_csv('../results/enhanced_comparison_results.csv', index=False)
        perf_df.to_csv('../results/performance_metrics.csv', index=True)
        
        print(f"\n💾 Detailed results saved to:")
        print(f"   - ../results/enhanced_comparison_results.csv")
        print(f"   - ../results/performance_metrics.csv")
        print(f"   - ../results/enhanced_comparison_analysis.png")

def main():
    comparator = EnhancedComparator()
    
    print("🚀 Starting Enhanced Comprehensive Comparison...")
    print("This will test 4 methods across diverse job types including edge cases.")
    print("Please be patient - this may take a few minutes...")
    
    results = comparator.run_comprehensive_comparison()
    df, perf_df = comparator.create_detailed_visualizations(results)
    comparator.print_comprehensive_summary(df, perf_df)
    
    print("\n🎉 ENHANCED COMPARISON COMPLETE!")
    print("You now have comprehensive data showing the strengths and weaknesses of each approach!")

if __name__ == "__main__":
    main()
