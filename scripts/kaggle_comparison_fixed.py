import pandas as pd
import numpy as np
import random
import subprocess
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from improve_rag import ImprovedJobScamRAG
from rag_system_fixed import JobScamRAG


class KaggleDatasetTester:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # ✅ Load Kaggle dataset safely
        self.kaggle_df = pd.read_csv(os.path.join(base_dir, '../data/fake_job_postings.csv'))
        self.kaggle_df.dropna(subset=['description'], inplace=True)
        self.kaggle_df = self.kaggle_df[self.kaggle_df['description'].str.strip() != '']

        # ✅ Initialize RAG systems
        self.baseline_rag = JobScamRAG()
        self.improved_rag = ImprovedJobScamRAG()

        # ✅ Load base HF classifier (real model, not base encoder)
        self.base_hf = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        # ✅ Load fine-tuned HF scam detector
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "../models/hf-job-scam-detector"))
        self.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(base_dir, "../models/hf-job-scam-detector"))
        self.finetuned_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

        self.results = []

    # -------------------------------------------------------
    # 🧠 Test a single job posting across all models
    # -------------------------------------------------------
    def test_single_job(self, job_text, true_label):
        try:
            # ✅ 1. Ollama base model
            result_base = subprocess.run(
                ['ollama', 'run', 'llama3.2:3b'],
                input=f"Analyze this job posting and answer ONLY with 'Real' or 'Fake': {job_text[:400]}",
                capture_output=True, text=True, timeout=30
            )
            base_pred = 'fake' if 'fake' in result_base.stdout.lower() else 'real'

            # ✅ 2. Ollama fine-tuned model
            result_finetuned = subprocess.run(
                ['ollama', 'run', 'fine-tuned-job-scam-model'],
                input=f"Analyze this job posting and answer ONLY with 'Real' or 'Fake': {job_text[:400]}",
                capture_output=True, text=True, timeout=30
            )
            finetuned_pred = 'fake' if 'fake' in result_finetuned.stdout.lower() else 'real'

            # ✅ 3. Baseline RAG
            baseline_result = self.baseline_rag.retrieve_similar_patterns(job_text, k=3)
            baseline_pred = 'fake' if 'fake' in str(baseline_result).lower() else 'real'

            # ✅ 4. Improved RAG
            improved_result = self.improved_rag.analyze_job_enhanced(job_text)
            improved_pred = 'fake' if 'fake' in str(improved_result).lower() else 'real'

            # ✅ 5. Base HF
            hf_pred = self.base_hf(job_text[:400])[0]['label']
            hf_pred = 'fake' if hf_pred.lower() in ['negative', 'fake'] else 'real'

            # ✅ 6. Fine-tuned HF
            fine_tuned_res = self.finetuned_pipeline(job_text[:400])[0]
            label = fine_tuned_res['label'].lower()
            if '1' in label or 'fake' in label:
                fine_tuned_pred = 'fake'
            else:
                fine_tuned_pred = 'real'

            return {
                'ollama_base': base_pred,
                'ollama_finetuned': finetuned_pred,
                'baseline_rag': baseline_pred,
                'improved_rag': improved_pred,
                'hf_base': hf_pred,
                'hf_finetuned': fine_tuned_pred,
                'true_label': 'fake' if true_label == 1 else 'real'
            }

        except Exception as e:
            print(f"❌ Error testing job: {e}")
            return None

    # -------------------------------------------------------
    # 📊 Run model comparisons on sample subset
    # -------------------------------------------------------
    def run_kaggle_comparison(self, num_samples=50):
        sampled_df = self.kaggle_df.sample(n=min(num_samples, len(self.kaggle_df)), random_state=42)

        for _, row in sampled_df.iterrows():
            job_text = str(row['description'])
            true_label = row['fraudulent']
            result = self.test_single_job(job_text, true_label)
            if result:
                self.results.append(result)

        df = pd.DataFrame(self.results)
        df.to_csv('kaggle_model_comparison_results.csv', index=False)
        print("✅ Comparison complete! Results saved to 'kaggle_model_comparison_results.csv'")
        return df

    # -------------------------------------------------------
    # 📈 Visualization & summary
    # -------------------------------------------------------
    def create_kaggle_comparison_visualizations(self):
        if not self.results:
            print("❌ No results found. Run run_kaggle_comparison() first.")
            return None

        df = pd.DataFrame(self.results)
        methods = ['ollama_base', 'ollama_finetuned', 'baseline_rag', 'improved_rag', 'hf_base', 'hf_finetuned']

        summary = []
        for m in methods:
            y_true = df['true_label']
            y_pred = df[m]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, pos_label='fake', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label='fake', zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label='fake', zero_division=0)
            summary.append({'method': m, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

        metrics_df = pd.DataFrame(summary)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df.melt(id_vars='method', var_name='metric', value_name='score'),
                    x='method', y='score', hue='metric')
        plt.xticks(rotation=45)
        plt.title("Model Comparison on Kaggle Job Scam Dataset")
        plt.tight_layout()
        plt.savefig('kaggle_model_comparison_metrics.png')
        plt.show()

        return metrics_df

    # -------------------------------------------------------
    # 📋 Detailed per-type summary
    # -------------------------------------------------------
    def print_detailed_kaggle_summary(self, df):
        print("\n=== Detailed Kaggle Comparison Summary ===")
        for _, row in df.iterrows():
            print(f"{row['method']:<20} | Acc: {row['accuracy']:.2f} | Prec: {row['precision']:.2f} | Rec: {row['recall']:.2f} | F1: {row['f1']:.2f}")


# -------------------------------------------------------
# 🚀 Run it all
# -------------------------------------------------------
def main():
    tester = KaggleDatasetTester()
    results = tester.run_kaggle_comparison(num_samples=50)
    metrics_df = tester.create_kaggle_comparison_visualizations()
    if metrics_df is not None:
        tester.print_detailed_kaggle_summary(metrics_df)


if __name__ == "__main__":
    main()

