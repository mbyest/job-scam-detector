import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def create_balanced_training():
    """Create properly balanced training data"""
    
    df = pd.read_csv('../data/fake_job_postings.csv')
    print(f"Original dataset: {len(df)} jobs")
    print(f"Real jobs: {len(df[df['fraudulent'] == 0])}")
    print(f"Fake jobs: {len(df[df['fraudulent'] == 1])}")
    
    # Get all fake jobs
    fake_jobs = df[df['fraudulent'] == 1]
    
    # Sample real jobs more carefully - focus on professional ones
    real_jobs = df[df['fraudulent'] == 0]
    
    # Filter for high-quality real jobs (those with company profiles, specific requirements)
    quality_real_jobs = real_jobs[
        (real_jobs['company_profile'].str.len() > 50) &
        (real_jobs['description'].str.len() > 100) &
        (real_jobs['requirements'].str.len() > 50)
    ]
    
    print(f"Quality real jobs: {len(quality_real_jobs)}")
    
    # Use balanced sampling
    training_real = quality_real_jobs.sample(n=min(len(fake_jobs) * 2, len(quality_real_jobs)), random_state=42)
    training_fake = fake_jobs
    
    # Combine
    balanced_df = pd.concat([training_real, training_fake])
    print(f"Balanced training set: {len(balanced_df)} jobs ({len(training_real)} real, {len(training_fake)} fake)")
    
    # Prepare training data
    texts = []
    labels = []
    
    for _, row in balanced_df.iterrows():
        title = str(row['title']) if pd.notna(row['title']) else "No Title"
        company = str(row.get('company_profile', '')) if pd.notna(row.get('company_profile')) else "No Company"
        description = str(row['description']) if pd.notna(row['description']) else "No Description"
        
        text = f"JOB: {title}. COMPANY: {company[:150]}. DESCRIPTION: {description[:400]}."
        texts.append(text)
        labels.append(int(row['fraudulent']))
    
    # Create dataset
    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    features = dataset.features.copy()
    features['label'] = ClassLabel(names=['real', 'fake'])
    dataset = dataset.cast(features)
    
    # Split
    dataset = dataset.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
    
    return dataset

def retrain_model():
    """Retrain the model with better balanced data"""
    
    dataset = create_balanced_training()
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1}
    )
    
    # Training arguments with early stopping
    training_args = TrainingArguments(
        output_dir="../models/hf-job-scam-detector-v2",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics
    )
    
    print("Retraining model with better balanced data...")
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print("New model performance:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Save new model
    trainer.save_model()
    tokenizer.save_pretrained("../models/hf-job-scam-detector-v2")
    print("✅ New model saved to: ../models/hf-job-scam-detector-v2/")

if __name__ == "__main__":
    retrain_model()
