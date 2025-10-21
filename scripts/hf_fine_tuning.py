import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_and_prepare_data():
    """Load your Kaggle dataset and prepare for training"""
    print("Loading Kaggle dataset...")
    
    df = pd.read_csv('../data/fake_job_postings.csv')
    print(f"Loaded {len(df)} job postings")
    print(f"Real jobs: {len(df[df['fraudulent'] == 0])}")
    print(f"Fake jobs: {len(df[df['fraudulent'] == 1])}")
    
    # Handle class imbalance - we'll use all fake jobs and sample real jobs
    fake_df = df[df['fraudulent'] == 1]
    real_df = df[df['fraudulent'] == 0]
    
    # Sample real jobs to balance the dataset (use all fake jobs)
    real_sampled = real_df.sample(n=min(len(fake_df) * 2, len(real_df)), random_state=42)
    
    # Combine balanced dataset
    balanced_df = pd.concat([fake_df, real_sampled])
    print(f"Balanced dataset: {len(balanced_df)} total ({len(fake_df)} fake, {len(real_sampled)} real)")
    
    # Create proper text samples
    texts = []
    labels = []
    
    for _, row in balanced_df.iterrows():
        # Combine relevant fields into a single text
        title = str(row['title']) if pd.notna(row['title']) else "No Title"
        company = str(row['company_profile']) if pd.notna(row['company_profile']) else "No Company Info"
        description = str(row['description']) if pd.notna(row['description']) else "No Description"
        requirements = str(row['requirements']) if pd.notna(row['requirements']) else "No Requirements"
        
        # Create the input text
        text = f"JOB TITLE: {title}. COMPANY: {company[:200]}. DESCRIPTION: {description[:500]}. REQUIREMENTS: {requirements[:200]}."
        texts.append(text)
        
        # Label: 0 for real, 1 for fake
        labels.append(int(row['fraudulent']))
    
    # Create dataset with proper class labels
    dataset = Dataset.from_dict({
        'text': texts,
        'label': labels
    })
    
    # Convert to ClassLabel for proper stratification
    features = dataset.features.copy()
    features['label'] = ClassLabel(names=['real', 'fake'])
    dataset = dataset.cast(features)
    
    # Split into train/validation (80/20) with stratification
    dataset = dataset.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
    
    return dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def fine_tune_model():
    """Fine-tune the model using Hugging Face"""
    
    # Load and prepare data
    dataset = load_and_prepare_data()
    
    print("Dataset prepared:")
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    
    # Use a smaller, faster model for your Mac
    model_name = "distilbert-base-uncased"  # Faster and lighter than DeBERTa
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=256  # Shorter for faster training
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1}
    )
    model.to(device)
    
    # Training arguments optimized for CPU/Mac
    training_args = TrainingArguments(
        output_dir="../models/hf-job-scam-detector",
        overwrite_output_dir=True,
        num_train_epochs=3,  # Fewer epochs for faster training
        per_device_train_batch_size=4,  # Smaller batch size for CPU
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir='../logs',
        logging_steps=10,
        save_total_limit=2,
        seed=42,
        no_cuda=True,  # Force CPU usage
        dataloader_pin_memory=False  # Better for CPU
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Start training
    print("Starting fine-tuning...")
    print("This will take 10-30 minutes on CPU...")
    trainer.train()
    
    # Evaluate final model
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print("Final evaluation results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("../models/hf-job-scam-detector")
    
    print("Fine-tuning complete! Model saved to: ../models/hf-job-scam-detector/")
    
    return trainer

if __name__ == "__main__":
    fine_tune_model()