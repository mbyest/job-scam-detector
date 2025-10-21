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
    
    # Handle class imbalance
    fake_df = df[df['fraudulent'] == 1]
    real_df = df[df['fraudulent'] == 0].sample(n=len(fake_df) * 2, random_state=42)
    
    # Combine balanced dataset
    balanced_df = pd.concat([fake_df, real_df])
    print(f"Balanced dataset: {len(balanced_df)} total ({len(fake_df)} fake, {len(real_df)} real)")
    
    # Create text samples
    texts = []
    labels = []
    
    for _, row in balanced_df.iterrows():
        title = str(row['title']) if pd.notna(row['title']) else "No Title"
        company = str(row['company_profile']) if pd.notna(row['company_profile']) else "No Company Info"
        description = str(row['description']) if pd.notna(row['description']) else "No Description"
        
        text = f"JOB: {title}. COMPANY: {company[:200]}. DESCRIPTION: {description[:500]}."
        texts.append(text)
        labels.append(int(row['fraudulent']))
    
    # Create dataset
    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    
    # Convert to ClassLabel
    features = dataset.features.copy()
    features['label'] = ClassLabel(names=['real', 'fake'])
    dataset = dataset.cast(features)
    
    # Split dataset
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
    """Fine-tune the model"""
    
    # Load data
    dataset = load_and_prepare_data()
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1}
    )
    model.to(device)
    
    # SIMPLE Training arguments that work with all versions
    training_args = TrainingArguments(
        output_dir="../models/hf-job-scam-detector",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",  # Try original name
        save_strategy="epoch",
        logging_steps=10,
        seed=42,
        no_cuda=True,
    )
    
    # If evaluation_strategy fails, try this alternative:
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            compute_metrics=compute_metrics,
        )
    except TypeError:
        # Fallback: remove problematic arguments
        training_args = TrainingArguments(
            output_dir="../models/hf-job-scam-detector",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=10,
            seed=42,
            no_cuda=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            compute_metrics=compute_metrics,
        )
    
    # Start training
    print("Starting fine-tuning...")
    trainer.train()
    
    # Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print("Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained("../models/hf-job-scam-detector")
    print("Model saved to: ../models/hf-job-scam-detector/")

if __name__ == "__main__":
    fine_tune_model()
