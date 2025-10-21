import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import time

# Load your fine-tuned model
print("Loading fine-tuned model...")
hf_tokenizer = AutoTokenizer.from_pretrained("../models/hf-job-scam-detector")
hf_model = AutoModelForSequenceClassification.from_pretrained("../models/hf-job-scam-detector")
hf_model.eval()

# Load base model
print("Loading base model...")
base_classifier = pipeline("text-classification", model="distilbert-base-uncased")

def analyze_with_fine_tuned(job_text):
    start_time = time.time()
    text = f"JOB: Unknown. COMPANY: Unknown. DESCRIPTION: {job_text}. REQUIREMENTS: Unknown."
    
    inputs = hf_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = hf_model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()
    
    return {
        'prediction': 'fake' if prediction == 1 else 'real',
        'confidence': confidence,
        'processing_time': time.time() - start_time,
        'method': 'fine_tuned'
    }

def analyze_with_base(job_text):
    start_time = time.time()
    text = f"Is this job legitimate? {job_text}"
    
    result = base_classifier(text[:512])
    
    # Simple mapping
    label = result[0]['label'].lower()
    if 'scam' in label or 'fake' in label:
        prediction = 'fake'
    else:
        prediction = 'real'
    
    return {
        'prediction': prediction,
        'confidence': result[0]['score'],
        'processing_time': time.time() - start_time,
        'method': 'base_model'
    }

# Test cases
test_cases = [
    {"job": "Work from home data entry. No experience. Earn $8,000 monthly. Send SSN to personal@gmail.com", "expected": "fake"},
    {"job": "Software Engineer at Google. Bachelor's degree required. Apply via careers.google.com", "expected": "real"},
]

print("🧪 SIMPLE COMPARISON: Base vs Fine-tuned")
print("=" * 60)

results = []
for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['job'][:50]}...")
    print(f"Expected: {test['expected'].upper()}")
    
    for analyzer, name in [(analyze_with_base, "Base Model"), (analyze_with_fine_tuned, "Fine-tuned")]:
        result = analyzer(test['job'])
        correct = "✅" if result['prediction'] == test['expected'] else "❌"
        print(f"  {name:<12} {correct} {result['prediction'].upper():<6} "
              f"(conf: {result['confidence']:.1%}, time: {result['processing_time']:.2f}s)")
        
        results.append({**result, 'test_case': i, 'expected': test['expected'], 
                       'correct': result['prediction'] == test['expected']})

# Simple plot
df = pd.DataFrame(results)
accuracy = df.groupby('method')['correct'].mean()

plt.figure(figsize=(8, 5))
bars = plt.bar(accuracy.index, accuracy.values, color=['red', 'green'])
plt.title('Base Model vs Fine-tuned Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
for bar, acc in zip(bars, accuracy.values):
    plt.text(bar.get_x() + bar.get_width()/2., acc + 0.02, f'{acc:.0%}', 
             ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('../results/simple_comparison.png', dpi=150)
plt.show()

print(f"\n📊 Results saved with plot!")
