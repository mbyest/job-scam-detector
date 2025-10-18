import csv
import random

def create_training_data():
    real_jobs = []
    fake_jobs = []
    
    # Read all data
    with open('../data/fake_job_postings.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['fraudulent'] == '1':
                fake_jobs.append(row)
            else:
                real_jobs.append(row)
    
    # Sample balanced training data (20 of each)
    training_real = random.sample(real_jobs, 20)
    training_fake = random.sample(fake_jobs, 20)
    
    print("Creating Modelfile with 40 training examples...")
    
    modelfile_content = """FROM llama3.2:3b

SYSTEM \"\"\"You are a cybersecurity expert that detects fraudulent job postings. Analyze each job posting and classify it as 'Real' or 'Fake'. Provide brief reasoning.\"\"\"

PARAMETER temperature 0.1

"""
    
    # Add real job examples
    for job in training_real:
        modelfile_content += f'message user "Job: {job["title"]}. Company: {job.get("company_profile", "No profile")[:100]}. Description: {job["description"][:200]}..."\n'
        modelfile_content += 'message assistant "Prediction: Real. Reasoning: Legitimate job posting with proper company details and realistic requirements."\n\n'
    
    # Add fake job examples  
    for job in training_fake:
        modelfile_content += f'message user "Job: {job["title"]}. Company: {job.get("company_profile", "No profile")[:100]}. Description: {job["description"][:200]}..."\n'
        modelfile_content += 'message assistant "Prediction: Fake. Reasoning: Shows signs of fraudulent posting such as vague details or suspicious elements."\n\n'
    
    # Save to file
    with open('../models/large_scam_detector.Modelfile', 'w') as f:
        f.write(modelfile_content)
    
    print("Created large_scam_detector.Modelfile with 40 training examples!")
    print(f"Used: {len(training_real)} real jobs + {len(training_fake)} fake jobs")

if __name__ == "__main__":
    create_training_data()
