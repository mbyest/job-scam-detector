import csv
import random

def create_balanced_training():
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
    
    print(f"Total real jobs: {len(real_jobs)}")
    print(f"Total fake jobs: {len(fake_jobs)}")
    
    # Use ALL fake jobs and sample an equal number of real jobs
    training_real = random.sample(real_jobs, len(fake_jobs))
    training_fake = fake_jobs  # Use all fake jobs
    
    print(f"Using {len(training_real)} real jobs and {len(training_fake)} fake jobs")
    
    modelfile_content = """FROM llama3.2:3b

SYSTEM \"\"\"You are a cybersecurity expert that detects fraudulent job postings. Analyze each job posting and classify it as 'Real' or 'Fake'. Provide brief reasoning.\"\"\"

PARAMETER temperature 0.1

"""
    
    # Add real job examples
    for job in training_real[:50]:  # Limit to 50 of each for now
        desc = job["description"][:150].replace('"', "'") if job["description"] else "No description"
        company = job.get("company_profile", "No company profile")[:80].replace('"', "'") if job.get("company_profile") else "No company profile"
        
        modelfile_content += f'message user "Job: {job["title"]}. Company: {company}. Description: {desc}"\n'
        modelfile_content += 'message assistant "Prediction: Real. Reasoning: Legitimate job posting with proper details."\n\n'
    
    # Add fake job examples  
    for job in training_fake[:50]:  # Limit to 50 of each for now
        desc = job["description"][:150].replace('"', "'") if job["description"] else "No description"
        company = job.get("company_profile", "No company profile")[:80].replace('"', "'") if job.get("company_profile") else "No company profile"
        
        modelfile_content += f'message user "Job: {job["title"]}. Company: {company}. Description: {desc}"\n'
        modelfile_content += 'message assistant "Prediction: Fake. Reasoning: Shows signs of fraudulent posting."\n\n'
    
    # Save to file
    with open('../models/balanced_scam_detector.Modelfile', 'w') as f:
        f.write(modelfile_content)
    
    print("Created balanced_scam_detector.Modelfile!")
    print(f"Used: 50 real jobs + 50 fake jobs (balanced)")
    print(f"Total fake jobs available: {len(fake_jobs)}")

if __name__ == "__main__":
    create_balanced_training()
