import csv
import json

def examine_data():
    """Look at both real and fake jobs to understand patterns"""
    real_jobs = []
    fake_jobs = []
    
    with open('../data/fake_job_postings.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['fraudulent'] == '1' and len(fake_jobs) < 5:
                fake_jobs.append(row)
            elif row['fraudulent'] == '0' and len(real_jobs) < 5:
                real_jobs.append(row)
            if len(real_jobs) >= 5 and len(fake_jobs) >= 5:
                break
    
    print("=== REAL JOBS ===")
    for i, job in enumerate(real_jobs):
        print(f"\nReal Job {i+1}:")
        print(f"Title: {job['title']}")
        print(f"Company: {job.get('company_profile', 'N/A')[:100]}...")
        print(f"Description: {job['description'][:150]}...")
    
    print("\n=== FAKE JOBS ===")
    for i, job in enumerate(fake_jobs):
        print(f"\nFake Job {i+1}:")
        print(f"Title: {job['title']}")
        print(f"Company: {job.get('company_profile', 'N/A')[:100]}...")
        print(f"Description: {job['description'][:150]}...")

if __name__ == "__main__":
    examine_data()
