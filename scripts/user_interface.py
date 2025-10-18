from flask import Flask, request, jsonify, render_template_string
import subprocess
import re

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Job Scam Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
        textarea { width: 100%; height: 200px; margin: 15px 0; padding: 10px; border: 1px solid #ddd; }
        button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 20px; border-radius: 8px; }
        .real { background: #d4edda; border: 2px solid #28a745; }
        .fake { background: #f8d7da; border: 2px solid #dc3545; }
        .loading { display: none; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Job Scam Detector</h1>
        <p><strong>Protect yourself from job scams!</strong> Paste any job posting below to check if it's legitimate.</p>
        
        <form onsubmit="analyzeJob(); return false;">
            <label><strong>Job Posting:</strong></label><br>
            <textarea id="jobText" placeholder="Paste the entire job description, company information, requirements, and contact details here..."></textarea>
            <br>
            <button type="submit">🔍 Analyze for Scams</button>
        </form>
        
        <div id="loading" class="loading">⏳ Analyzing job posting... (This may take 10-20 seconds)</div>
        
        <div id="result"></div>
    </div>

    <script>
        async function analyzeJob() {
            const jobText = document.getElementById('jobText').value;
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            if (!jobText.trim()) {
                alert('Please paste a job posting first!');
                return;
            }
            
            resultDiv.innerHTML = '';
            loadingDiv.style.display = 'block';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({job: jobText})
                });
                
                const data = await response.json();
                loadingDiv.style.display = 'none';
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="result" style="background: #fff3cd; border-color: #ffc107;">
                        <h3>⚠️ Error</h3>
                        <p>${data.error}</p>
                    </div>`;
                    return;
                }
                
                const isFake = data.prediction.toLowerCase().includes('fake');
                const cssClass = isFake ? 'fake' : 'real';
                const icon = isFake ? '❌' : '✅';
                
                resultDiv.innerHTML = `<div class="result ${cssClass}">
                    <h3>${icon} ${data.prediction}</h3>
                    <p><strong>Analysis:</strong> ${data.reasoning}</p>
                    ${isFake ? '<p><strong>⚠️ Warning:</strong> This job shows signs of being fraudulent. Be cautious about sharing personal information or sending money.</p>' : ''}
                </div>`;
                
            } catch (error) {
                loadingDiv.style.display = 'none';
                resultDiv.innerHTML = `<div class="result" style="background: #f8d7da; border-color: #dc3545;">
                    <h3>❌ Error</h3>
                    <p>Failed to analyze job. Please try again.</p>
                </div>`;
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/analyze', methods=['POST'])
def analyze_job():
    job_text = request.json['job']
    
    if len(job_text) < 20:
        return jsonify({'error': 'Job description too short. Please provide more details.'})
    
    try:
        # Use your balanced model
        result = subprocess.run([
            'ollama', 'run', 'balanced-scam-detector',
            f'Analyze this job posting and determine if it is Real or Fake: {job_text[:1500]}'
        ], capture_output=True, text=True, timeout=30)
        
        output = result.stdout.strip()
        
        # Parse the output
        if "fake" in output.lower():
            prediction = "Potential Scam Detected ❌"
        elif "real" in output.lower():
            prediction = "Likely Legitimate ✅"
        else:
            prediction = "Inconclusive ⚠️"
            
        return jsonify({
            'prediction': prediction,
            'reasoning': output,
            'status': 'success'
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Analysis timed out. Please try a shorter job description.'})
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

if __name__ == '__main__':
    print("Starting Job Scam Detector...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, port=5000)
