from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS, cross_origin
import pandas as pd
import io
import csv
from datetime import datetime
import os
import time
import subprocess

app = Flask(__name__)
# Enable CORS for all routes - this fixes the browser extension issue
CORS(app)

# Statistics tracking
analysis_stats = {
    'total_analysis': 0,
    'scams_detected': 0,
    'total_analysis_time': 0,
    'common_patterns': {}
}

PROFESSIONAL_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JobTrust AI - Job Scam Detection System</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-600: #4b5563;
            --gray-800: #1f2937;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: var(--gray-800);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .app-container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-bottom: 30px;
        }
        
        .tabs {
            display: flex;
            background: var(--gray-100);
            border-bottom: 1px solid var(--gray-200);
        }
        
        .tab {
            padding: 16px 24px;
            background: none;
            border: none;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            flex: 1;
            text-align: center;
        }
        
        .tab.active {
            background: white;
            border-bottom: 3px solid var(--primary);
            font-weight: 600;
        }
        
        .tab-content {
            display: none;
            padding: 30px;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--gray-800);
        }
        
        textarea, input[type="text"], input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px solid var(--gray-200);
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        textarea:focus, input:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        textarea {
            height: 200px;
            resize: vertical;
            font-family: inherit;
        }
        
        .btn {
            background: var(--primary);
            color: white;
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
        }
        
        .btn-secondary {
            background: var(--gray-600);
        }
        
        .btn-secondary:hover {
            background: var(--gray-800);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid var(--gray-200);
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result {
            margin-top: 30px;
            padding: 24px;
            border-radius: 12px;
            border-left: 6px solid;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-safe {
            background: #f0fdf4;
            border-color: var(--success);
        }
        
        .result-warning {
            background: #fef3c7;
            border-color: var(--warning);
        }
        
        .result-danger {
            background: #fef2f2;
            border-color: var(--danger);
        }
        
        .result h3 {
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .red-flags {
            margin-top: 16px;
        }
        
        .red-flag {
            background: white;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 6px;
            border-left: 3px solid var(--danger);
        }
        
        .confidence-meter {
            margin: 16px 0;
        }
        
        .confidence-bar {
            height: 8px;
            background: var(--gray-200);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }
        
        .file-upload-area {
            border: 2px dashed var(--gray-200);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .file-upload-area:hover {
            border-color: var(--primary);
            background: var(--gray-100);
        }
        
        .file-upload-area.dragover {
            border-color: var(--primary);
            background: var(--gray-100);
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }
        
        .feature-card {
            background: white;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 16px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .tabs {
                flex-direction: column;
            }
            
            .tab-content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ JobTrust AI</h1>
            <p>Advanced Job Scam Detection System</p>
        </div>
        
        <div class="app-container">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('single-tab')">Single Job Analysis</button>
                <button class="tab" onclick="switchTab('batch-tab')">Batch Analysis</button>
                <button class="tab" onclick="switchTab('api-tab')">API Access</button>
                <button class="tab" onclick="switchTab('stats-tab')">Statistics</button>
            </div>
            
            <!-- Single Job Analysis Tab -->
            <div id="single-tab" class="tab-content active">
                <h2>Analyze Single Job Posting</h2>
                <p>Paste a job description below to check for potential scams</p>
                
                <div class="form-group">
                    <label for="jobText">Job Description:</label>
                    <textarea id="jobText" placeholder="Paste the complete job posting including company information, requirements, salary details, and contact information..."></textarea>
                </div>
                
                <button class="btn" onclick="analyzeSingleJob()">
                    🔍 Analyze Job Posting
                </button>
                
                <div id="single-loading" class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing job posting for scam indicators...</p>
                </div>
                
                <div id="single-result"></div>
            </div>
            
            <!-- Batch Analysis Tab -->
            <div id="batch-tab" class="tab-content">
                <h2>Batch Job Analysis</h2>
                <p>Upload a CSV file with multiple job postings for bulk analysis</p>
                
                <div class="file-upload-area" onclick="document.getElementById('batchFile').click()" 
                     ondrop="handleFileDrop(event)" ondragover="handleDragOver(event)">
                    <input type="file" id="batchFile" accept=".csv" style="display: none;" onchange="handleFileSelect(this.files)">
                    <div style="font-size: 3rem;">📁</div>
                    <h3>Drop CSV file here or click to upload</h3>
                    <p>CSV should have columns: job_title, company, description, requirements</p>
                </div>
                
                <div id="batch-file-info" style="margin-top: 15px;"></div>
                
                <button class="btn" onclick="analyzeBatchJobs()" id="batch-btn" disabled>
                    📊 Analyze Batch Jobs
                </button>
                
                <div id="batch-loading" class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing batch jobs... This may take a few minutes</p>
                </div>
                
                <div id="batch-result"></div>
            </div>
            
            <!-- API Access Tab -->
            <div id="api-tab" class="tab-content">
                <h2>API Access</h2>
                <p>Integrate JobTrust AI into your applications</p>
                
                <div class="form-group">
                    <label>API Endpoint:</label>
                    <input type="text" value="http://localhost:5001/api/analyze" readonly 
                           style="background: var(--gray-100);">
                </div>
                
                <div class="form-group">
                    <label>Example Request (cURL):</label>
                    <textarea readonly style="height: 120px; font-family: monospace; background: var(--gray-100);">
curl -X POST http://localhost:5001/api/analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "job_text": "Your job description here...",
    "model": "improved_rag"
  }'</textarea>
                </div>
                
                <div class="form-group">
                    <label>Example Response:</label>
                    <textarea readonly style="height: 200px; font-family: monospace; background: var(--gray-100);">
{
  "prediction": "fake",
  "confidence": 0.92,
  "red_flags": ["unrealistic_salary", "personal_contact"],
  "reasoning": "Matches known scam patterns...",
  "analysis_time": 0.45
}</textarea>
                </div>
            </div>
            
            <!-- Statistics Tab -->
            <div id="stats-tab" class="tab-content">
                <h2>System Statistics</h2>
                <p>Real-time performance and detection metrics</p>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number" id="total-analysis">0</div>
                        <div class="stat-label">Total Analysis</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="scams-detected">0</div>
                        <div class="stat-label">Scams Detected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="accuracy-rate">0%</div>
                        <div class="stat-label">Accuracy Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="avg-time">0s</div>
                        <div class="stat-label">Avg. Analysis Time</div>
                    </div>
                </div>
                
                <h3 style="margin-top: 30px;">Common Scam Patterns Detected</h3>
                <div id="common-patterns">
                    <!-- Will be populated by JavaScript -->
                </div>
            </div>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>AI-Powered Detection</h3>
                <p>Advanced machine learning models trained on thousands of job scams</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Real-Time Analysis</h3>
                <p>Get instant results with detailed reasoning and confidence scores</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <h3>Multiple Models</h3>
                <p>Combines RAG pattern matching with fine-tuned AI models</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Batch Processing</h3>
                <p>Analyze multiple job postings simultaneously</p>
            </div>
        </div>
    </div>

    <script>
        let currentBatchFile = null;
        
        function switchTab(tabId) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabId).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Load stats when stats tab is opened
            if (tabId === 'stats-tab') {
                loadStatistics();
            }
        }
        
        async function analyzeSingleJob() {
            const jobText = document.getElementById('jobText').value.trim();
            const loadingDiv = document.getElementById('single-loading');
            const resultDiv = document.getElementById('single-result');
            
            if (!jobText) {
                alert('Please enter a job description to analyze.');
                return;
            }
            
            if (jobText.length < 50) {
                alert('Please provide a more detailed job description (at least 50 characters).');
                return;
            }
            
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ job_text: jobText })
                });
                
                const data = await response.json();
                loadingDiv.style.display = 'none';
                displaySingleResult(data);
                
            } catch (error) {
                loadingDiv.style.display = 'none';
                resultDiv.innerHTML = `
                    <div class="result result-danger">
                        <h3>❌ Analysis Failed</h3>
                        <p>Unable to analyze job posting. Please try again.</p>
                    </div>
                `;
            }
        }
        
        function displaySingleResult(data) {
            const resultDiv = document.getElementById('single-result');
            const isScam = data.prediction === 'fake';
            const confidencePercent = Math.round(data.confidence * 100);
            
            let resultClass = 'result-warning';
            let icon = '⚠️';
            let title = 'Suspicious Job';
            
            if (isScam) {
                resultClass = 'result-danger';
                icon = '❌';
                title = 'Potential Scam Detected';
            } else if (data.confidence > 0.8) {
                resultClass = 'result-safe';
                icon = '✅';
                title = 'Likely Legitimate';
            }
            
            let redFlagsHTML = '';
            if (data.red_flags && data.red_flags.length > 0) {
                redFlagsHTML = `
                    <div class="red-flags">
                        <h4>🚩 Red Flags Detected:</h4>
                        ${data.red_flags.map(flag => `
                            <div class="red-flag">${flag.replace(/_/g, ' ').toUpperCase()}</div>
                        `).join('')}
                    </div>
                `;
            }
            
            resultDiv.innerHTML = `
                <div class="result ${resultClass}">
                    <h3>${icon} ${title}</h3>
                    <p><strong>Confidence:</strong> ${confidencePercent}%</p>
                    
                    <div class="confidence-meter">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%; background: ${
                                isScam ? '#ef4444' : data.confidence > 0.8 ? '#10b981' : '#f59e0b'
                            }"></div>
                        </div>
                    </div>
                    
                    <p><strong>Analysis:</strong> ${data.reasoning || 'No detailed reasoning available.'}</p>
                    
                    ${redFlagsHTML}
                    
                    <p><strong>Analysis Time:</strong> ${data.analysis_time || data.processing_time || 'N/A'} seconds</p>
                    
                    ${isScam ? `
                        <div style="margin-top: 16px; padding: 12px; background: #fef2f2; border-radius: 6px;">
                            <strong>⚠️ Safety Warning:</strong> This job shows strong signs of being fraudulent. 
                            Do not share personal information or send money. Report suspicious job postings to the platform.
                        </div>
                    ` : ''}
                </div>
            `;
        }
        
        function handleDragOver(event) {
            event.preventDefault();
            event.currentTarget.classList.add('dragover');
        }
        
        function handleFileDrop(event) {
            event.preventDefault();
            event.currentTarget.classList.remove('dragover');
            const files = event.dataTransfer.files;
            handleFileSelect(files);
        }
        
        function handleFileSelect(files) {
            if (files.length > 0) {
                const file = files[0];
                if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                    currentBatchFile = file;
                    document.getElementById('batch-file-info').innerHTML = `
                        <div style="background: #f0fdf4; padding: 12px; border-radius: 8px;">
                            ✅ File selected: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)
                        </div>
                    `;
                    document.getElementById('batch-btn').disabled = false;
                } else {
                    alert('Please select a CSV file.');
                }
            }
        }
        
        async function analyzeBatchJobs() {
            if (!currentBatchFile) {
                alert('Please select a CSV file first.');
                return;
            }
            
            const loadingDiv = document.getElementById('batch-loading');
            const resultDiv = document.getElementById('batch-result');
            
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            
            const formData = new FormData();
            formData.append('file', currentBatchFile);
            
            try {
                const response = await fetch('/analyze-batch', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = 'job_analysis_results.csv';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    
                    resultDiv.innerHTML = `
                        <div class="result result-safe">
                            <h3>✅ Batch Analysis Complete</h3>
                            <p>Results have been downloaded. Check your downloads folder for "job_analysis_results.csv"</p>
                        </div>
                    `;
                } else {
                    throw new Error('Batch analysis failed');
                }
                
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="result result-danger">
                        <h3>❌ Batch Analysis Failed</h3>
                        <p>Unable to process batch file. Please try again.</p>
                    </div>
                `;
            } finally {
                loadingDiv.style.display = 'none';
            }
        }
        
        async function loadStatistics() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                document.getElementById('total-analysis').textContent = data.total_analysis.toLocaleString();
                document.getElementById('scams-detected').textContent = data.scams_detected.toLocaleString();
                document.getElementById('accuracy-rate').textContent = data.accuracy_rate + '%';
                document.getElementById('avg-time').textContent = data.avg_analysis_time + 's';
                
                // Display common patterns
                const patternsDiv = document.getElementById('common-patterns');
                patternsDiv.innerHTML = data.common_patterns.map(pattern => `
                    <div style="background: white; padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid #ef4444;">
                        <strong>${pattern.pattern}</strong> - ${pattern.count} detections
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('Failed to load statistics:', error);
            }
        }
        
        // Load initial statistics
        loadStatistics();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(PROFESSIONAL_HTML)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
@cross_origin()
def analyze_job():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    job_text = request.json.get('job_text', '')
    
    if len(job_text) < 20:
        return jsonify({'error': 'Job description too short. Please provide more details.'})
    
    # Update statistics
    analysis_stats['total_analysis'] += 1
    
    try:
        start_time = time.time()
        
                # Use the Accurate RAG system for better precision
        result = subprocess.run([
            'python3', '-c', 
            f'''
from fix_rag_accuracy import AccurateJobScamRAG
import sys
rag = AccurateJobScamRAG()
result = rag.analyze_job_enhanced(sys.argv[1])
print(f"PREDICTION:{{result["prediction"]}}")
print(f"CONFIDENCE:{{result["confidence"]}}")
print(f"REASONING:{{result["reasoning"]}}")
print(f"RED_FLAGS:{{",".join(result["red_flags"])}}")
            ''',
            job_text[:2000]  # Limit text length
        ], capture_output=True, text=True, timeout=30)
        
        processing_time = time.time() - start_time
        analysis_stats['total_analysis_time'] += processing_time
        
        # Parse the output
        output = result.stdout.strip()
        prediction = 'real'
        confidence = 0.5
        reasoning = "Analysis completed"
        red_flags = []
        
        for line in output.split('\n'):
            if line.startswith('PREDICTION:'):
                prediction = line.replace('PREDICTION:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                confidence = float(line.replace('CONFIDENCE:', '').strip())
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
            elif line.startswith('RED_FLAGS:'):
                flags = line.replace('RED_FLAGS:', '').strip()
                red_flags = flags.split(',') if flags else []
        
        # Update scam detection count
        if prediction == 'fake':
            analysis_stats['scams_detected'] += 1
            
        # Track common patterns in reasoning
        if 'unrealistic_salary' in red_flags:
            analysis_stats['common_patterns']['unrealistic_salary'] = analysis_stats['common_patterns'].get('unrealistic_salary', 0) + 1
        if 'personal_contact' in red_flags:
            analysis_stats['common_patterns']['personal_contact'] = analysis_stats['common_patterns'].get('personal_contact', 0) + 1
        if 'sensitive_info_request' in red_flags:
            analysis_stats['common_patterns']['sensitive_info_request'] = analysis_stats['common_patterns'].get('sensitive_info_request', 0) + 1
            
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': reasoning,
            'red_flags': red_flags,
            'analysis_time': round(processing_time, 2)
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Analysis timed out. Please try a shorter job description.'})
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

@app.route('/analyze-batch', methods=['POST'])
@cross_origin()
def analyze_batch_jobs():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Create results dataframe
        results = []
        for index, row in df.iterrows():
            job_text = f"TITLE: {row.get('job_title', '')} | COMPANY: {row.get('company', '')} | DESCRIPTION: {row.get('description', '')} | REQUIREMENTS: {row.get('requirements', '')}"
            
            # Simple analysis for batch processing
            is_scam = any(indicator in job_text.lower() for indicator in 
                         ['ssn', 'social security', '$5000', '$8000', '$10000', '@gmail.com', '@yahoo.com', 'no experience', 'immediately'])
            
            results.append({
                'job_title': row.get('job_title', ''),
                'company': row.get('company', ''),
                'prediction': 'fake' if is_scam else 'real',
                'confidence': 0.8 if is_scam else 0.6,
                'risk_level': 'High' if is_scam else 'Low'
            })
        
        # Create output CSV
        output = io.StringIO()
        results_df = pd.DataFrame(results)
        results_df.to_csv(output, index=False)
        
        # Save submission record
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f'../results/user_submissions/batch_analysis_{timestamp}.csv', index=False)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'job_analysis_results_{timestamp}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

@app.route('/stats')
@cross_origin()
def get_stats():
    avg_time = analysis_stats['total_analysis_time'] / max(analysis_stats['total_analysis'], 1)
    accuracy_rate = (analysis_stats['scams_detected'] / max(analysis_stats['total_analysis'], 1)) * 100
    
    # Get top common patterns
    common_patterns = sorted(
        [{'pattern': k, 'count': v} for k, v in analysis_stats['common_patterns'].items()],
        key=lambda x: x['count'],
        reverse=True
    )[:5]
    
    return jsonify({
        'total_analysis': analysis_stats['total_analysis'],
        'scams_detected': analysis_stats['scams_detected'],
        'accuracy_rate': round(accuracy_rate, 1),
        'avg_analysis_time': round(avg_time, 2),
        'common_patterns': common_patterns
    })

@app.route('/api/analyze', methods=['POST'])
@cross_origin()
def api_analyze():
    """API endpoint for programmatic access"""
    data = request.json
    job_text = data.get('job_text', '')
    model = data.get('model', 'improved_rag')
    
    if not job_text:
        return jsonify({'error': 'job_text is required'}), 400
    
    # Call the analysis function
    response = analyze_job()
    return response

if __name__ == '__main__':
    print("🚀 Starting JobTrust AI - Professional Job Scam Detector")
    print("📍 Access the application at: http://localhost:5001")
    print("📊 Features:")
    print("   • Single job analysis with detailed reports")
    print("   • Batch CSV processing")
    print("   • REST API for developers")
    print("   • Real-time statistics")
    print("   • Professional user interface")
    print("\nPress Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5001)