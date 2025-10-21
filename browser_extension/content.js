// This script runs automatically on job pages
console.log('JobTrust AI: Monitoring job page for scams...');

// Observe page changes for single-page applications
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'childList') {
            // Check if we're on a job details page
            if (isJobDetailsPage()) {
                injectSafetyBadge();
            }
        }
    });
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

function isJobDetailsPage() {
    // Check various job site patterns
    const patterns = [
        // LinkedIn
        document.querySelector('.jobs-details-top-card__job-title'),
        // Indeed
        document.querySelector('.jobsearch-JobInfoHeader-title'),
        // Glassdoor
        document.querySelector('.jobTitle'),
        // Monster
        document.querySelector('.job-title'),
        // ZipRecruiter
        document.querySelector('.job_description')
    ];
    
    return patterns.some(element => element !== null);
}

function injectSafetyBadge() {
    // Don't inject if already exists
    if (document.getElementById('jobtrust-badge')) {
        return;
    }

    const badge = document.createElement('div');
    badge.id = 'jobtrust-badge';
    badge.innerHTML = `
        <div style="
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            border: 2px solid #2563eb;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 10000;
            font-family: 'Segoe UI', system-ui, sans-serif;
            max-width: 300px;
        ">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 16px;">🛡️</span>
                <strong style="color: #2563eb;">JobTrust AI</strong>
            </div>
            <p style="margin: 0 0 8px 0; font-size: 12px; color: #6b7280;">
                This job is being monitored for safety
            </p>
            <button id="quickAnalyzeBtn" style="
                background: #2563eb;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
                cursor: pointer;
                width: 100%;
            ">
                🔍 Quick Analyze
            </button>
        </div>
    `;

    document.body.appendChild(badge);

    // Add click handler for quick analysis
    document.getElementById('quickAnalyzeBtn').addEventListener('click', function() {
        analyzeCurrentJob();
    });
}

async function analyzeCurrentJob() {
    const jobData = extractJobData();
    
    if (!jobData.text) {
        alert('JobTrust AI: Could not extract job information.');
        return;
    }

    // Show analyzing state
    const badge = document.getElementById('jobtrust-badge');
    badge.innerHTML = `
        <div style="text-align: center; padding: 8px;">
            <div style="
                border: 2px solid #f3f4f6;
                border-top: 2px solid #2563eb;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                margin: 0 auto 8px;
            "></div>
            <p style="margin: 0; font-size: 12px; color: #6b7280;">
                Analyzing job safety...
            </p>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    `;

    try {
        const response = await fetch('http://localhost:5001/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            'Accept': 'application/json'
            },
            body: JSON.stringify({
                job_text: jobData.text,
                source: 'browser_extension_auto'
            })
        });

        const result = await response.json();
        displayAnalysisResult(result, badge);

    } catch (error) {
        badge.innerHTML = `
            <div style="text-align: center; padding: 8px;">
                <p style="margin: 0; color: #ef4444; font-size: 12px;">
                    ❌ Analysis failed
                </p>
            </div>
        `;
    }
}

function displayAnalysisResult(result, badge) {
    const isScam = result.prediction === 'fake';
    const confidence = Math.round(result.confidence * 100);
    
    let badgeColor = '#10b981'; // Safe - green
    let badgeText = '✅ Likely Safe';
    let badgeMessage = 'This job appears legitimate';

    if (isScam) {
        badgeColor = '#ef4444'; // Scam - red
        badgeText = '❌ Potential Scam';
        badgeMessage = 'Exercise caution';
    } else if (result.confidence < 0.8) {
        badgeColor = '#f59e0b'; // Suspicious - yellow
        badgeText = '⚠️ Suspicious';
        badgeMessage = 'Review carefully';
    }

    badge.innerHTML = `
        <div style="text-align: center; padding: 8px;">
            <div style="color: ${badgeColor}; font-weight: bold; margin-bottom: 4px;">
                ${badgeText}
            </div>
            <div style="font-size: 11px; color: #6b7280; margin-bottom: 6px;">
                ${badgeMessage}
            </div>
            <div style="background: #e5e7eb; height: 4px; border-radius: 2px; margin-bottom: 6px;">
                <div style="
                    background: ${badgeColor};
                    height: 100%;
                    width: ${confidence}%;
                    border-radius: 2px;
                "></div>
            </div>
            <div style="font-size: 10px; color: #6b7280;">
                Confidence: ${confidence}%
            </div>
            <button onclick="showDetailedAnalysis()" style="
                background: ${badgeColor};
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 10px;
                cursor: pointer;
                margin-top: 6px;
                width: 100%;
            ">
                📊 Details
            </button>
        </div>
    `;

    // Store result for detailed view
    badge._analysisResult = result;
}

// Make function available globally for the button
window.showDetailedAnalysis = function() {
    const badge = document.getElementById('jobtrust-badge');
    const result = badge._analysisResult;
    
    if (!result) return;

    // Create detailed modal
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        border-radius: 12px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        padding: 20px;
        z-index: 10001;
        max-width: 400px;
        max-height: 80vh;
        overflow-y: auto;
        font-family: 'Segoe UI', system-ui, sans-serif;
    `;

    const isScam = result.prediction === 'fake';
    
    modal.innerHTML = `
        <div style="margin-bottom: 16px;">
            <h3 style="margin: 0 0 8px 0; color: ${isScam ? '#ef4444' : '#10b981'};">
                ${isScam ? '❌ Potential Scam Detected' : '✅ Likely Legitimate Job'}
            </h3>
            <div style="font-size: 14px; color: #6b7280;">
                Confidence: ${Math.round(result.confidence * 100)}%
            </div>
        </div>
        
        <div style="margin-bottom: 16px;">
            <strong>Analysis:</strong>
            <p style="font-size: 14px; margin: 8px 0;">${result.reasoning || 'No detailed analysis available.'}</p>
        </div>
        
        ${result.red_flags && result.red_flags.length > 0 ? `
            <div style="margin-bottom: 16px;">
                <strong>🚩 Red Flags:</strong>
                ${result.red_flags.map(flag => `
                    <div style="
                        background: #fef2f2;
                        padding: 6px 10px;
                        margin: 4px 0;
                        border-radius: 4px;
                        font-size: 12px;
                        border-left: 3px solid #ef4444;
                    ">${flag.replace(/_/g, ' ').toUpperCase()}</div>
                `).join('')}
            </div>
        ` : ''}
        
        <button onclick="this.parentElement.remove()" style="
            background: #6b7280;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        ">
            Close
        </button>
    `;

    // Add overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 10000;
    `;
    overlay.onclick = function() {
        modal.remove();
        overlay.remove();
    };

    document.body.appendChild(overlay);
    document.body.appendChild(modal);
};

// Re-use the extraction function from popup
function extractJobData() {
    // Same function as in popup.js
    const jobData = {
        title: '',
        company: '',
        description: '',
        text: ''
    };

    // Try to extract from LinkedIn
    if (window.location.hostname.includes('linkedin.com')) {
        const titleEl = document.querySelector('.jobs-details-top-card__job-title');
        const companyEl = document.querySelector('.jobs-details-top-card__company-url');
        const descEl = document.querySelector('.jobs-description-content__text');
        
        jobData.title = titleEl?.innerText || '';
        jobData.company = companyEl?.innerText || '';
        jobData.description = descEl?.innerText || '';
    }
    // Try to extract from Indeed
    else if (window.location.hostname.includes('indeed.com')) {
        const titleEl = document.querySelector('.jobsearch-JobInfoHeader-title');
        const companyEl = document.querySelector('[data-company-name]');
        const descEl = document.querySelector('#jobDescriptionText');
        
        jobData.title = titleEl?.innerText || '';
        jobData.company = companyEl?.innerText || '';
        jobData.description = descEl?.innerText || '';
    }
    // Try to extract from Glassdoor
    else if (window.location.hostname.includes('glassdoor.com')) {
        const titleEl = document.querySelector('.jobTitle');
        const companyEl = document.querySelector('.employerName');
        const descEl = document.querySelector('.jobDescriptionContent');
        
        jobData.title = titleEl?.innerText || '';
        jobData.company = companyEl?.innerText || '';
        jobData.description = descEl?.innerText || '';
    }

    // Combine all text for analysis
    jobData.text = `TITLE: ${jobData.title} | COMPANY: ${jobData.company} | DESCRIPTION: ${jobData.description}`;
    
    return jobData;
}
