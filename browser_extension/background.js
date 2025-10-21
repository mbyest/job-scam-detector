// Background script for extension management
chrome.runtime.onInstalled.addListener(() => {
    console.log('JobTrust AI extension installed');
});

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'analyzeJob') {
        analyzeJob(request.jobText)
            .then(result => sendResponse(result))
            .catch(error => sendResponse({ error: error.message }));
        return true; // Will respond asynchronously
    }
});

async function analyzeJob(jobText) {
    try {
        const response = await fetch('http://localhost:5001/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                job_text: jobText,
                source: 'browser_extension'
            })
        });

        if (!response.ok) {
            throw new Error('Analysis service unavailable');
        }

        return await response.json();
    } catch (error) {
        throw new Error('Failed to connect to analysis service');
    }
}
