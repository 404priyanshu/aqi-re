/**
 * AQI Monitoring Dashboard - JavaScript Application
 */

// API Base URL
const API_BASE = '';

// Chart instances
let aqiTrendChart = null;
let pollutantChart = null;
let modelComparisonChart = null;

// Chart.js availability flag
const chartsAvailable = typeof Chart !== 'undefined';

// Current data
let currentData = null;
let historyData = [];

// AQI Colors
const AQI_COLORS = {
    'Good': '#00e400',
    'Moderate': '#ffff00',
    'Unhealthy for Sensitive Groups': '#ff7e00',
    'Unhealthy': '#ff0000',
    'Very Unhealthy': '#8f3f97',
    'Hazardous': '#7e0023'
};

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    if (chartsAvailable) {
        initializeCharts();
    } else {
        console.warn('Chart.js not available. Charts will be disabled.');
    }
    refreshData();
    
    // Auto-refresh every 5 minutes
    setInterval(refreshData, 300000);
});

/**
 * Initialize Chart.js charts
 */
function initializeCharts() {
    if (!chartsAvailable) return;
    
    // AQI Trend Chart
    const trendCtx = document.getElementById('aqiTrendChart').getContext('2d');
    aqiTrendChart = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'AQI',
                data: [],
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#c9d1d9' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#8b949e' },
                    grid: { color: 'rgba(139, 148, 158, 0.1)' }
                },
                y: {
                    ticks: { color: '#8b949e' },
                    grid: { color: 'rgba(139, 148, 158, 0.1)' },
                    beginAtZero: true
                }
            }
        }
    });

    // Pollutant Distribution Chart
    const pollutantCtx = document.getElementById('pollutantChart').getContext('2d');
    pollutantChart = new Chart(pollutantCtx, {
        type: 'bar',
        data: {
            labels: ['PM2.5', 'PM10', 'CO', 'NO₂', 'SO₂', 'O₃'],
            datasets: [{
                label: 'Current Level',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(255, 205, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(153, 102, 255, 0.8)'
                ],
                borderColor: [
                    'rgb(255, 99, 132)',
                    'rgb(255, 159, 64)',
                    'rgb(255, 205, 86)',
                    'rgb(75, 192, 192)',
                    'rgb(54, 162, 235)',
                    'rgb(153, 102, 255)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#c9d1d9' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#8b949e' },
                    grid: { color: 'rgba(139, 148, 158, 0.1)' }
                },
                y: {
                    ticks: { color: '#8b949e' },
                    grid: { color: 'rgba(139, 148, 158, 0.1)' },
                    beginAtZero: true
                }
            }
        }
    });

    // Model Comparison Chart
    const modelCtx = document.getElementById('modelComparisonChart').getContext('2d');
    modelComparisonChart = new Chart(modelCtx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#c9d1d9' }
                }
            },
            scales: {
                r: {
                    ticks: { 
                        color: '#8b949e',
                        backdropColor: 'transparent'
                    },
                    grid: { color: 'rgba(139, 148, 158, 0.2)' },
                    pointLabels: { color: '#c9d1d9' },
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

/**
 * Refresh all data
 */
async function refreshData() {
    updateStatus('Refreshing data...');
    
    try {
        // Fetch live AQI first (needed for prediction)
        await fetchLiveAQI();
        
        // Then fetch prediction and other data in parallel
        await Promise.all([
            fetchPrediction(),
            fetchHistory(),
            fetchModelMetrics()
        ]);
        
        updateStatus('Connected');
        updateLastUpdate();
        showToast('Data refreshed successfully', 'success');
    } catch (error) {
        console.error('Error refreshing data:', error);
        updateStatus('Error - Check connection');
        showToast('Failed to refresh data', 'error');
    }
}

/**
 * Fetch live AQI data
 */
async function fetchLiveAQI() {
    try {
        const response = await fetch(`${API_BASE}/api/aqi/live`);
        const result = await response.json();
        
        if (result.status === 'success') {
            currentData = result.data;
            updateAQIDisplay(currentData);
            updatePollutantChart(currentData);
        }
    } catch (error) {
        console.error('Error fetching live AQI:', error);
        throw error;
    }
}

/**
 * Fetch prediction
 */
async function fetchPrediction() {
    if (!currentData) return;
    
    try {
        const params = new URLSearchParams({
            pm25: currentData.pm25 || 0,
            pm10: currentData.pm10 || 0,
            co: currentData.co || 0,
            no2: currentData.no2 || 0,
            so2: currentData.so2 || 0,
            o3: currentData.o3 || 0,
            temperature: currentData.temperature || 25,
            humidity: currentData.humidity || 50
        });
        
        const response = await fetch(`${API_BASE}/api/aqi/predict?${params}`);
        const result = await response.json();
        
        if (result.status === 'success') {
            updatePredictionDisplay(result.prediction);
        }
    } catch (error) {
        console.error('Error fetching prediction:', error);
    }
}

/**
 * Fetch historical data
 */
async function fetchHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/aqi/history?limit=50`);
        const result = await response.json();
        
        if (result.status === 'success' && result.data.length > 0) {
            historyData = result.data.reverse();
            updateTrendChart(historyData);
        }
    } catch (error) {
        console.error('Error fetching history:', error);
    }
}

/**
 * Fetch model metrics
 */
async function fetchModelMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/model/metrics`);
        const result = await response.json();
        
        if (result.status === 'success' && result.metrics.models) {
            updateMetricsTable(result.metrics);
            updateModelComparisonChart(result.metrics);
        }
    } catch (error) {
        console.error('Error fetching model metrics:', error);
    }
}

/**
 * Update AQI display
 */
function updateAQIDisplay(data) {
    const aqiValue = document.getElementById('aqi-value');
    const aqiCategory = document.getElementById('aqi-category');
    const aqiCity = document.getElementById('aqi-city');
    const aqiTime = document.getElementById('aqi-time');
    const aqiCard = document.getElementById('aqi-card');
    
    aqiValue.textContent = data.aqi || '--';
    aqiCategory.textContent = data.category || 'Unknown';
    aqiCity.textContent = `📍 ${data.city || 'Unknown'}`;
    aqiTime.textContent = `🕐 ${formatTime(data.timestamp)}`;
    
    // Set color based on category
    const categoryClass = getAQICategoryClass(data.category);
    aqiCategory.className = `aqi-category ${categoryClass}`;
    aqiValue.style.color = AQI_COLORS[data.category] || '#c9d1d9';
}

/**
 * Update prediction display
 */
function updatePredictionDisplay(prediction) {
    const predictionCategory = document.getElementById('prediction-category');
    const predictionConfidence = document.getElementById('prediction-confidence');
    const predictionModel = document.getElementById('prediction-model');
    const predictionCard = document.getElementById('prediction-card');
    
    predictionCategory.textContent = prediction.category || 'Unknown';
    predictionCategory.style.color = prediction.color || '#c9d1d9';
    
    if (prediction.confidence) {
        predictionConfidence.textContent = `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;
    } else {
        predictionConfidence.textContent = 'Confidence: N/A';
    }
    
    predictionModel.textContent = prediction.note || 'Model: ML Classifier';
}

/**
 * Update pollutant values and chart
 */
function updatePollutantChart(data) {
    // Update individual pollutant displays
    document.getElementById('pm25').textContent = data.pm25?.toFixed(1) || '--';
    document.getElementById('pm10').textContent = data.pm10?.toFixed(1) || '--';
    document.getElementById('co').textContent = data.co?.toFixed(2) || '--';
    document.getElementById('no2').textContent = data.no2?.toFixed(1) || '--';
    document.getElementById('so2').textContent = data.so2?.toFixed(1) || '--';
    document.getElementById('o3').textContent = data.o3?.toFixed(1) || '--';
    
    // Update chart if available
    if (pollutantChart) {
        pollutantChart.data.datasets[0].data = [
            data.pm25 || 0,
            data.pm10 || 0,
            data.co || 0,
            data.no2 || 0,
            data.so2 || 0,
            data.o3 || 0
        ];
        pollutantChart.update();
    }
}

/**
 * Update trend chart with historical data
 */
function updateTrendChart(data) {
    if (!aqiTrendChart) return;
    
    const labels = data.map(d => formatTime(d.timestamp));
    const values = data.map(d => d.aqi);
    
    aqiTrendChart.data.labels = labels;
    aqiTrendChart.data.datasets[0].data = values;
    aqiTrendChart.update();
}

/**
 * Update metrics table
 */
function updateMetricsTable(metrics) {
    const tbody = document.getElementById('metrics-body');
    const bestModel = metrics.best_model;
    
    let html = '';
    for (const [name, m] of Object.entries(metrics.models)) {
        const isBest = name === bestModel;
        html += `
            <tr class="${isBest ? 'best-model' : ''}">
                <td>${name}</td>
                <td>${(m.accuracy * 100).toFixed(2)}%</td>
                <td>${(m.precision * 100).toFixed(2)}%</td>
                <td>${(m.recall * 100).toFixed(2)}%</td>
                <td>${(m.f1_score * 100).toFixed(2)}%</td>
            </tr>
        `;
    }
    tbody.innerHTML = html;
}

/**
 * Update model comparison radar chart
 */
function updateModelComparisonChart(metrics) {
    if (!modelComparisonChart) return;
    
    const colors = [
        'rgba(255, 99, 132, 0.7)',
        'rgba(54, 162, 235, 0.7)',
        'rgba(255, 206, 86, 0.7)',
        'rgba(75, 192, 192, 0.7)',
        'rgba(153, 102, 255, 0.7)'
    ];
    
    const datasets = [];
    let colorIndex = 0;
    
    for (const [name, m] of Object.entries(metrics.models)) {
        datasets.push({
            label: name,
            data: [m.accuracy, m.precision, m.recall, m.f1_score],
            borderColor: colors[colorIndex % colors.length],
            backgroundColor: colors[colorIndex % colors.length].replace('0.7', '0.2'),
            pointBackgroundColor: colors[colorIndex % colors.length]
        });
        colorIndex++;
    }
    
    modelComparisonChart.data.datasets = datasets;
    modelComparisonChart.update();
}

/**
 * Retrain model
 */
async function retrainModel() {
    showToast('Starting model retraining...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/api/model/retrain`, {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.status === 'accepted') {
            showToast('Model retraining initiated. This may take a few minutes.', 'success');
            
            // Poll for completion
            setTimeout(fetchModelMetrics, 30000);
        }
    } catch (error) {
        console.error('Error retraining model:', error);
        showToast('Failed to start retraining', 'error');
    }
}

/**
 * Get AQI category CSS class
 */
function getAQICategoryClass(category) {
    const mapping = {
        'Good': 'aqi-good',
        'Moderate': 'aqi-moderate',
        'Unhealthy for Sensitive Groups': 'aqi-usg',
        'Unhealthy': 'aqi-unhealthy',
        'Very Unhealthy': 'aqi-very-unhealthy',
        'Hazardous': 'aqi-hazardous'
    };
    return mapping[category] || '';
}

/**
 * Format timestamp for display
 */
function formatTime(timestamp) {
    if (!timestamp) return '--';
    try {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return '--';
    }
}

/**
 * Update status display
 */
function updateStatus(status) {
    document.getElementById('status').textContent = `Status: ${status}`;
}

/**
 * Update last update time
 */
function updateLastUpdate() {
    const now = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    document.getElementById('last-update').textContent = `Last Update: ${now}`;
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}
