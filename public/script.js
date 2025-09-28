document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selectors ---
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const stockSelector = document.getElementById('stockSelector');
    const initialMessage = document.getElementById('initialMessage');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');
    const chartDiv = document.getElementById('predictionChart');
    const timeRangeButtons = document.getElementById('timeRangeButtons');
    const sentimentSummary = document.getElementById('sentimentSummary');
    const lastActualPriceElem = document.getElementById('lastActualPrice');
    const predictedNextDayPriceElem = document.getElementById('predictedNextDayPrice');
    const trendPredictionElem = document.getElementById('trendPrediction');
    const directionalAccuracyElem = document.getElementById('directionalAccuracy');
    const mapeElem = document.getElementById('mape');
    
    let allStocks = [];
    let currentStockData = null;
    const dataCache = new Map();

    // --- Core Functions ---
    function startLoadingAnimation() {
        const path = document.querySelector('#loading-animation path');
        if (!path) return;
        const length = path.getTotalLength();
        path.style.strokeDasharray = length;
        path.style.strokeDashoffset = length;
        anime({
            targets: path,
            strokeDashoffset: [anime.setDashoffset, 0],
            easing: 'easeInOutSine',
            duration: 2000,
            delay: 500,
            direction: 'alternate',
            loop: true
        });
    }

    async function initializeApp() {
        try {
            loadingText.textContent = 'Loading available stocks...';
            const manifestResponse = await fetch('./predictions/manifest.json');
            if (!manifestResponse.ok) throw new Error('Manifest file not found.');
            allStocks = await manifestResponse.json();
            
            populateDropdown(allStocks);
            
            const stocksToPreload = allStocks.slice(0, 5);
            loadingText.textContent = `Caching data for top stocks...`;

            await Promise.all(
                stocksToPreload.map(stock =>
                    fetch(`./predictions/${stock}_prediction.json`)
                        .then(res => res.json())
                        .then(data => dataCache.set(stock, data))
                        .catch(err => console.error(`Failed to cache ${stock}:`, err))
                )
            );
            
            await new Promise(resolve => setTimeout(resolve, 1500));

            loadingText.textContent = 'Initialization complete!';
            loadingOverlay.classList.add('hidden');
            
            stockSelector.disabled = false;
        } catch (error) {
            loadingText.textContent = `Error: ${error.message}`;
            console.error("Failed to initialize app:", error);
        }
    }

    function populateDropdown(stocks) {
        stockSelector.innerHTML = '<option value="" disabled selected>Select a stock...</option>';
        stocks.forEach(stock => {
            const option = document.createElement('option');
            option.value = stock;
            option.textContent = stock;
            stockSelector.appendChild(option);
        });
    }

    async function loadPredictionData(stock) {
        if (!stock) return;
        resetUI();
        
        if (dataCache.has(stock)) {
            currentStockData = dataCache.get(stock);
            renderAllComponents(currentStockData);
            return;
        }

        loader.style.display = 'flex';
        try {
            const dataResponse = await fetch(`./predictions/${stock}_prediction.json`);
            if (!dataResponse.ok) throw new Error(`Prediction data for ${stock} not found.`);
            currentStockData = await dataResponse.json();
            dataCache.set(stock, currentStockData);
            renderAllComponents(currentStockData);
        } catch (error) {
            displayError(error.message);
        } finally {
            loader.style.display = 'none';
        }
    }
    
    function renderAllComponents(data) {
        initialMessage.style.display = 'none';
        
        chartDiv.classList.remove('hidden');
        timeRangeButtons.classList.remove('hidden');

        updateDetails(data);
        drawChart(data);
    }

    function updateDetails(data) {
        sentimentSummary.textContent = data.summary || 'N/A';
        
        const lastActual = data.historical && data.historical.length > 0 ? data.historical[data.historical.length - 1].close : null;
        lastActualPriceElem.textContent = lastActual !== null ? `$${lastActual.toFixed(2)}` : 'N/A';
        
        const nextDayPred = data.next_day_prediction;
        predictedNextDayPriceElem.textContent = nextDayPred !== null && nextDayPred !== undefined ? `$${nextDayPred.toFixed(2)}` : 'N/A';

        const endPrediction = data.predictions && data.predictions.length > 0 ? data.predictions[data.predictions.length - 1].predicted_close : null;
        
        if (lastActual !== null && endPrediction !== null) {
            const trend = endPrediction > lastActual ? 'Upward 📈' : 'Downward 📉';
            const trendColor = endPrediction > lastActual ? 'text-green-400' : 'text-red-500';
            trendPredictionElem.textContent = trend;
            trendPredictionElem.className = `font-medium ${trendColor}`;
        } else {
             trendPredictionElem.textContent = 'N/A';
        }

        if (data.accuracy && data.accuracy.mape !== null && data.accuracy.directional_accuracy !== null) {
            directionalAccuracyElem.textContent = `${data.accuracy.directional_accuracy.toFixed(2)}%`;
            mapeElem.textContent = `${data.accuracy.mape.toFixed(2)}%`;
        } else {
            directionalAccuracyElem.textContent = 'N/A';
            mapeElem.textContent = 'N/A';
        }
    }

    function drawChart(data) {
        const historical = data.historical;
        const predictions = data.predictions;

        const trace1 = {
            x: historical.map(d => d.date),
            open: historical.map(d => d.open),
            high: historical.map(d => d.high),
            low: historical.map(d => d.low),
            close: historical.map(d => d.close),
            type: 'candlestick',
            name: 'Historical Price',
            increasing: { line: { color: '#22c55e' } },
            decreasing: { line: { color: '#d20f39' } }
        };
        
        const lastHistoricalDate = historical.length > 0 ? historical[historical.length - 1].date : new Date().toISOString().split('T')[0];
        const lastHistoricalClose = historical.length > 0 ? historical[historical.length - 1].close : null;

        const trace2 = {
            x: [lastHistoricalDate, ...predictions.map(d => d.date)],
            y: [lastHistoricalClose, ...predictions.map(d => d.predicted_close)],
            mode: 'lines+markers',
            name: 'Predicted Trend',
            // --- UPDATED: Added glow effect and improved tooltip ---
            line: { 
                color: '#ea76cb', 
                width: 2.5,
                shape: 'spline', // Smoother line
                shadow: { // Glow effect
                    color: '#ea76cb',
                    width: 8,
                    blur: 4
                }
            },
            marker: { size: 4 },
            hovertemplate: '<b>Predicted Price</b><br>%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>',
            hoverlabel: {
                bgcolor: '#ea76cb',
                font: { color: '#000000' }
            }
        };

        const allDates = [...historical.map(d => d.date), ...predictions.map(d => d.date)];
        const endDate = allDates.length > 0 ? new Date(Math.max.apply(null, allDates.map(d => new Date(d)))) : new Date();
        const startDate = new Date(endDate);
        startDate.setMonth(endDate.getMonth() - 6);

        const layout = {
            title: `<b>${data.stock} Price Prediction</b>`,
            xaxis: { 
                title: 'Date',
                rangeslider: { visible: false },
                type: 'date',
                range: [startDate.toISOString().split('T')[0], endDate.toISOString().split('T')[0]]
            },
            yaxis: { 
                title: 'Price (USD)',
                autorange: true 
            },
            template: 'plotly_dark',
            paper_bgcolor: '#161b22',
            plot_bgcolor: '#161b22',
            font: { color: '#c9d1d9', family: 'Inter, sans-serif' },
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1
            },
            margin: { l: 60, r: 20, t: 60, b: 50 }
        };

        Plotly.newPlot(chartDiv, [trace1, trace2], layout, {responsive: true});
        
        document.querySelectorAll('.time-range-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.range === '6M');
        });
    }

    function updateChartRange(range) {
        if (!currentStockData || !chartDiv.data) return;

        const allDates = [...currentStockData.historical.map(d => d.date), ...currentStockData.predictions.map(d => d.date)];
        if (allDates.length === 0) return;
        
        const endDate = new Date(Math.max.apply(null, allDates.map(d => new Date(d))));
        let startDate;

        switch(range) {
            case '6M':
                startDate = new Date(endDate);
                startDate.setMonth(endDate.getMonth() - 6);
                break;
            case '1Y':
                startDate = new Date(endDate);
                startDate.setFullYear(endDate.getFullYear() - 1);
                break;
            case '2Y':
                startDate = new Date(endDate);
                startDate.setFullYear(endDate.getFullYear() - 2);
                break;
            case '3Y':
                startDate = new Date(endDate);
                startDate.setFullYear(endDate.getFullYear() - 3);
                break;
        }

        Plotly.update(chartDiv, {}, {
            'xaxis.range': [startDate.toISOString().split('T')[0], endDate.toISOString().split('T')[0]],
            'yaxis.autorange': true
        });

        document.querySelectorAll('.time-range-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.range === range);
        });
    }

    function displayError(message) {
        errorMessage.textContent = `Error: ${message}.`;
        errorMessage.style.display = 'block';
        chartDiv.classList.add('hidden');
        timeRangeButtons.classList.add('hidden');
    }

    function resetUI() {
        chartDiv.innerHTML = '';
        chartDiv.classList.add('hidden');
        timeRangeButtons.classList.add('hidden');
        errorMessage.style.display = 'none';
        loader.style.display = 'none';
        initialMessage.style.display = 'flex';
        sentimentSummary.textContent = 'Select a stock to see the summary.';
        lastActualPriceElem.textContent = '-';
        predictedNextDayPriceElem.textContent = '-';
        trendPredictionElem.textContent = '-';
        trendPredictionElem.className = 'font-medium';
        directionalAccuracyElem.textContent = '-';
        mapeElem.textContent = '-';
    }

    // --- Event Listeners ---
    stockSelector.addEventListener('change', (e) => {
        loadPredictionData(e.target.value);
    });

    timeRangeButtons.addEventListener('click', (e) => {
        if (e.target.classList.contains('time-range-btn')) {
            updateChartRange(e.target.dataset.range);
        }
    });
    
    // --- App Initialization ---
    startLoadingAnimation();
    initializeApp();
});

