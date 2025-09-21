document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selectors ---
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const searchInput = document.getElementById('stockSearchInput');
    const dropdownList = document.getElementById('stockDropdownList');
    const initialMessage = document.getElementById('initialMessage');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');
    const chartFrame = document.getElementById('predictionChartFrame');
    const sentimentSummary = document.getElementById('sentimentSummary');
    const lastActualPriceElem = document.getElementById('lastActualPrice');
    const predictedNextDayPriceElem = document.getElementById('predictedNextDayPrice');
    const trendPredictionElem = document.getElementById('trendPrediction');
    const directionalAccuracyElem = document.getElementById('directionalAccuracy');
    const mapeElem = document.getElementById('mape');
    
    // --- Caching and State Management ---
    let allStocks = [];
    const dataCache = new Map();

    // --- Core Functions ---

    /**
     * Creates and controls the loading screen animation using Anime.js.
     */
    function startLoadingAnimation() {
        const path = document.querySelector('#loading-animation path');
        const length = path.getTotalLength();
        path.style.strokeDasharray = length;
        path.style.strokeDashoffset = length;

        anime({
            targets: path,
            strokeDashoffset: [anime.setDashoffset, 0],
            easing: 'easeInOutSine',
            // --- UPDATED: Faster animation duration (2 seconds per cycle) ---
            duration: 1500,
            delay: 500,
            direction: 'alternate',
            loop: true
        });
    }

    /**
     * Fetches and caches the prediction data for the top N stocks,
     * while ensuring the loading screen is visible for a minimum duration.
     */
    async function preloadTopStocks() {
        try {
            loadingText.textContent = 'Loading available stocks...';
            const manifestResponse = await fetch('./predictions/manifest.json');
            if (!manifestResponse.ok) throw new Error('Manifest file not found.');
            allStocks = await manifestResponse.json();
            
            const stocksToPreload = allStocks.slice(0, 10);
            loadingText.textContent = `Caching data for ${stocksToPreload.length} top stocks...`;

            // Promise 1: The data caching operation
            const cachingPromise = Promise.all(
                stocksToPreload.map(stock =>
                    fetch(`./predictions/${stock}_prediction.json`)
                        .then(res => res.json())
                        .then(data => dataCache.set(stock, data))
                        .catch(err => console.error(`Failed to cache ${stock}:`, err))
                )
            );

            // Promise 2: A simple timer that resolves after 3 seconds
            const timerPromise = new Promise(resolve => setTimeout(resolve, 3000));

            // Wait for BOTH the caching and the timer to finish
            await Promise.all([cachingPromise, timerPromise]);

            loadingText.textContent = 'Initialization complete!';
            
            // Fade out the loading screen
            loadingOverlay.classList.add('hidden');
            
            // Populate the main UI now that everything is ready
            renderDropdownList(allStocks);
            searchInput.disabled = false;
            searchInput.placeholder = `Search or select from ${allStocks.length} stocks...`;

        } catch (error) {
            loadingText.textContent = `Error: ${error.message}`;
            console.error("Failed to preload data:", error);
        }
    }


    function renderDropdownList(stocks) {
        // (This function remains the same)
        dropdownList.innerHTML = '';
        if (stocks.length === 0) {
            dropdownList.innerHTML = '<div class="dropdown-item text-gray-500">No stocks found</div>';
        }
        stocks.forEach(stock => {
            const item = document.createElement('div');
            item.className = 'dropdown-item';
            item.textContent = stock;
            item.dataset.stock = stock;
            dropdownList.appendChild(item);
        });
    }

    /**
     * Loads prediction data, first checking the cache for instant display.
     */
    async function loadPredictionData(stock) {
        // (This function is correct and remains unchanged)
        resetUI();
        initialMessage.style.display = 'none';

        if (dataCache.has(stock)) {
            console.log(`Cache hit for ${stock}. Displaying instantly.`);
            const data = dataCache.get(stock);
            updateDetails(data);
            chartFrame.src = `./predictions/${stock}_chart.html`;
            chartFrame.classList.remove('hidden');
            return;
        }

        console.log(`Cache miss for ${stock}. Fetching from network...`);
        loader.style.display = 'flex';
        try {
            const dataResponse = await fetch(`./predictions/${stock}_prediction.json`);
            if (!dataResponse.ok) throw new Error(`Prediction JSON for ${stock} not found.`);
            const data = await dataResponse.json();
            
            dataCache.set(stock, data);
            updateDetails(data);
            chartFrame.src = `./predictions/${stock}_chart.html`;
            chartFrame.classList.remove('hidden');

        } catch (error) {
            displayError(error.message);
        } finally {
            loader.style.display = 'none';
        }
    }
    
    // (The rest of the functions: updateDetails, displayError, resetUI, and event listeners remain the same)
    function updateDetails(data) {
        sentimentSummary.textContent = data.summary || 'N/A';
        const lastActual = data.historical && data.historical.length > 0 ? data.historical[data.historical.length - 1].close : null;
        lastActualPriceElem.textContent = lastActual !== null ? `$${lastActual.toFixed(2)}` : 'N/A';
        const hybridPrediction = data.hybrid_next_day_prediction;
        predictedNextDayPriceElem.textContent = hybridPrediction !== null && hybridPrediction !== undefined ? `$${hybridPrediction.toFixed(2)}` : 'N/A';
        const lstmEndPrediction = data.predictions && data.predictions.length > 0 ? data.predictions[data.predictions.length - 1].predicted_close : null;
        if (lastActual !== null && lstmEndPrediction !== null) {
            const trend = lstmEndPrediction > lastActual ? 'Upward 📈' : 'Downward 📉';
            const trendColor = lstmEndPrediction > lastActual ? 'text-green-400' : 'text-red-500';
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
    function displayError(message) {
        errorMessage.textContent = `Error: ${message}.`;
        errorMessage.style.display = 'block';
        chartFrame.classList.add('hidden');
    }
    function resetUI() {
        chartFrame.src = 'about:blank';
        chartFrame.classList.add('hidden');
        errorMessage.style.display = 'none';
        loader.style.display = 'none';
        initialMessage.style.display = 'flex';
        sentimentSummary.textContent = 'Select a stock to see the summary.';
        lastActualPriceElem.textContent = '-';
        predictedNextDayPriceElem.textContent = '-';
        trendPredictionElem.textContent = '-';
        trendPredictionElem.className = 'font-medium text-white';
        directionalAccuracyElem.textContent = '-';
        mapeElem.textContent = '-';
    }
    searchInput.addEventListener('focus', () => { dropdownList.classList.remove('hidden'); renderDropdownList(allStocks); });
    searchInput.addEventListener('keyup', () => {
        const filter = searchInput.value.toUpperCase();
        const filteredStocks = allStocks.filter(stock => stock.toUpperCase().startsWith(filter));
        renderDropdownList(filteredStocks);
    });
    dropdownList.addEventListener('click', (e) => {
        if (e.target && e.target.classList.contains('dropdown-item')) {
            const selectedStock = e.target.dataset.stock;
            searchInput.value = selectedStock;
            dropdownList.classList.add('hidden');
            loadPredictionData(selectedStock);
        }
    });
    document.addEventListener('click', (e) => { if (!searchInput.contains(e.target) && !dropdownList.contains(e.target)) { dropdownList.classList.add('hidden'); } });
    
    // --- Initial Load ---
    startLoadingAnimation();
    preloadTopStocks();
});

