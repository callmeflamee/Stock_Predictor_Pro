document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selectors ---
    const stockSelector = document.getElementById('stockSelector');
    const initialMessage = document.getElementById('initialMessage');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');
    const chartFrame = document.getElementById('predictionChartFrame');
    const sentimentSummary = document.getElementById('sentimentSummary');
    const lastActualPriceElem = document.getElementById('lastActualPrice');
    const predictedNextDayPriceElem = document.getElementById('predictedNextDayPrice');
    const trendPredictionElem = document.getElementById('trendPrediction');
    // NEW: Accuracy Element Selectors
    const directionalAccuracyElem = document.getElementById('directionalAccuracy');
    const mapeElem = document.getElementById('mape');

    /**
     * Fetches the manifest of available stocks and populates the dropdown selector.
     */
    async function populateStockSelector() {
        try {
            const response = await fetch('./predictions/manifest.json');
            if (!response.ok) throw new Error('Manifest file not found. Please run the Python pipeline.');
            const stocks = await response.json();
            stockSelector.innerHTML = '<option selected disabled>Select a stock</option>';
            stocks.forEach(stock => {
                const option = document.createElement('option');
                option.value = stock;
                option.textContent = stock;
                stockSelector.appendChild(option);
            });
            stockSelector.disabled = false;
        } catch (error) {
            stockSelector.innerHTML = `<option selected disabled>Error: ${error.message}</option>`;
            console.error("Failed to load manifest:", error);
        }
    }

    /**
     * Loads the prediction data and chart for a specific stock.
     * @param {string} stock The stock ticker symbol.
     */
    async function loadPredictionData(stock) {
        resetUI();
        loader.style.display = 'flex';
        initialMessage.style.display = 'none';

        try {
            // Fetch the JSON data for details panel
            const dataResponse = await fetch(`./predictions/${stock}_prediction.json`);
            if (!dataResponse.ok) throw new Error(`Prediction JSON for ${stock} not found.`);
            const data = await dataResponse.json();
            
            // Update the side panel details
            updateDetails(data);

            // Set the iframe source to the pre-generated HTML chart
            chartFrame.src = `./predictions/${stock}_chart.html`;
            chartFrame.style.display = 'block';

        } catch (error) {
            displayError(error.message);
            console.error(`Failed to load prediction data for ${stock}:`, error);
        } finally {
            loader.style.display = 'none';
        }
    }

    /**
     * Updates the side panel with prediction details and sentiment summary.
     * @param {object} data The prediction and historical data for a stock.
     */
    function updateDetails(data) {
        sentimentSummary.textContent = data.summary;
        const lastActual = data.historical[data.historical.length - 1].close;
        lastActualPriceElem.textContent = `$${lastActual.toFixed(2)}`;
        const nextDayPrediction = data.predictions[0].predicted_close;
        predictedNextDayPriceElem.textContent = `$${parseFloat(nextDayPrediction).toFixed(2)}`;

        const trend = parseFloat(nextDayPrediction) > lastActual ? 'Upward 📈' : 'Downward 📉';
        const trendColor = parseFloat(nextDayPrediction) > lastActual ? 'text-green-400' : 'text-red-500';
        
        trendPredictionElem.textContent = trend;
        trendPredictionElem.className = `font-medium ${trendColor}`;

        // NEW: Update Accuracy Metrics
        if (data.accuracy && data.accuracy.mape !== null && data.accuracy.directional_accuracy !== null) {
            directionalAccuracyElem.textContent = `${data.accuracy.directional_accuracy.toFixed(2)}%`;
            mapeElem.textContent = `${data.accuracy.mape.toFixed(2)}%`;
        } else {
            directionalAccuracyElem.textContent = 'N/A';
            mapeElem.textContent = 'N/A';
        }
    }

    /**
     * Displays an error message in the chart area.
     * @param {string} message The error message to display.
     */
    function displayError(message) {
        errorMessage.textContent = `Error: ${message}. Please ensure you have run the full Python pipeline via main.py.`;
        errorMessage.style.display = 'block';
        chartFrame.style.display = 'none';
    }

    /**
     * Resets the UI to its initial state before loading new data.
     */
    function resetUI() {
        chartFrame.src = 'about:blank'; // Clear the iframe
        chartFrame.style.display = 'none';
        errorMessage.style.display = 'none';
        loader.style.display = 'none';
        initialMessage.style.display = 'flex';
        sentimentSummary.textContent = 'Select a stock to see the summary.';
        lastActualPriceElem.textContent = '-';
        predictedNextDayPriceElem.textContent = '-';
        trendPredictionElem.textContent = '-';
        trendPredictionElem.className = 'font-medium text-white';
        // NEW: Reset Accuracy Metrics
        directionalAccuracyElem.textContent = '-';
        mapeElem.textContent = '-';
    }

    // --- Event Listeners ---
    stockSelector.addEventListener('change', (e) => {
        const selectedStock = e.target.value;
        if (selectedStock && selectedStock !== "Select a stock") {
            loadPredictionData(selectedStock);
        }
    });
    
    // --- Initial Load ---
    populateStockSelector();
});

