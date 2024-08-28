document.getElementById('analyzeButton').addEventListener('click', function() {
    const stockSymbol = document.getElementById('stockSymbol').value.trim().toUpperCase();
    
    if (stockSymbol === '') {
        alert('Please enter a valid stock symbol.');
        return;
    }

    // Display a loading message while fetching data
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '<p>Loading analysis...</p>';

    // Simulate fetching data
    setTimeout(() => {
        // Display a mock result
        resultDiv.innerHTML = `
            <h2>Analysis Result for ${stockSymbol}</h2>
            <p>Predicted price for the next day: $264.78</p>
            <p>Predicted price for 30 days: $278.45</p>
            <p>Predicted price for 60 days: $290.12</p>
            <p>Predicted price for 90 days: $302.50</p>
        `;
    }, 2000); // Simulated delay
});
