document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const model = document.getElementById('model').value;
    const replacement = document.getElementById('replacement').value;
    const time = document.getElementById('time').value;

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model, replacement, time })
    });

    const result = await response.json();
    document.getElementById('result').innerText = `Predicted Compressive Strength: ${result.compressive_strength.toFixed(2)} MPa`;
});
