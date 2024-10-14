document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    // Extract the aggregate from the URL
    const pathSegments = window.location.pathname.split('/');
    const aggregate = pathSegments[1]; // Assuming the aggregate is the first segment in the path

    const model = document.getElementById('model').value;
    const replacement = document.getElementById('replacement').value;
    const time = document.getElementById('time').value;

    const response = await fetch(`/${aggregate}/predict`, {  // Include aggregate in the URL
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model, replacement, time })
    });

    const result = await response.json();
    document.getElementById('result').innerText = `Predicted Value: ${result.compressive_strength.toFixed(2)}`;
});
