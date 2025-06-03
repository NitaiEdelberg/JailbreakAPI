// function that Called when the "Check Prompt" button is clicked
async function checkPrompt() {
  const promptText = document.getElementById("promptInput").value;
  const resultDiv = document.getElementById("result");

  // Reset the result area to default style and message
  resultDiv.className = "mt-4 fw-bold text-center";
  resultDiv.textContent = "Checking...";

  try {
    const response = await fetch("https://jailbreak-api-backend.onrender.com/detect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: promptText }) // Send user input to backend
    });

    if (response.status === 403) {
      // Blocked input detected (Forbidden)
      resultDiv.classList.add("text-danger");
      resultDiv.textContent = "Blocked: Dangerous prompt detected (403 Forbidden)";
    } else if (response.ok) {
      const data = await response.json();

      // Check if the backend detected a suspicious prompt
      if (data.detected === true) {
        resultDiv.classList.add("text-warning");
        resultDiv.textContent = "Warning: Suspicious prompt detected.";
      } else {
        resultDiv.classList.add("text-success");
        resultDiv.textContent = "Safe: No threat detected.";
      }
    } else {
      // Server returned an unexpected status code
      resultDiv.classList.add("text-muted");
      resultDiv.textContent = "Unexpected error: " + response.status;
    }
  } catch (error) {
    // Network or connection error
    resultDiv.classList.add("text-danger");
    resultDiv.textContent = "Network error: " + error;
  }
}
