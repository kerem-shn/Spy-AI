// Wait for DOM
window.addEventListener("DOMContentLoaded", () => {
    // Create a dummy file and trigger the analysis
    const fileContent = "This is a dummy file to test the upload and analysis process of the application.";
    const blob = new Blob([fileContent], { type: "text/plain" });
    const file = new File([blob], "dummy.txt", { type: "text/plain" });

    // Assuming the file input exists, we don't even need to use the UI, we can just call runAnalysis
    // But runAnalysis relies on selectedFile variable.
    if(window.selectedFile !== undefined) {
        window.selectedFile = file;
        
        // Find the Analyze button and click it
        const analyzeBtn = document.getElementById("analyze-btn");
        if(analyzeBtn) {
            analyzeBtn.click();
            console.log("Analyze button clicked with dummy file.");
        } else {
            console.error("Analyze button not found!");
        }
    }
});
