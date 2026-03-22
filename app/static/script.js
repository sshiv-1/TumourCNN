document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("image-upload");
    const dropZone = document.getElementById("drop-zone");
    const errorText = document.getElementById("file-error");
    const fileNameText = document.getElementById("file-name");
    const submitBtn = document.getElementById("submit-btn");
    const spinner = document.getElementById("loading-spinner");
    const resultContainer = document.getElementById("result-container");
    const predictedLabelEl = document.getElementById("predicted-label");
    const confidenceText = document.getElementById("confidence-text");
    const confidenceBar = document.getElementById("confidence-bar");
    const scoresList = document.getElementById("scores-list");

    // Load Metrics on start
    fetchMetrics();

    // Setup Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        if(files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });

    fileInput.addEventListener("change", handleFileSelect);

    function handleFileSelect() {
        errorText.textContent = "";
        resultContainer.classList.add("hidden");
        
        const file = fileInput.files[0];
        if (file) {
            const validTypes = ["image/jpeg", "image/jpg", "image/png"];
            if (!validTypes.includes(file.type)) {
                errorText.textContent = "Invalid file type. Please upload a jpg, jpeg, or png.";
                fileNameText.classList.add("hidden");
                submitBtn.disabled = true;
                return;
            }
            
            fileNameText.textContent = `Selected: ${file.name}`;
            fileNameText.classList.remove("hidden");
            submitBtn.disabled = false;
        } else {
            fileNameText.classList.add("hidden");
            submitBtn.disabled = true;
        }
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        errorText.textContent = "";
        resultContainer.classList.add("hidden");
        
        const file = fileInput.files[0];
        if (!file) {
            errorText.textContent = "Please select an image file.";
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        // Show loading state
        spinner.classList.remove("hidden");
        submitBtn.disabled = true;
        submitBtn.textContent = "Analyzing...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => null);
                throw new Error(errData?.detail || "Prediction failed on server. Ensure model.pth is correctly placed.");
            }

            const data = await response.json();
            displayResult(data);
        } catch (error) {
            errorText.textContent = error.message;
        } finally {
            spinner.classList.add("hidden");
            submitBtn.disabled = false;
            submitBtn.textContent = "Run Analysis";
        }
    });

    function displayResult(data) {
        const { label, confidence, all_scores } = data;
        
        predictedLabelEl.textContent = label.toUpperCase();
        
        // Define tumor logic based on classes
        const isTumor = label !== "no_tumor";
        
        predictedLabelEl.className = ""; // clear classes
        confidenceBar.className = "progress-bar-fill";
        confidenceBar.style.width = "0%";
        
        if (isTumor) {
            predictedLabelEl.classList.add("text-danger");
            confidenceBar.classList.add("bg-danger");
        } else {
            predictedLabelEl.classList.add("text-success");
            confidenceBar.classList.add("bg-success");
        }

        const percentage = (confidence * 100).toFixed(2);
        confidenceText.textContent = `${percentage}%`;
        
        // Reflow to restart CSS animation
        void confidenceBar.offsetWidth;
        
        setTimeout(() => {
            confidenceBar.style.width = `${percentage}%`;
        }, 50);

        // Render all scores
        scoresList.innerHTML = "";
        Object.entries(all_scores).forEach(([className, score]) => {
            const scorePct = (score * 100).toFixed(2);
            
            let barColorClass = "bg-danger";
            if (className === "no_tumor") barColorClass = "bg-success";
            
            const html = `
                <div class="score-item">
                    <div class="score-label-row">
                        <span>${className}</span>
                        <span>${scorePct}%</span>
                    </div>
                    <div class="progress-bar-bg" style="height: 8px;">
                        <div class="progress-bar-fill ${barColorClass}" style="width: ${scorePct}%;"></div>
                    </div>
                </div>
            `;
            scoresList.insertAdjacentHTML("beforeend", html);
        });

        resultContainer.classList.remove("hidden");
        
        // Smooth scroll to result
        setTimeout(() => {
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    async function fetchMetrics() {
        try {
            const res = await fetch("/metrics");
            if (!res.ok) return;
            const metrics = await res.json();
            renderMetrics(metrics);
        } catch (e) {
            console.error("Failed to load metrics", e);
        }
    }

    function renderMetrics(metrics) {
        const cm = metrics.confusion_matrix;
        if (cm) {
            document.getElementById("cm-tp").textContent = `TP: ${cm.TP}`;
            document.getElementById("cm-fp").textContent = `FP: ${cm.FP}`;
            document.getElementById("cm-fn").textContent = `FN: ${cm.FN}`;
            document.getElementById("cm-tn").textContent = `TN: ${cm.TN}`;
        }

        const ctx = document.getElementById('metricsChart').getContext('2d');
        
        // ChartJS config defaults
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.color = "#475569";

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                datasets: [{
                    label: 'Score',
                    data: [
                        metrics.accuracy, 
                        metrics.precision, 
                        metrics.recall, 
                        metrics.f1_score
                    ],
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.85)',
                        'rgba(16, 185, 129, 0.85)',
                        'rgba(245, 158, 11, 0.85)',
                        'rgba(139, 92, 246, 0.85)'
                    ],
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        grid: {
                            color: '#e2e8f0',
                            drawBorder: false
                        }
                    },
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        padding: 12,
                        titleFont: { size: 14, weight: '600' },
                        bodyFont: { size: 14 },
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                return `Score: ${(context.raw * 100).toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
});
