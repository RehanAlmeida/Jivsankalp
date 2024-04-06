const video = document.getElementById('camera');
const resultDisplay = document.getElementById('result');
const percentageDisplay = document.getElementById('percentage');
const locationDisplay = document.getElementById('location'); // Added location display element
let map;

navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        video.srcObject = stream;
        detectImage(stream);
    })
    .catch(function (err) {
        console.error('Camera access error:', err);
    });

async function detectImage(stream) {
    const detector = new ImageDetector();

    detector.onprediction = function(prediction) {
        resultDisplay.textContent = "Predicted Class: " + prediction.class;
        percentageDisplay.textContent = "Matching Percentage: " + prediction.percentage;

        // Display location if available and percentage is above 75%
        if (prediction.latitude !== null && prediction.longitude !== null && parseFloat(prediction.percentage) > 75) {
            locationDisplay.textContent = "Latitude: " + prediction.latitude + ", Longitude: " + prediction.longitude;
            displayLocationOnMap(prediction.latitude, prediction.longitude);
        } else {
            locationDisplay.textContent = ""; // Clear location display if not applicable
        }
    };

    await detector.startDetecting(stream);
}

class ImageDetector {
    constructor() {
        this.modelUrl = '/';
        this.videoElement = document.getElementById('camera');
        this.canvasElement = document.createElement('canvas');
        this.context = this.canvasElement.getContext('2d');
    }

    async startDetecting(stream) {
        this.videoElement.srcObject = stream;
        await this.loadModel();
        this.detectFrame();
    }

    async loadModel() {
        // Load model if needed
    }

    detectFrame() {
        const self = this;
        requestAnimationFrame(async function () {
            self.context.drawImage(self.videoElement, 0, 0, self.canvasElement.width, self.canvasElement.height);
            const imageDataUrl = self.canvasElement.toDataURL('image/jpeg');
            const prediction = await self.sendToServer(imageDataUrl);
            self.onprediction(prediction);
            self.detectFrame();
        });
    }

    async sendToServer(imageDataUrl) {
        const response = await fetch("/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ imgDataUrl: imageDataUrl })
        });

        if (!response.ok) {
            throw new Error('HTTP error! Status: ${response.status}');
        }

        return await response.json();
    }

    onprediction(prediction) {
        resultDisplay.textContent = "Predicted Class: " + prediction.class;
        percentageDisplay.textContent = "Matching Percentage: " + prediction.percentage;

        // Display location if available and percentage is above 75%
        if (prediction.latitude !== null && prediction.longitude !== null && parseFloat(prediction.percentage) > 75) {
            locationDisplay.textContent = "Latitude: " + prediction.latitude + ", Longitude: " + prediction.longitude;
            displayLocationOnMap(prediction.latitude, prediction.longitude);
        } else {
            locationDisplay.textContent = ""; // Clear location display if not applicable
        }
    }
}

// Function to display location on map
function displayLocationOnMap(latitude, longitude) {
    if (!map) {
        // Create map if it doesn't exist
        map = L.map('map').setView([latitude, longitude], 10);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    } else {
        // Update map center and marker position
        map.setView([latitude, longitude], 10);
        const marker = L.marker([latitude, longitude]).addTo(map);
        marker.setLatLng([latitude, longitude]);
    }
    // Show the map if it's hidden
    document.getElementById('map').style.display = 'block';
}