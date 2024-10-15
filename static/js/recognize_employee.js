document.addEventListener('DOMContentLoaded', function() {
    // Recognition system
    let recognitionInterval;
    let video = document.getElementById('recognize-employee-video');
    let startButton = document.getElementById('start-camera');
    let stopButton = document.getElementById('stop-recognition');
    let statusDiv = document.getElementById('recognize-employee-status');
    let captureButton = document.getElementById('capture');
    let canvas = document.createElement('canvas');  // Using canvas for capture
    let context = canvas.getContext('2d');
    let loadingAnimation = document.getElementById('loading-animation');
    let loadingDots = document.getElementById('loading-dots');
    
    let isRecognizing = false;
    let firstRecognitionReceived = false;
    let loadingIntervalId = null;

    function startLoadingAnimation() {
        loadingAnimation.style.display = 'block';
        statusDiv.style.display = 'none'; // Hide status div while loading
        let dotsCount = 0;
        loadingIntervalId = setInterval(() => {
            dotsCount = (dotsCount + 1) % 4;
            loadingDots.textContent = '.'.repeat(dotsCount);
        }, 300); // Faster animation for better visibility
    }

    function stopLoadingAnimation() {
        if (loadingIntervalId) {
            clearInterval(loadingIntervalId);
            loadingIntervalId = null;
        }
        loadingAnimation.style.display = 'none';
        statusDiv.style.display = 'block'; // Show status div after loading
    }
    
    if (startButton) {
        startButton.addEventListener('click', function() {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                    document.getElementById('recognize-employee-video-container').style.display = 'block';
                    startButton.style.display = 'none';
                    stopButton.style.display = 'inline';
                    isRecognizing = true;
                    firstRecognitionReceived = false;
                    startLoadingAnimation(); // Start loading animation immediately
                    // Delay the start of recognition to allow camera to initialize
                    setTimeout(() => {
                        recognitionInterval = setInterval(captureAndRecognize, 1000);
                    }, 1000);
                })
                .catch(error => {
                    console.error('Error accessing webcam:', error);
                    statusDiv.textContent = 'Could not access the webcam.';
                });
        });
    }

    if (stopButton) {
        stopButton.addEventListener('click', function() {
            let stream = video.srcObject;
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;

            clearInterval(recognitionInterval);
            isRecognizing = false;
            stopLoadingAnimation();
            document.getElementById('recognize-employee-video-container').style.display = 'none';
            startButton.style.display = 'inline';
            stopButton.style.display = 'none';
            statusDiv.textContent = '';
        });
    }

    if (captureButton) {
        captureButton.addEventListener('click', captureAndRecognize);
    }
    
    function captureAndRecognize() {
        canvas.width = 224;
        canvas.height = 224;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
            if (blob) {
                console.log('Blob created for recognition image:', blob.size);
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');

                fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (!firstRecognitionReceived) {
                        firstRecognitionReceived = true;
                        stopLoadingAnimation();
                    }
                    if (data.recognized) {
                        statusDiv.innerHTML = data.name;
                    } else {
                        statusDiv.innerHTML = 'Unrecognized - Access Denied';
                    }
                })
                .catch(error => {
                    console.error('Error during recognition:', error);
                    if (!firstRecognitionReceived) {
                        statusDiv.textContent = 'Error during recognition. Please try again.';
                        stopLoadingAnimation();
                    }
                });
            } else {
                console.error('No blob created for recognition image.');
                if (!firstRecognitionReceived) {
                    statusDiv.textContent = 'Failed to capture image. Please try again.';
                    stopLoadingAnimation();
                }
            }
        }, 'image/jpeg');
    }
});