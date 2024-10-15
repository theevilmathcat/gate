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
    
    
    if (startButton) {
        startButton.addEventListener('click', function() {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                    document.getElementById('recognize-employee-video-container').style.display = 'block';
                    startButton.style.display = 'none';
                    stopButton.style.display = 'inline';
                    isRecognizing = true; // Set flag to true
                    recognitionInterval = setInterval(captureAndRecognize, 1000); // Delay between recognition requests
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
            isRecognizing = false; // Set flag to false
            document.getElementById('recognize-employee-video-container').style.display = 'none';
            startButton.style.display = 'inline';
            stopButton.style.display = 'none';
        });
    }

    if (captureButton) {
        captureButton.addEventListener('click', function() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);
    
            canvas.toBlob(blob => {
                if (blob) {
                    console.log('Blob created for capture image:', blob.size); // Log blob size
                    const formData = new FormData();
                    formData.append('image', blob, 'capture.jpg');
    
                    fetch('/api/recognize', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.recognized) {
                            statusDiv.textContent = `${data.name} (Confidence: ${(data.confidence * 100).toFixed(2)}%)`;
                        } else {
                            statusDiv.textContent = 'Unrecognized - Access Denied';
                        }
                    })
                    .catch(error => {
                        console.error('Error during recognition:', error);
                        statusDiv.textContent = 'An error occurred during recognition.';
                    });
                } else {
                    console.error('No blob created for capture image.');
                    statusDiv.textContent = 'Failed to capture image.';
                }
            }, 'image/jpeg');
        });
    }
    
    function captureAndRecognize() {
        canvas.width = 224; // Use the size you want for recognition
        canvas.height = 224;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
            if (blob) {
                console.log('Blob created for recognition image:', blob.size); // Log blob size
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');

                fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.recognized) {
                        statusDiv.innerHTML = data.name;
                    } else {
                        statusDiv.innerHTML = 'Unrecognized - Access Denied';
                    }
                })
                .catch(error => {
                    console.error('Error during recognition:', error);
                });
            } else {
                console.error('No blob created for recognition image.');
            }
        }, 'image/jpeg');
    }
});