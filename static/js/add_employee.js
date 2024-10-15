document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('add-employee-video');
    const startButton = document.getElementById('start-camera');
    const snapButton = document.getElementById('snap');
    const finishButton = document.getElementById('finish-capture');
    const submitButton = document.getElementById('submit-button');
    const employeeForm = document.getElementById('employee-form');
    const statusDiv = document.getElementById('add-employee-status');
    const canvas = document.getElementById('add-employee-canvas');
    const context = canvas.getContext('2d');

    let photoDataList = [];
    let photosTaken = false;
    let employeeAdded = false;
    let recognitionInterval;


    // Set initial state of the statusDiv
    statusDiv.style.display = 'block'; // Make sure it's visible by default
    statusDiv.textContent = `Photos taken: ${photoDataList.length}`; // Set initial count

    // Functionality for add_employee.html
    if (employeeForm) {
        // Start camera button click event
        startButton.addEventListener('click', () => {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
        
                    // Set canvas width and height to match the video stream dimensions
                    video.onloadedmetadata = () => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                    };
        
                    snapButton.disabled = false;
                    finishButton.disabled = true;
                    submitButton.disabled = true;
        
                    // Start recognition if needed
                    recognitionInterval = setInterval(captureAndRecognize, 1000);
                })
                .catch(error => console.error('Error accessing camera:', error));
        });

        // Snap button functionality
        snapButton.addEventListener('click', () => {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg');
            photoDataList.push(dataURL);
            photosTaken = true;
            console.log('Photo captured');

            // Show the statusDiv only when at least one photo is taken
            if (photoDataList.length === 1) {
                statusDiv.style.display = 'block';// Make statusDiv visible
            }

            finishButton.disabled = false;
            statusDiv.textContent = `Photos taken: ${photoDataList.length}`;
        });

        // Finish button functionality
        finishButton.addEventListener('click', () => {
            if (photosTaken) {
                alert(`You have captured ${photoDataList.length} photo(s).`);
                submitButton.disabled = false;
                finishButton.disabled = true;
                snapButton.disabled = true;
                console.log('Finished capturing photos.');
                clearInterval(recognitionInterval);
            } else {
                alert('You must take at least 1 photo.');
            }
        });

        // Employee form submission
        employeeForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            if (employeeAdded) {
                alert('Employee already added.');
                return;
            }

            const formData = new FormData(employeeForm);
            if (photoDataList.length > 0) {
                formData.append('photos', JSON.stringify(photoDataList));
            }

            try {
                const response = await fetch('/add_employee', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    console.log(result.message);
                    employeeAdded = true;
                    window.location.href = '/';
                } else {
                    console.error('Error adding employee:', response.statusText);
                    alert('Error adding employee. Please try again.');
                }
            } catch (error) {
                console.error('Error submitting the form:', error);
                alert('An unexpected error occurred. Please try again.');
            }
        });
    }

    // Functionality for recognize_employee.html
    const startButtonRecognize = document.getElementById('start-camera'); // This ID may be the same as the other page; consider changing it for clarity
    const stopButton = document.getElementById('stop-recognition'); // Ensure this button exists in recognize_employee.html

    if (startButtonRecognize) {
        startButtonRecognize.addEventListener('click', () => {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                    //startButtonRecognize.style.display = 'none'; // Hide start button
                    stopButton.style.display = 'inline'; // Show stop button

                    // Start recognition logic here if needed
                    recognitionInterval = setInterval(captureAndRecognize, 1000); // Start recognition loop
                })
                .catch(error => console.error('Error accessing camera:', error));
        });
    }

    // Stop recognition button functionality for Recognize Employee page
    if (stopButton) {
        stopButton.addEventListener('click', () => {
            clearInterval(recognitionInterval); // Stop the recognition loop
            video.srcObject.getTracks().forEach(track => track.stop()); // Stop the video stream
            video.style.display = 'none'; // Hide video element
            startButtonRecognize.style.display = 'inline'; // Show start button again
            stopButton.style.display = 'none'; // Hide stop button
            console.log('Recognition stopped.');
        });
    }

    // Placeholder function for recognition logic
    function captureAndRecognize() {
        // Implement the capture and recognition logic here
    }
});
