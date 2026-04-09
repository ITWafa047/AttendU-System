
function startCamera(webcam) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            webcam.srcObject = stream;
        })
        .catch(error => {
            alert("Unable to access the camera. Please allow camera access and try again.");
            console.log(error);
        });
}

function stopCamera(webcam) {
    if (webcam.srcObject) {
        let stream = webcam.srcObject;
        let tracks = stream.getTracks();
        tracks.forEach(track => track.stop())
        webcam.srcObject = null;
    }
}

function toggleCamera(btn, webcam) {
    
    if (webcam.srcObject) {
        stopCamera(webcam);
        btn.textContent = "تشغيل الكاميرا";
        btn.classList.replace("btn-danger", "btn-success")
    }
    else {
        startCamera(webcam);
        btn.textContent = "إغلاق الكاميرا";
        btn.classList.replace("btn-success", "btn-danger")
    }
}