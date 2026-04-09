const video = document.getElementById("video");

let btnCamera = document.getElementById("btnCamera");

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(error => {
            alert("Unable to access the camera. Please allow camera access and try again.");
            console.log(error);
        });
}

function stopCamera(){
    if (video.srcObject){
        let stream = video.srcObject;
        let tracks = stream.getTracks();
        tracks.forEach(track => track.stop())
        video.srcObject = null;
    }
}

function toggleCamera(){
    if (video.srcObject){
        stopCamera();
        btnCamera.textContent = "Start Camera";
    }
    else{
        startCamera();
        btnCamera.textContent = "Stop Camera";
    }
}

btnCamera.addEventListener("click", toggleCamera);
