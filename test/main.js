const video = document.getElementById("video");

let btnStartCamera = document.getElementById("startCamera");
btnStartCamera.addEventListener("click", startCamera);

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            // video.style.display = "block";
        })
        .catch(error => {
            alert("Unable to access the camera. Please allow camera access and try again.");
            console.log(error);
        });
}
