let usernameInput = document.getElementById('username');
let passwordInput = document.getElementById('password');

let logindata = {
    "username": "AttendU78",
    "password": "b0ESC0OG4ikm"
}

function validateLogin(username, password) {
    if (username === logindata['username'] && password === logindata['password']) {
        return true;
    }
    return false;
}

document.getElementById('loginForm').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent form submission

    let username = usernameInput.value.trim();
    let password = passwordInput.value.trim();

    if (validateLogin(username, password)) {

        window.location.href = './dashboard.html';

    } else {
        alert('اسم المستخدم أو كلمة المرور غير صحيحة');
    }
});