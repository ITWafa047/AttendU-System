// DOM Elements
let contentWrapper = document.getElementById("contentWrapper");
let roleSelect = document.getElementById("roleSelect");
const instructorSections = ["liveSession", "sessions", "sync"];
const adminSections = ['overview', "schedules", "students", "reports", "policy", "warnings"];

function loadSection(tab) {
    // Clear existing content
    contentWrapper.innerHTML = "";

    fetch(`./pages/${tab}.html`)
        .then(response => response.text())
        .then(html => {
            contentWrapper.innerHTML = html;
            // live session
            if (tab === "liveSession") {
                const videoCam = document.getElementById("webcam");
                let btnCamera = document.getElementById("btnCamera");
                if (btnCamera) {
                    const freahBtn = btnCamera.cloneNode(true);
                    btnCamera.replaceWith(freahBtn);
                    freahBtn.addEventListener('click', () => toggleCamera(freahBtn, videoCam));
                }
            }
        })
        .catch(() => {
            contentWrapper.innerHTML = "<h2>تعذر تحميل الصفحة</h2><p>الرجاء المحاولة مرة أخرى.</p>";
        });
}

function updateNavForRole(role) {
    let allowedSections = [];

    if (role === "admin") {
        allowedSections = adminSections;
    } else if (role === "instructor") {
        allowedSections = instructorSections;
    }

    let tabs = document.querySelectorAll("[data-tab]");
    const navLinks = document.querySelectorAll('.nav-link');

    tabs.forEach(tab => {
        const tabName = tab.getAttribute("data-tab");
        if (allowedSections.includes(tabName)) {
            tab.classList.remove("disabled");
            tab.style.pointerEvents = "auto";
            tab.style.opacity = "1";
        } else {
            tab.classList.add("disabled");
            tab.style.pointerEvents = "none";
            navLinks.forEach(link => link.classList.remove("active"));
            tab.style.opacity = "0.5";
        }
    });

    navLinks.forEach(link => link.classList.remove("active"));
}

// Initialize with default role
updateNavForRole(roleSelect.value);

roleSelect.addEventListener("change", function () {
    const selectedRole = roleSelect.value;
    updateNavForRole(selectedRole);
});

