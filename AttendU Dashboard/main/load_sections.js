// DOM Elements
let contentWrapper = document.getElementById("contentWrapper");
let roleSelect = document.getElementById("roleSelect");

const instructorSections = ["liveSession", "sessions"];
const adminSections = ['overview',"schedules", "students", "reports", "policy", "warnings", "sync"];

function loadSection(tab) {
    // Clear existing content
    contentWrapper.innerHTML = "";
    fetch(`./pages/${tab}.html`)
        .then(response => response.text())
        .then(data => {
            contentWrapper.innerHTML = data;
        });

    // Load new content based on section
    switch (tab) {

        case "overview":
            contentWrapper.innerHTML = './pages/overview.html';
            break;

        case "liveSession":
            contentWrapper.innerHTML = './pages/liveSession.html';
            break;

        case "schedules":
            contentWrapper.innerHTML = './pages/schedules.html';
            break;

        case "sessions":
            contentWrapper.innerHTML = './pages/sessions.html';
            break;

        case "students":
            contentWrapper.innerHTML = './pages/students.html';
            break;
        
        case "reports":
            contentWrapper.innerHTML = './pages/reports.html';
            break;
        
        case "policy":
            contentWrapper.innerHTML = './pages/policy.html';
            break;

        case "warnings":
            contentWrapper.innerHTML = './pages/warnings.html';
            break;
        
        case "sync":
            contentWrapper.innerHTML = './pages/sync.html';
            break;

        default:
            contentWrapper.innerHTML = "<h2>الصفحة غير موجودة</h2><p>الرجاء اختيار قسم من القائمة.</p>";
            break;
    }
}

function updateNavForRole(role){
    let allowedSections = [];

    if (role === "admin"){
        allowedSections = adminSections;
    }else if (role === "instructor"){
        allowedSections = instructorSections;
    }

    let tabs = document.querySelectorAll("[data-tab]");

    tabs.forEach(tab => {
        const tabName = tab.getAttribute("data-tab");

        if (allowedSections.includes(tabName)){
            tab.classList.remove("disabled");
            tab.style.pointerEvents = "auto";
            tab.style.opacity = "1";
        }else{
            tab.classList.add("disabled");
            tab.style.pointerEvents = "none";
            tab.style.opacity = "0.5";
        }
    });
}

// Initialize with default role
updateNavForRole(roleSelect.value);

roleSelect.addEventListener("change", function(){
    const selectedRole = roleSelect.value;
    updateNavForRole(selectedRole);
});

