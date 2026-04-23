// Sidebar Toggle Functionality
document.addEventListener('DOMContentLoaded', function () {
    const sidebar = document.querySelector('.sidebar');
    const menuToggle = document.getElementById('menuToggle');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const navLinks = document.querySelectorAll('.nav-link');

    // Mobile menu toggle
    if (menuToggle) {
        menuToggle.addEventListener('click', function () {
            sidebar.classList.toggle('active');
        });
    }

    // Sidebar toggle button
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function () {
            sidebar.classList.toggle('active');
        });
    }

    // Close sidebar when a link is clicked on mobile
    navLinks.forEach(link => {
        link.addEventListener('click', function () {
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            // Add active class to clicked link
            this.classList.add('active');

            // Close sidebar on mobile
            if (window.innerWidth < 768) {
                sidebar.classList.remove('active');
            }
        });
    });

    // Handle window resize
    window.addEventListener('resize', function () {
        if (window.innerWidth >= 768) {
            sidebar.classList.remove('active');
        }
    });

    // Set active link based on current page
    const currentPage = window.location.pathname.split('/').pop() || 'overview';
    navLinks.forEach(link => {
        const href = link.getAttribute('href').split('/').pop();
        if (href === currentPage || (currentPage === '' && href === 'overview')) {
            link.classList.add('active');
        }
    });

    // Logout functionality
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function (e) {
            e.preventDefault();
            if (confirm('هل أنت متأكد من رغبتك في تسجيل الخروج؟')) {
                // Redirect to login page
                window.location.href = './index.html';
            }
        });
    }
});

// Date and Time Display
let CurrentDate = document.getElementById('current-datetime');

function updateDateTime() {
    let now = new Date();
    let options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' };
    CurrentDate.textContent = now.toLocaleDateString('ar-EG', options);
}

setInterval(updateDateTime, 1000);
updateDateTime();