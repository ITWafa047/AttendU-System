let contentWrapper = document.getElementById("contentWrapper");

function loadSection(tab) {
    // Clear existing content
    contentWrapper.innerHTML = "";
    // Load new content based on section
    switch (tab) {

        case "overview":
            contentWrapper.innerHTML = "<h2>نظرة عامة</h2><p>محتوى نظرة عامة...</p>";
            break;

        case "live-session":
            contentWrapper.innerHTML = "<h2>الجلسة المباشرة</h2><p>محتوى الجلسة المباشرة...</p>";
            break;

        case "schedules":
            contentWrapper.innerHTML = "<h2>الجداول الزمنية</h2><p>محتوى الجداول الزمنية...</p>";
            break;

        case "sessions":
            contentWrapper.innerHTML = "<h2>الجلسات</h2><p>محتوى الجلسات...</p>";
            break;

        case "students":
            contentWrapper.innerHTML = "<h2>الطلاب</h2><p>محتوى الطلاب...</p>";
            break;
        
        case "reports":
            contentWrapper.innerHTML = "<h2>التقارير</h2><p>محتوى التقارير...</p>";
            break;
        
        case "policy":
            contentWrapper.innerHTML = "<h2>السياسة</h2><p>محتوى السياسة...</p>";
            break;

        case "warnings":
            contentWrapper.innerHTML = "<h2>التحذيرات</h2><p>محتوى التحذيرات...</p>";
            break;
        
        case "sync":
            contentWrapper.innerHTML = "<h2>المزامنة</h2><p>محتوى المزامنة...</p>";
            break;

        default:
            contentWrapper.innerHTML = "<h2>الصفحة غير موجودة</h2><p>الرجاء اختيار قسم من القائمة.</p>";
            break;
    }
}