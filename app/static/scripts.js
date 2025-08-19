// Page navigation functionality
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });

    // Show the selected page
    document.getElementById(`${pageId}-page`).classList.add('active');

    // Update URL without refreshing
    history.pushState(null, null, `#${pageId}`);

    // Scroll to top
    window.scrollTo(0, 0);
}

// Initialize page based on URL hash
function initPage() {
    const hash = window.location.hash.substring(1);
    const validPages = ['home', 'evaluate', 'compare', 'visualize'];

    if (validPages.includes(hash)) {
        showPage(hash);
    } else {
        showPage('home');
    }
}

// Set up event listeners for range inputs
function setupRangeInputs() {
    // Evaluation page range input
    const sampleCountEval = document.getElementById('sample-count-eval');
    const sampleCountValueEval = document.getElementById('sample-count-value-eval');

    if (sampleCountEval && sampleCountValueEval) {
        sampleCountEval.addEventListener('input', function() {
            sampleCountValueEval.textContent = this.value;
        });
    }

    // Comparison page range input
    const sampleCountComp = document.getElementById('sample-count-comp');
    const sampleCountValueComp = document.getElementById('sample-count-value-comp');

    if (sampleCountComp && sampleCountValueComp) {
        sampleCountComp.addEventListener('input', function() {
            sampleCountValueComp.textContent = this.value;
        });
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initPage();
    setupRangeInputs();

    // Handle browser back/forward buttons
    window.addEventListener('popstate', initPage);
});