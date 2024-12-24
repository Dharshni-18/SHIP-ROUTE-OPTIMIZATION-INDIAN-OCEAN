// Toggle between Login and Sign Up forms
document.getElementById('signPage').onclick = function () {
    document.querySelector('.login-box').style.display = 'none';
    document.querySelector('.signUp-box').style.display = 'block';
};

document.getElementById('loginPage').onclick = function () {
    document.querySelector('.signUp-box').style.display = 'none';
    document.querySelector('.login-box').style.display = 'block';
};

// Validate login form
document.getElementById('loginForm').onsubmit = function () {
    const username = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    if (username === 'DSS' && password === '12345') {
        return true;
    } else {
        alert('Invalid credentials. Please try again.');
        return false;
    }
};
