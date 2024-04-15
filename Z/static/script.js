// static/script.js
$(document).ready(function(){
    $('#recordBtn').click(function(){
        console.log("Button clicked"); // Add this line
        $.ajax({
            type: 'POST',
            url: '/',
            success: function(response) {
                console.log("Response received:", response); // Add this line
                $('#result').text("Predicted Class: " + response.predicted_class);
            }
        });
    });
});
