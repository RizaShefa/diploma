(function ($) {
    "use strict";

    // Spinner
    var spinner = function () {
        setTimeout(function () {
            if ($('#spinner').length > 0) {
                $('#spinner').removeClass('show');
            }
        }, 1);
    };
    spinner(0);
    
    
    // Initiate the wowjs
    new WOW().init();


    // Sticky Navbar
    $(window).scroll(function () {
        if ($(this).scrollTop() > 45) {
            $('.navbar').addClass('sticky-top shadow-sm');
        } else {
            $('.navbar').removeClass('sticky-top shadow-sm');
        }
    });


    // Hero Header carousel
    $(".header-carousel").owlCarousel({
        animateOut: 'slideOutDown',
        items: 1,
        autoplay: true,
        smartSpeed: 1000,
        dots: false,
        loop: true,
        nav : true,
        navText : [
            '<i class="bi bi-arrow-left"></i>',
            '<i class="bi bi-arrow-right"></i>'
        ],
    });


    // International carousel
    $(".testimonial-carousel").owlCarousel({
        autoplay: true,
        items: 1,
        smartSpeed: 1500,
        dots: true,
        loop: true,
        margin: 25,
        nav : true,
        navText : [
            '<i class="bi bi-arrow-left"></i>',
            '<i class="bi bi-arrow-right"></i>'
        ]
    });


    // Modal Video
    $(document).ready(function () {
        var $videoSrc;
        $('.btn-play').click(function () {
            $videoSrc = $(this).data("src");
        });
        console.log($videoSrc);

        $('#videoModal').on('shown.bs.modal', function (e) {
            $("#video").attr('src', $videoSrc + "?autoplay=1&amp;modestbranding=1&amp;showinfo=0");
        })

        $('#videoModal').on('hide.bs.modal', function (e) {
            $("#video").attr('src', $videoSrc);
        })
    });


    // testimonial carousel
    $(".testimonial-carousel").owlCarousel({
        autoplay: true,
        smartSpeed: 1000,
        center: true,
        dots: true,
        loop: true,
        margin: 25,
        nav : true,
        navText : [
            '<i class="bi bi-arrow-left"></i>',
            '<i class="bi bi-arrow-right"></i>'
        ],
        responsiveClass: true,
        responsive: {
            0:{
                items:1
            },
            576:{
                items:1
            },
            768:{
                items:1
            },
            992:{
                items:1
            },
            1200:{
                items:1
            }
        }
    });

    
    

    document.addEventListener("DOMContentLoaded", function () {
        document.getElementById("contactForm").addEventListener("submit", function (e) {
            e.preventDefault(); // prevent form from reloading the page

            // Show success message
            const msg = document.getElementById("Send");
            msg.style.display = "block";
            msg.textContent = "✅ Message sent successfully!";

            // Optional: Clear form
            this.reset();

            // Hide message after 5 seconds
            setTimeout(() => {
                msg.style.display = "none";
            }, 5000);
        });
    });

  

    document.addEventListener("DOMContentLoaded", function () {
        document.getElementById("appointmentForm").addEventListener("submit", function (e) {
            e.preventDefault(); // prevent form from reloading the page

            // Show success message
            const msg = document.getElementById("submit");
            msg.style.display = "block";
            msg.textContent = "✅ Your appointment is booked.Wait for our call for more details.";

            // Optional: Clear form
            this.reset();

            // Hide message after 5 seconds
            setTimeout(() => {
                msg.style.display = "none";
            }, 5000);
        });
    });

    $(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr( 'src', e.target.result );
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });
    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });

});

$(document).ready(function () {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#btn-treatment').hide();
    $('#treatment-details').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr('src', e.target.result);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('').hide();
        $('#btn-treatment').hide();
        $('#treatment-details').hide();
        readURL(this);
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600).text('Result: ' + data);
                $('#btn-treatment').show();

                // Save result to check later
                $('#btn-treatment').data('result', data);
            }
        });
    });

    // Show treatment based on prediction
    $('#btn-treatment').click(function () {
        const result = $(this).data('result');
        $('#treatment-details').show();

        if (result.includes("Yes")) {
            $('#treatment-details').html(`
                <h4 class="text-deep-danger">Treatment Options for Breast Cancer</h4>
                <ul>
                    <li class="text-dark-black"><strong>Surgery:</strong> Lumpectomy or Mastectomy</li>
                    <li class="text-dark-black"><strong>Radiation Therapy</strong></li>
                    <li class="text-dark-black"><strong>Chemotherapy</strong></li>
                    <li class="text-dark-black"><strong>Hormone Therapy</strong> (for hormone-positive cancers)</li>
                    <li class="text-dark-black"><strong>Targeted Therapy</strong> (e.g. HER2+)</li>
                    <li class="text-dark-black"><strong>Immunotherapy</strong> (if eligible)</li>
                </ul>
                <p class="text-dark-black">Please consult an oncologist immediately for a personalized plan.</p>
            `);
        } else {
            $('#treatment-details').html(`
                <h4 class="text-strong-green">Great News!</h4>
                <p class="text-dark-black">No signs of breast cancer detected. To stay healthy:</p>
                <ul class="text-dark-black">
                    <li class="text-dark-black">Continue regular screenings and checkups</li>
                    <li class="text-dark-black">Do monthly self-exams</li>
                    <li class="text-dark-black">Maintain a healthy lifestyle and diet</li>
                    <li class="text-dark-black">Know your family history</li>
                </ul>
            `);
        }
    });
});


})(jQuery);

