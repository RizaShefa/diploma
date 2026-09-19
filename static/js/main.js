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

    // ------------------------------------------------------------------
    // Prediction flow
    //
    // Replaces two duplicated $(document).ready blocks that each bound a click
    // handler to #btn-predict, causing every click to fire two POST requests.
    // Now a single handler calls /api/predict and renders the structured
    // report: classification, confidence, Grad-CAM overlay and model details.
    // ------------------------------------------------------------------
    $(document).ready(function () {

        var $result = $('#result');
        var $panel = $('#prediction-panel');

        function resetOutput() {
            $result.text('').hide();
            $panel.hide().empty();
            $('#btn-treatment').hide();
            $('#treatment-details').hide().empty();
        }

        $('.image-section').hide();
        $('.loader').hide();
        resetOutput();

        function readURL(input) {
            if (input.files && input.files[0]) {
                var reader = new FileReader();
                reader.onload = function (e) {
                    $('#imagePreview').attr('src', e.target.result);
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        $('#imageUpload').change(function () {
            $('.image-section').show();
            $('#btn-predict').show();
            resetOutput();
            readURL(this);
        });

        function escapeHtml(value) {
            return $('<div>').text(value === undefined || value === null ? '' : value).html();
        }

        function pct(value) {
            return (value * 100).toFixed(1) + '%';
        }

        function bandClass(band) {
            if (band === 'high') { return 'success'; }
            if (band === 'moderate') { return 'secondary'; }
            return 'warning';
        }

        function renderReport(report) {
            var c = report.classification;
            var conf = report.confidence;
            var model = report.model;
            var explanation = report.explanation;
            var isPositive = c.label === c.positive_class;

            var html = '';

            // --- Classification -------------------------------------------
            html += '<div class="card mb-3"><div class="card-body">';
            html += '<h5 class="card-title mb-3">Model output</h5>';
            html += '<p class="fs-5 mb-2">' + escapeHtml(c.statement) + '</p>';
            html += '<span class="badge bg-' + (isPositive ? 'danger' : 'success') +
                    ' fs-6 me-2">' + escapeHtml(c.label) + '</span>';
            html += '<span class="text-muted small">' + escapeHtml(c.threshold_note) + '</span>';

            html += '<div class="mt-3">';
            Object.keys(c.probabilities).forEach(function (name) {
                var value = c.probabilities[name];
                html += '<div class="mb-2"><div class="d-flex justify-content-between small">' +
                        '<span>' + escapeHtml(name) + '</span><span>' + pct(value) + '</span></div>' +
                        '<div class="progress" style="height:8px;">' +
                        '<div class="progress-bar bg-' + (name === c.positive_class ? 'danger' : 'success') +
                        '" style="width:' + (value * 100).toFixed(1) + '%"></div></div></div>';
            });
            html += '</div>';

            // --- Confidence ------------------------------------------------
            html += '<hr><h6 class="mb-2">Confidence</h6>';
            html += '<span class="badge bg-' + bandClass(conf.band) + ' me-2">' +
                    escapeHtml(conf.band) + '</span>';
            html += '<span class="mono">' + conf.value.toFixed(3) + '</span>';
            html += '<p class="small text-muted mt-2 mb-0">' + escapeHtml(conf.note) + '</p>';
            html += '</div></div>';

            // --- Explanation ------------------------------------------------
            if (explanation && explanation.overlay_png) {
                html += '<div class="card mb-3"><div class="card-body">';
                html += '<h5 class="card-title mb-1">Visual explanation</h5>';
                html += '<p class="small text-muted mb-3">' + escapeHtml(explanation.method) +
                        ' &mdash; ' + escapeHtml(explanation.reference) + '</p>';
                html += '<img src="' + explanation.overlay_png +
                        '" alt="Grad-CAM overlay" class="img-fluid rounded border" style="max-width:340px;">';
                html += '<div class="alert alert-warning small mt-3 mb-0">' +
                        '<strong>This is not a lesion segmentation.</strong> ' +
                        escapeHtml(explanation.disclaimer) + '</div>';
                if (explanation.resolution_note) {
                    html += '<p class="small text-muted mt-2 mb-0">' +
                            escapeHtml(explanation.resolution_note) + '</p>';
                }
                if (explanation.is_degenerate) {
                    html += '<p class="small text-danger mt-2 mb-0">The heatmap is empty for this ' +
                            'image: no region increased the predicted class score.</p>';
                }
                html += '</div></div>';
            } else if (explanation && explanation.available === false) {
                html += '<div class="alert alert-secondary small">' +
                        escapeHtml(explanation.message) + '</div>';
            }

            // --- Model ------------------------------------------------------
            html += '<div class="card mb-3"><div class="card-body">';
            html += '<h5 class="card-title mb-3">Model information</h5>';
            html += '<table class="table table-sm mb-2"><tbody>';
            html += '<tr><th scope="row" class="w-50">Name</th><td>' + escapeHtml(model.name) +
                    (model.legacy ? ' <span class="badge bg-warning text-dark">legacy</span>' : '') + '</td></tr>';
            html += '<tr><th scope="row">Architecture</th><td>' + escapeHtml(model.architecture) + '</td></tr>';
            html += '<tr><th scope="row">Parameters</th><td>' +
                    (model.parameters ? model.parameters.toLocaleString() : '&mdash;') + '</td></tr>';
            html += '<tr><th scope="row">Preprocessing</th><td>' +
                    escapeHtml(model.preprocessing_profile) + '</td></tr>';
            html += '</tbody></table>';

            var perf = model.validated_performance || {};
            if (perf.available) {
                html += '<h6 class="mt-3">Validated test performance</h6><ul class="small mb-0">';
                ['sensitivity', 'specificity', 'roc_auc'].forEach(function (key) {
                    if (perf[key]) {
                        html += '<li>' + key.replace('_', ' ') + ': <span class="mono">' +
                                escapeHtml(perf[key]) + '</span></li>';
                    }
                });
                if (perf.n_test) { html += '<li>test set size: ' + perf.n_test + '</li>'; }
                html += '</ul>';
            } else {
                html += '<div class="alert alert-secondary small mb-0">' +
                        escapeHtml(perf.message || 'No validated test metrics recorded.') + '</div>';
            }
            html += '</div></div>';

            // --- Limitations + disclaimer ------------------------------------
            html += '<div class="card mb-3 border-warning"><div class="card-body">';
            html += '<h5 class="card-title mb-3">Limitations</h5><ul class="small">';
            (report.limitations || []).forEach(function (item) {
                html += '<li>' + escapeHtml(item) + '</li>';
            });
            html += '</ul>';
            html += '<div class="alert alert-danger small mb-0"><strong>Disclaimer.</strong> ' +
                    escapeHtml(report.disclaimer) + '</div>';
            html += '</div></div>';

            if (report.history_id) {
                html += '<p class="small"><a href="/api/report/' + report.history_id +
                        '.txt" target="_blank">Download this report as text</a> &middot; ' +
                        '<a href="/history">View prediction history</a></p>';
            }

            $panel.html(html).fadeIn(300);
        }

        function renderError(message) {
            $panel.html('<div class="alert alert-danger"><strong>Could not analyse this image.</strong>' +
                        '<br>' + escapeHtml(message) + '</div>').fadeIn(200);
        }

        $('#btn-predict').off('click').on('click', function () {
            var fileInput = document.getElementById('imageUpload');
            if (!fileInput || !fileInput.files.length) {
                renderError('Please choose an image first.');
                return;
            }

            var formData = new FormData($('#upload-file')[0]);
            var $button = $(this);
            $button.hide();
            $('.loader').show();
            resetOutput();

            $.ajax({
                type: 'POST',
                url: '/api/predict',
                data: formData,
                contentType: false,
                cache: false,
                processData: false,
                dataType: 'json',
                success: function (report) {
                    $('.loader').hide();
                    $button.show();
                    renderReport(report);
                    $('#btn-treatment')
                        .data('label', report.classification.label)
                        .data('positive', report.classification.positive_class)
                        .show();
                },
                error: function (xhr) {
                    $('.loader').hide();
                    $button.show();
                    var message = 'The server could not process the request.';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        message = xhr.responseJSON.error;
                    }
                    renderError(message);
                }
            });
        });

        // General educational information about typical follow-up pathways.
        // Keyed on the structured label rather than substring-matching the old
        // "Yes"/"No" response string.
        $('#btn-treatment').off('click').on('click', function () {
            var label = $(this).data('label');
            var positive = $(this).data('positive');
            var $details = $('#treatment-details');

            var preamble = '<div class="alert alert-warning small"><strong>General information only.</strong> ' +
                'The following describes how findings of this type are typically investigated in ' +
                'clinical practice. It is not a recommendation, not a care plan, and is not based ' +
                'on your individual circumstances. Only a qualified clinician can advise you.</div>';

            if (label === positive) {
                $details.html(preamble +
                    '<h5 class="text-dark-black">Typical follow-up for a suspicious ultrasound finding</h5>' +
                    '<ul class="text-dark-black">' +
                    '<li>Review by a radiologist and correlation with clinical examination</li>' +
                    '<li>Additional imaging (e.g. diagnostic mammography, targeted ultrasound)</li>' +
                    '<li>Tissue sampling (core needle biopsy) if the finding remains suspicious</li>' +
                    '<li>Histopathology determines whether a lesion is malignant &mdash; imaging alone does not</li>' +
                    '</ul>' +
                    '<p class="text-dark-black small">Any management decision follows biopsy and specialist review, ' +
                    'never an imaging classifier.</p>');
            } else {
                $details.html(preamble +
                    '<h5 class="text-dark-black">Typical follow-up for a probably benign finding</h5>' +
                    '<ul class="text-dark-black">' +
                    '<li>Routine screening according to local guidelines and personal risk</li>' +
                    '<li>Short-interval follow-up imaging where a radiologist advises it</li>' +
                    '<li>Prompt clinical review of any new or changing symptom</li>' +
                    '</ul>' +
                    '<p class="text-dark-black small">A benign classification from this prototype does not ' +
                    'exclude disease and must not delay clinical assessment.</p>');
            }
            $details.show();
        });
    });

})(jQuery);
