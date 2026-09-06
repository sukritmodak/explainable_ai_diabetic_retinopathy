
// ============================================================
// AI DIABETIC RETINOPATHY WEBSITE
// Frontend JavaScript
// ============================================================

// IMPORTANT:
// This URL will be changed to the public FastAPI URL
// when we deploy the backend.

const API_URL = window.location.origin;


// ============================================================
// ELEMENTS
// ============================================================

const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const selectedFile = document.getElementById("selectedFile");
const analyzeButton = document.getElementById("analyzeButton");

const loading = document.getElementById("loading");
const results = document.getElementById("results");

const prediction = document.getElementById("prediction");
const probability = document.getElementById("probability");
const resultBadge = document.getElementById("resultBadge");

const originalImage = document.getElementById("originalImage");
const heatmapImage = document.getElementById("heatmapImage");
const overlayImage = document.getElementById("overlayImage");


// ============================================================
// SELECTED FILE
// ============================================================

let selectedImage = null;


// ============================================================
// IMAGE SELECTION
// ============================================================

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    selectedImage = file;

    selectedFile.textContent =
        "Selected: " + file.name;

    selectedFile.classList.remove("hidden");

    analyzeButton.disabled = false;

    // Hide previous results
    results.classList.add("hidden");
});


// ============================================================
// DRAG & DROP
// ============================================================

uploadArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadArea.style.borderColor =
            "#176b87";
    }
);


uploadArea.addEventListener(
    "dragleave",
    function () {

        uploadArea.style.borderColor =
            "";
    }
);


uploadArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        uploadArea.style.borderColor =
            "";

        const file =
            event.dataTransfer.files[0];

        if (!file) {
            return;
        }

        if (!file.type.startsWith("image/")) {

            alert(
                "Please select a retinal image."
            );

            return;
        }

        selectedImage = file;

        selectedFile.textContent =
            "Selected: " + file.name;

        selectedFile.classList.remove(
            "hidden"
        );

        analyzeButton.disabled = false;

        results.classList.add(
            "hidden"
        );
    }
);


// ============================================================
// ANALYZE IMAGE
// ============================================================

analyzeButton.addEventListener(
    "click",
    async function () {

        if (!selectedImage) {

            alert(
                "Please select a fundus image first."
            );

            return;
        }


        // ----------------------------------------------------
        // UI — START LOADING
        // ----------------------------------------------------

        analyzeButton.disabled = true;

        loading.classList.remove(
            "hidden"
        );

        results.classList.add(
            "hidden"
        );


        // ----------------------------------------------------
        // CREATE FORM DATA
        // ----------------------------------------------------

        const formData =
            new FormData();

        formData.append(
            "file",
            selectedImage
        );


        try {

            // ------------------------------------------------
            // SEND IMAGE TO FASTAPI
            // ------------------------------------------------

            const response =
                await fetch(
                    API_URL + "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            // ------------------------------------------------
            // HANDLE SERVER ERROR
            // ------------------------------------------------

            if (!response.ok) {

                let errorMessage =
                    "Server error.";

                try {

                    const errorData =
                        await response.json();

                    if (errorData.detail) {
                        errorMessage =
                            errorData.detail;
                    }

                } catch (e) {
                    // Ignore JSON parsing error
                }

                throw new Error(
                    errorMessage
                );
            }


            // ------------------------------------------------
            // GET RESULT
            // ------------------------------------------------

            const data =
                await response.json();


            // ------------------------------------------------
            // DISPLAY PREDICTION
            // ------------------------------------------------

            prediction.textContent =
                data.prediction;

            probability.textContent =
                data.probability_percent +
                "%";


            // ------------------------------------------------
            // RESULT BADGE
            // ------------------------------------------------

            resultBadge.textContent =
                data.prediction;


            // ------------------------------------------------
            // DISPLAY IMAGES
            // ------------------------------------------------

            originalImage.src =
                "data:image/png;base64," +
                data.original_image;

            heatmapImage.src =
                "data:image/png;base64," +
                data.gradcam_heatmap;

            overlayImage.src =
                "data:image/png;base64," +
                data.gradcam_overlay;


            // ------------------------------------------------
            // SHOW RESULTS
            // ------------------------------------------------

            results.classList.remove(
                "hidden"
            );


            // Scroll smoothly to results
            setTimeout(function () {

                results.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

            }, 100);


        } catch (error) {

            console.error(
                "Prediction error:",
                error
            );

            alert(
                "Unable to analyze the image.\n\n" +
                error.message +
                "\n\nMake sure the FastAPI backend is running."
            );

        } finally {

            loading.classList.add(
                "hidden"
            );

            analyzeButton.disabled = false;
        }

    }
);
