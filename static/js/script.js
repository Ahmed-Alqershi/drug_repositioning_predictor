// JavaScript for the Drug-Disease Relationship Predictor

document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('prediction-form');
    const resultSection = document.getElementById('result-section');
    const errorAlert = document.getElementById('error-alert');
    
    // Input elements
    const drugInput = document.getElementById('drug_input');
    const diseaseInput = document.getElementById('disease_input');
    const inputType = document.getElementById('input_type');
    
    // Selected info elements
    const selectedDrugInfo = document.getElementById('selected-drug-info');
    const selectedDrugName = document.getElementById('selected-drug-name');
    const selectedDrugCui = document.getElementById('selected-drug-cui');
    const selectedDiseaseInfo = document.getElementById('selected-disease-info');
    const selectedDiseaseName = document.getElementById('selected-disease-name');
    const selectedDiseaseCui = document.getElementById('selected-disease-cui');
    
    // Data storage for autocomplete
    let drugsData = [];
    let diseasesData = [];
    
    // Fetch drug data for autocomplete
    fetch('/api/drugs')
        .then(response => response.json())
        .then(data => {
            drugsData = data;
            setupDrugAutocomplete();
        })
        .catch(error => {
            console.error('Error fetching drug data:', error);
        });
    
    // Fetch disease data for autocomplete
    fetch('/api/diseases')
        .then(response => response.json())
        .then(data => {
            diseasesData = data;
            setupDiseaseAutocomplete();
        })
        .catch(error => {
            console.error('Error fetching disease data:', error);
        });
    
    // Set up autocomplete for drugs
    function setupDrugAutocomplete() {
        $(drugInput).autocomplete({
            source: function(request, response) {
                const searchTerm = request.term.toLowerCase();
                let filtered = [];
                
                // First try exact CUI matches (starts with 'C')
                if (searchTerm.startsWith('c')) {
                    filtered = drugsData.filter(item => 
                        item.cui.toLowerCase().includes(searchTerm)
                    ).slice(0, 10);
                }
                
                // Then add name matches
                const nameMatches = drugsData.filter(item => 
                    item.name.toLowerCase().includes(searchTerm) &&
                    !filtered.some(f => f.cui === item.cui)
                ).slice(0, 15);
                
                filtered = filtered.concat(nameMatches);
                
                // If still not many matches, try partial CUI matches that don't start with 'C'
                if (filtered.length < 5 && !searchTerm.startsWith('c')) {
                    const partialCuiMatches = drugsData.filter(item => 
                        item.cui.toLowerCase().includes(searchTerm) &&
                        !filtered.some(f => f.cui === item.cui)
                    ).slice(0, 5);
                    
                    filtered = filtered.concat(partialCuiMatches);
                }
                
                response(filtered);
            },
            minLength: 2,
            select: function(event, ui) {
                // Determine if this appears to be a CUI selection
                const isCuiSelection = ui.item.cui.toLowerCase().includes(drugInput.value.toLowerCase());
                
                // Set input type to help backend know how to process this
                if (isCuiSelection) {
                    inputType.value = 'cui';
                    drugInput.value = ui.item.cui;
                } else {
                    inputType.value = 'name';
                    drugInput.value = ui.item.name;
                }
                
                // Store the full information
                selectedDrugName.textContent = ui.item.name;
                selectedDrugCui.textContent = ui.item.cui;
                selectedDrugInfo.classList.remove('d-none');
                
                return false;
            }
        }).autocomplete("instance")._renderItem = function(ul, item) {
            return $("<li>")
                .append("<div class='autocomplete-item'>" +
                        "<span class='autocomplete-name'>" + item.name + "</span>" +
                        "<span class='autocomplete-cui'>" + item.cui + "</span>" +
                        "</div>")
                .appendTo(ul);
        };
    }
    
    // Set up autocomplete for diseases
    function setupDiseaseAutocomplete() {
        $(diseaseInput).autocomplete({
            source: function(request, response) {
                const searchTerm = request.term.toLowerCase();
                let filtered = [];
                
                // First try exact CUI matches (starts with 'C')
                if (searchTerm.startsWith('c')) {
                    filtered = diseasesData.filter(item => 
                        item.cui.toLowerCase().includes(searchTerm)
                    ).slice(0, 10);
                }
                
                // Then add name matches
                const nameMatches = diseasesData.filter(item => 
                    item.name.toLowerCase().includes(searchTerm) &&
                    !filtered.some(f => f.cui === item.cui)
                ).slice(0, 15);
                
                filtered = filtered.concat(nameMatches);
                
                // If still not many matches, try partial CUI matches that don't start with 'C'
                if (filtered.length < 5 && !searchTerm.startsWith('c')) {
                    const partialCuiMatches = diseasesData.filter(item => 
                        item.cui.toLowerCase().includes(searchTerm) &&
                        !filtered.some(f => f.cui === item.cui)
                    ).slice(0, 5);
                    
                    filtered = filtered.concat(partialCuiMatches);
                }
                
                response(filtered);
            },
            minLength: 2,
            select: function(event, ui) {
                // Determine if this appears to be a CUI selection
                const isCuiSelection = ui.item.cui.toLowerCase().includes(diseaseInput.value.toLowerCase());
                
                // Set input type to help backend know how to process this
                if (isCuiSelection) {
                    inputType.value = 'cui';
                    diseaseInput.value = ui.item.cui;
                } else {
                    inputType.value = 'name';
                    diseaseInput.value = ui.item.name;
                }
                
                // Store the full information
                selectedDiseaseName.textContent = ui.item.name;
                selectedDiseaseCui.textContent = ui.item.cui;
                selectedDiseaseInfo.classList.remove('d-none');
                
                return false;
            }
        }).autocomplete("instance")._renderItem = function(ul, item) {
            return $("<li>")
                .append("<div class='autocomplete-item'>" +
                        "<span class='autocomplete-name'>" + item.name + "</span>" +
                        "<span class='autocomplete-cui'>" + item.cui + "</span>" +
                        "</div>")
                .appendTo(ul);
        };
    }
    
    // Reset input fields and selection info
    function resetInputs() {
        drugInput.value = '';
        diseaseInput.value = '';
        selectedDrugInfo.classList.add('d-none');
        selectedDiseaseInfo.classList.add('d-none');
    }
    
    // Clear selection info when input is manually changed
    drugInput.addEventListener('input', function() {
        if (drugInput.value === '') {
            selectedDrugInfo.classList.add('d-none');
        }
    });
    
    diseaseInput.addEventListener('input', function() {
        if (diseaseInput.value === '') {
            selectedDiseaseInfo.classList.add('d-none');
        }
    });
    
    // Form submission handler
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Check if we have valid entity selections
        let isValid = true;
        let errorMessage = '';
        
        if (selectedDrugInfo.classList.contains('d-none')) {
            isValid = false;
            errorMessage = 'Please select a valid drug from the dropdown suggestions';
        } else if (selectedDiseaseInfo.classList.contains('d-none')) {
            isValid = false;
            errorMessage = 'Please select a valid disease from the dropdown suggestions';
        }
        
        if (!isValid) {
            errorAlert.textContent = errorMessage;
            errorAlert.classList.remove('d-none');
            return;
        }
        
        // Get form data
        const formData = new FormData(predictionForm);
        
        // Make sure the right input values are sent
        if (selectedDrugCui.textContent) {
            formData.set('drug_cui', selectedDrugCui.textContent);
            formData.set('drug_input', selectedDrugCui.textContent);
        }
        
        if (selectedDiseaseCui.textContent) {
            formData.set('disease_cui', selectedDiseaseCui.textContent);
            formData.set('disease_input', selectedDiseaseCui.textContent);
        }
        
        // Force input_type to CUI to ensure the backend uses CUIs for prediction
        formData.set('input_type', 'cui');
        
        // Show loading state
        document.querySelector('button[type="submit"]').innerHTML = 
            '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        document.querySelector('button[type="submit"]').disabled = true;
        
        // Hide previous results and errors
        resultSection.classList.add('d-none');
        errorAlert.classList.add('d-none');
        
        // Send the request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            document.querySelector('button[type="submit"]').innerHTML = 'Predict Relationship';
            document.querySelector('button[type="submit"]').disabled = false;
            
            if (data.error) {
                // Show error
                errorAlert.textContent = data.error;
                errorAlert.classList.remove('d-none');
            } else {
                // Display results
                document.getElementById('result-drug-name').textContent = data.drug_name;
                document.getElementById('result-drug-cui').textContent = data.drug_cui;
                document.getElementById('result-disease-name').textContent = data.disease_name;
                document.getElementById('result-disease-cui').textContent = data.disease_cui;
                document.getElementById('result-probability').textContent = 
                    `${data.probability_percent.toFixed(2)}%`;
                
                // Set interpretation with appropriate color class
                const interpretation = document.getElementById('result-interpretation');
                interpretation.textContent = data.interpretation;
                
                // Remove previous color classes
                interpretation.classList.remove('very-high', 'high', 'medium', 'low', 'very-low');
                
                // Add appropriate color class
                if (data.probability > 0.8) {
                    interpretation.classList.add('very-high');
                } else if (data.probability > 0.6) {
                    interpretation.classList.add('high');
                } else if (data.probability > 0.4) {
                    interpretation.classList.add('medium');
                } else if (data.probability > 0.2) {
                    interpretation.classList.add('low');
                } else {
                    interpretation.classList.add('very-low');
                }
                
                // Update progress bar
                const probabilityBar = document.getElementById('probability-bar');
                probabilityBar.style.width = `${data.probability_percent}%`;
                probabilityBar.setAttribute('aria-valuenow', data.probability_percent);
                
                // Set progress bar color based on probability
                probabilityBar.className = 'progress-bar';
                if (data.probability > 0.8) {
                    probabilityBar.classList.add('bg-success');
                    probabilityBar.textContent = `${data.probability_percent.toFixed(1)}%`;
                } else if (data.probability > 0.6) {
                    probabilityBar.classList.add('bg-info');
                    probabilityBar.textContent = `${data.probability_percent.toFixed(1)}%`;
                } else if (data.probability > 0.4) {
                    probabilityBar.classList.add('bg-warning');
                    probabilityBar.textContent = `${data.probability_percent.toFixed(1)}%`;
                } else if (data.probability > 0.2) {
                    probabilityBar.classList.add('bg-danger');
                    probabilityBar.textContent = `${data.probability_percent.toFixed(1)}%`;
                } else {
                    probabilityBar.classList.add('bg-danger');
                    probabilityBar.textContent = `${data.probability_percent.toFixed(1)}%`;
                }
                
                // Show result section
                resultSection.classList.remove('d-none');
                
                // Scroll to result
                resultSection.scrollIntoView({ behavior: 'smooth' });
            }
        })
        .catch(error => {
            // Reset button state
            document.querySelector('button[type="submit"]').innerHTML = 'Predict Relationship';
            document.querySelector('button[type="submit"]').disabled = false;
            
            // Show error
            errorAlert.textContent = 'An error occurred during prediction. Please try again.';
            errorAlert.classList.remove('d-none');
            console.error('Error:', error);
        });
    });
});