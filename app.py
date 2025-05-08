import os
import sys
import pandas as pd
import logging
from flask import Flask, render_template, request, jsonify


# Add parent directory to path so we can import from predict.py
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import predict_drug_disease_pair

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load the entity data for lookups
drug_data = pd.read_csv('unique_subjects.csv')
disease_data = pd.read_csv('unique_objects.csv')

# Create dictionaries for fast lookups
drug_cui_to_name = dict(zip(drug_data['Subject_CUI'], drug_data['Subject_Name']))
drug_name_to_cui = dict(zip(drug_data['Subject_Name'], drug_data['Subject_CUI']))
disease_cui_to_name = dict(zip(disease_data['Object_CUI'], disease_data['Object_Name']))
disease_name_to_cui = dict(zip(disease_data['Object_Name'], disease_data['Object_CUI']))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/drugs', methods=['GET'])
def get_drugs():
    """Return a list of all drugs for autocomplete"""
    items = []
    for _, row in drug_data.iterrows():
        items.append({
            'cui': row['Subject_CUI'],
            'name': row['Subject_Name']
        })
    return jsonify(items)

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Return a list of all diseases for autocomplete"""
    items = []
    for _, row in disease_data.iterrows():
        items.append({
            'cui': row['Object_CUI'],
            'name': row['Object_Name']
        })
    return jsonify(items)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data
        drug_input = request.form.get('drug_input')
        disease_input = request.form.get('disease_input')
        input_type = request.form.get('input_type', 'auto')  # Default to auto if not specified
        
        # Validate input
        if not drug_input or not disease_input:
            return jsonify({'error': 'Please provide both drug and disease information'}), 400
        
        # Determine if inputs are CUIs or names
        drug_cui = None
        disease_cui = None
        
        # Try to use direct CUIs first (most reliable)
        if drug_input in drug_cui_to_name:
            drug_cui = drug_input
            logger.info(f"Found direct drug CUI match for {drug_input}")
        elif drug_input in drug_name_to_cui:
            drug_cui = drug_name_to_cui.get(drug_input)
            logger.info(f"Found drug name match for {drug_input}, CUI: {drug_cui}")
        else:
            # Try case-insensitive search for partial matches
            drug_cui_lower = drug_input.lower()
            for cui, name in drug_cui_to_name.items():
                if cui.lower() == drug_cui_lower:
                    drug_cui = cui
                    break
            
            if not drug_cui:
                for name, cui in drug_name_to_cui.items():
                    if name.lower() == drug_input.lower():
                        drug_cui = cui
                        break
        
        # Same approach for disease
        if disease_input in disease_cui_to_name:
            disease_cui = disease_input
            logger.info(f"Found direct disease CUI match for {disease_input}")
        elif disease_input in disease_name_to_cui:
            disease_cui = disease_name_to_cui.get(disease_input)
            logger.info(f"Found disease name match for {disease_input}, CUI: {disease_cui}")
        else:
            # Try case-insensitive search for partial matches
            disease_cui_lower = disease_input.lower()
            for cui, name in disease_cui_to_name.items():
                if cui.lower() == disease_cui_lower:
                    disease_cui = cui
                    break
            
            if not disease_cui:
                for name, cui in disease_name_to_cui.items():
                    if name.lower() == disease_input.lower():
                        disease_cui = cui
                        break
        
        # Validate that we found valid CUIs
        if not drug_cui:
            return jsonify({'error': f'Could not find a matching drug for "{drug_input}" in our database'}), 400
        if not disease_cui:
            return jsonify({'error': f'Could not find a matching disease for "{disease_input}" in our database'}), 400
        
        # Get entity names for display
        drug_name = drug_cui_to_name.get(drug_cui, "Unknown")
        disease_name = disease_cui_to_name.get(disease_cui, "Unknown")
    
        # Default model paths
        model_path = 'best_model.keras'
        embedding_model_path = 'fasttext_model.pkl'
        
        # Get prediction
        probability = predict_drug_disease_pair(
            drug_cui, 
            disease_cui,
            model_path=model_path,
            embedding_model_path=embedding_model_path,
            cnn=0,
        )
        
        # Determine interpretation
        if probability > 0.8:
            interpretation = "Very likely to treat the disease"
        elif probability > 0.6:
            interpretation = "Likely to treat the disease"
        elif probability > 0.4:
            interpretation = "Uncertain relationship"
        elif probability > 0.2:
            interpretation = "Unlikely to treat the disease"
        else:
            interpretation = "Very unlikely to treat the disease"
        
        # Return the result with full entity information
        return jsonify({
            'drug_cui': drug_cui,
            'drug_name': drug_name,
            'disease_cui': disease_cui,
            'disease_name': disease_name,
            'probability': float(probability),
            'probability_percent': float(probability * 100),
            'interpretation': interpretation
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)