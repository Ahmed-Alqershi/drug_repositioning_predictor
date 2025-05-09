"""
Script to prepare test dataset for predictions and run the model.
Also provides functionality to predict probability of drug-disease relationships.
"""

import logging
import numpy as np
import tensorflow as tf
from gensim.models import FastText
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def predict_drug_disease_pair(drug_cui, disease_cui):
    """
    Predict the probability that a specific drug treats a specific disease.
    
    Args:
        drug_cui: The CUI code for the drug
        disease_cui: The CUI code for the disease
        model_path: Path to the SNN model
        embedding_model_path: Path to the FastText model
        cnn: Whether CNN was used in model (1=yes, 0=no)
        
    Returns:
        Probability (float) that the drug treats the disease
    """

    model_path = 'best_model.keras'

    logger.info(f"Predicting relationship between drug {drug_cui} and disease {disease_cui}")

    # Load models
    try:
        # Load JSON file for drug/disease word vectors
        with open("subjects.json") as f:
            subjects = json.load(f)

        with open("objects.json") as f:
            objects = json.load(f)
        
        # Load SNN model
        model = tf.keras.models.load_model(model_path)

        drug_vec = subjects.get(drug_cui)
        disease_vec = objects.get(disease_cui)

        # Reshape for model input
        drug_arr = np.array([drug_vec])
        disease_arr = np.array([disease_vec])
        
        # drug_arr = np.expand_dims(drug_arr, axis=-1)
        # disease_arr = np.expand_dims(disease_arr, axis=-1)
        
        # Get prediction
        prediction = model.predict([drug_arr, disease_arr])[0][0]
        
        logger.info(f"Prediction for {drug_cui}-{disease_cui}: {prediction:.4f}")
        return prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
