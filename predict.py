"""
Script to prepare test dataset for predictions and run the model.
Also provides functionality to predict probability of drug-disease relationships.
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from gensim.models import FastText
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run predictions on test dataset")
    parser.add_argument('--model_path', type=str, default='results/models/best_model.keras',
                        help='Path to the trained model')
    parser.add_argument('--embedding_model', type=str, default='results/models/fasttext_model.pkl',
                        help='Path to the saved FastText model')
    parser.add_argument('--cnn', type=int, default=0, help='1; CNN was used, 0; Sequential model was used')
    parser.add_argument('--we', type=int, default=1024, help='Word Embedding vector size (for fallback)')
    return parser.parse_args()

def load_data():
    """Load and prepare the test dataset."""
    data_dir = "drug_repo/data"
    logger.info("Loading test dataset")
    test_df = pd.read_csv(f"{data_dir}/test_dataset.csv")
    
    # Extract drug-disease pairs
    test_data = test_df[
        (test_df.SUBJECT_SEMTYPE.str.contains("phsu")) &
        (test_df.OBJECT_SEMTYPE.str.contains("dsyn"))
    ]
    
    # Create a fresh dataset with only relevant columns
    test_pairs = test_data[['SUBJECT_CUI', 'OBJECT_CUI', 'status']].copy()
    
    logger.info(f"Extracted {len(test_pairs)} drug-disease pairs from test dataset")
    
    return test_pairs

def load_embedding_model(model_path):
    """Load the saved FastText model."""
    logger.info(f"Loading FastText model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            ft_model = pickle.load(f)
        logger.info("FastText model loaded successfully")
        return ft_model
    except FileNotFoundError:
        logger.error(f"FastText model file not found at {model_path}")
        logger.warning("You need to save the FastText model during training.")
        logger.warning("Add the following to main.py after building the model:")
        logger.warning("import pickle")
        logger.warning("os.makedirs('results/models', exist_ok=True)")
        logger.warning("with open('results/models/fasttext_model.pkl', 'wb') as f:")
        logger.warning("    pickle.dump(ft_model, f)")
        raise
    except Exception as e:
        logger.error(f"Error loading FastText model: {e}")
        raise

def create_test_vectors(test_pairs, ft_model, args):
    """Create vectors for test pairs using the loaded FastText model."""
    logger.info("Creating vectors for test pairs")
    
    # Function to get word vectors
    def get_word_vec(word):
        try:
            return ft_model.wv.get_vector(word)
        except KeyError:
            logger.warning(f"Word '{word}' not in vocabulary. Using zeros vector.")
            return np.zeros(args.we)
    
    # Get vectors for drugs and diseases
    test_drugs_vec = test_pairs.SUBJECT_CUI.apply(get_word_vec)
    test_disease_vec = test_pairs.OBJECT_CUI.apply(get_word_vec)
    
    # Convert to numpy arrays
    test_drugs_arr = np.array(test_drugs_vec.tolist())
    test_disease_arr = np.array(test_disease_vec.tolist())
    
    # Reshape for CNN if needed
    if args.cnn:
        test_drugs_arr = np.expand_dims(test_drugs_arr, axis=-1)
        test_disease_arr = np.expand_dims(test_disease_arr, axis=-1)
    
    logger.info(f"Created test vectors with shape: drugs {test_drugs_arr.shape}, diseases {test_disease_arr.shape}")
    
    return test_drugs_arr, test_disease_arr

def load_snn_model(model_path):
    """Load the trained SNN model."""
    logger.info(f"Loading SNN model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("SNN model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading SNN model: {e}")
        raise

def predict(model, test_drugs_arr, test_disease_arr, test_pairs):
    """Make predictions on test pairs."""
    logger.info("Making predictions on test pairs")
    
    # Get predictions
    predictions = model.predict([test_drugs_arr, test_disease_arr])
    
    # Add predictions to test_pairs
    test_pairs['prediction'] = predictions
    test_pairs['predicted_class'] = (predictions > 0.5).astype(int)
    
    # Log some statistics
    avg_pred = test_pairs['prediction'].mean()
    pos_pred = test_pairs[test_pairs['predicted_class'] == 1].shape[0]
    
    logger.info(f"Average prediction score: {avg_pred:.4f}")
    logger.info(f"Positive predictions: {pos_pred} ({pos_pred/len(test_pairs)*100:.2f}%)")
    
    # Create a clean results dataframe with only Drug, Disease, P(treats)
    clean_results = test_pairs[['SUBJECT_CUI', 'OBJECT_CUI', 'prediction']].copy()
    clean_results.columns = ['Drug', 'Disease', 'P(treats)']
    
    # Sort by probability from highest to lowest
    clean_results = clean_results.sort_values(by='P(treats)', ascending=False)
    
    # Save the full results for detailed analysis
    test_pairs.to_csv("results/test_predictions_full.csv", index=False)
    
    # Save the clean sorted results
    output_file = "results/test_predictions.csv"
    clean_results.to_csv(output_file, index=False)
    logger.info(f"Clean predictions saved to {output_file}")
    
    return test_pairs

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Load test data
    test_pairs = load_data()
    
    # Load FastText model
    ft_model = load_embedding_model(args.embedding_model)
    
    # Create vectors for test pairs
    test_drugs_arr, test_disease_arr = create_test_vectors(test_pairs, ft_model, args)
    
    # Load SNN model
    snn_model = load_snn_model(args.model_path)
    
    # Make predictions
    results = predict(snn_model, test_drugs_arr, test_disease_arr, test_pairs)
    
    # Analyze by status
    logger.info("\nPrediction analysis by status:")
    status_analysis = results.groupby('status').agg({
        'prediction': ['mean', 'std', 'count'],
        'predicted_class': ['sum']
    })
    
    status_analysis.columns = ['mean_score', 'std_score', 'count', 'positive_count']
    status_analysis['positive_percentage'] = status_analysis['positive_count'] / status_analysis['count'] * 100
    
    print("\nPrediction analysis by status:")
    print(status_analysis)
    
    # Save the analysis
    status_analysis.to_csv("results/status_analysis.csv")
    logger.info("Analysis saved to results/status_analysis.csv")
    
    # Print top 10 highest probability predictions for reference
    top_predictions = pd.read_csv("results/test_predictions.csv").head(10)
    print("\nTop 10 highest probability drug-disease pairs:")
    print(top_predictions)


def predict_drug_disease_pair(drug_cui, disease_cui, model_path=None, embedding_model_path=None, cnn=0):
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
    # Set default paths if not provided
    if model_path is None:
        model_path = 'results/models/best_model.keras'
    if embedding_model_path is None:
        embedding_model_path = 'results/models/fasttext_model.pkl'
    
    logger.info(f"Predicting relationship between drug {drug_cui} and disease {disease_cui}")
    
    # Load models
    try:
        # Load FastText model
        with open(embedding_model_path, 'rb') as f:
            ft_model = pickle.load(f)

        vocab = list(ft_model.wv.key_to_index.keys())

        if drug_cui.lower() not in vocab:
            raise ValueError(f"Drug CUI '{drug_cui}' not in vocabulary. Please check the input.")
        if disease_cui.lower() not in vocab:
            raise ValueError(f"Disease CUI '{disease_cui}' not in vocabulary. Please check the input.")
        
        # Load SNN model
        model = tf.keras.models.load_model(model_path)
        
        # Get vectors for drug and disease
        def get_word_vec(word):
            vec = ft_model.wv.get_vector(word)
            return vec
        
        drug_vec = get_word_vec(drug_cui)
        disease_vec = get_word_vec(disease_cui)
        
        # Reshape for model input
        drug_arr = np.array([drug_vec])
        disease_arr = np.array([disease_vec])
        
        # Further reshape for CNN if needed
        if cnn:
            drug_arr = np.expand_dims(drug_arr, axis=-1)
            disease_arr = np.expand_dims(disease_arr, axis=-1)
        
        # Get prediction
        prediction = model.predict([drug_arr, disease_arr])[0][0]
        
        logger.info(f"Prediction for {drug_cui}-{disease_cui}: {prediction:.4f}")
        return prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()