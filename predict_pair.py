#!/usr/bin/env python3
"""
Standalone script to predict the probability of a drug-disease relationship.
"""

import argparse
import sys
import logging
from predict import predict_drug_disease_pair

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict the probability that a drug treats a disease"
    )
    parser.add_argument('drug_cui', type=str, help='CUI code for the drug')
    parser.add_argument('disease_cui', type=str, help='CUI code for the disease')
    parser.add_argument('--model_path', type=str, default='results/models/best_model.keras',
                        help='Path to the trained model')
    parser.add_argument('--embedding_model', type=str, default='results/models/fasttext_model.pkl',
                        help='Path to the saved FastText model')
    parser.add_argument('--cnn', type=int, default=0, help='1; CNN was used, 0; Sequential model was used')
    parser.add_argument('--we', type=int, default=1024, help='Word Embedding vector size')
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Get prediction
        probability = predict_drug_disease_pair(
            args.drug_cui, 
            args.disease_cui,
            model_path=args.model_path,
            embedding_model_path=args.embedding_model,
            cnn=args.cnn,
            we=args.we
        )
        
        # Print result
        print(f"\nPrediction Results:")
        print(f"------------------")
        print(f"Drug CUI:     {args.drug_cui}")
        print(f"Disease CUI:  {args.disease_cui}")
        print(f"Probability:  {probability:.4f} ({probability*100:.2f}%)")
        print(f"Interpretation: ", end="")
        if probability > 0.8:
            print("Very likely to treat the disease")
        elif probability > 0.6:
            print("Likely to treat the disease")
        elif probability > 0.4:
            print("Uncertain relationship")
        elif probability > 0.2:
            print("Unlikely to treat the disease")
        else:
            print("Very unlikely to treat the disease")
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()