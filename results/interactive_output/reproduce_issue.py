import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import IsolationForest

# Create a sample DataFrame
X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

# Set up a warning filter to catch the specific warning
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    
    # Fit the model with contamination != 'auto'
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    
    # Check if the warning was raised
    found = False
    for w in caught_warnings:
        # Check for the specific warning message that is common in scikit-learn
        if "X does not have valid feature names" in str(w.message):
            print(f"Warning found: {w.message}")
            found = True
            
    if not found:
        print("No spurious warning found.")
    else:
        print("Spurious warning found.")
        
    # Also test score_samples directly as that's where validation happens
    print("Testing score_samples directly...")
    model.score_samples(X) 
