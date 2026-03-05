import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings
import numpy as np

# Create a sample DataFrame
X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

# Use a different model instance to ensure no prior state
model = IsolationForest(contamination=0.1, random_state=42)

# This should no longer trigger the warning
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    
    # Fit the model
    model.fit(X)
    
    # Check if the warning was raised
    found = False
    for w in caught_warnings:
        if "X does not have valid feature names" in str(w.message):
            print(f"Warning found during fit: {w.message}")
            found = True
            
    if not found:
        print("No warning found during fit!")
    else:
        print("Warning found during fit!")

# Now test score_samples directly, which triggers the validation
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    
    # Pass a numpy array that doesn't have feature names
    model.score_samples(X.to_numpy())
    
    # Check if the warning was raised
    found = False
    for w in caught_warnings:
        if "X does not have valid feature names" in str(w.message):
            print(f"Warning found during score_samples: {w.message}")
            found = True
    if not found:
        print("No warning found for score_samples(numpy_array)")
    else:
        print("Warning found for score_samples(numpy_array)")
