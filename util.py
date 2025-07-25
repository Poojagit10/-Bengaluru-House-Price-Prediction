import pickle
import json
import numpy as np
import pandas as pd

# --- Global Variables ---
__locations = None
__data_columns = None
__model = None
__location_mapping = None  # For case-insensitive lookup


def get_estimated_price(location, sqft, bhk, bath):
    """
    Estimates the price of a property in Bengaluru.

    Args:
        location (str): The property's location (case-insensitive).
        sqft (float): The total square footage.
        bhk (int): The number of bedrooms.
        bath (int): The number of bathrooms.

    Returns:
        float: The estimated price in lakhs, rounded to two decimal places.
    """
    # Create the feature vector with zeros
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Find the canonical (correctly cased) location name from user input
    canonical_location = __location_mapping.get(location.strip().lower())

    # If the location is found, set its corresponding one-hot feature to 1
    if canonical_location:
        try:
            # Find the index of the canonical location
            loc_index = list(__data_columns).index(canonical_location)
            x[loc_index] = 1
        except ValueError:
            # This case should ideally not be reached if mapping is correct
            print(f"Warning: Location '{canonical_location}' not found in model columns.")
            pass

    # Create a DataFrame with the correct feature names for prediction
    x_df = pd.DataFrame([x], columns=__data_columns)

    return round(__model.predict(x_df)[0], 2)


def get_location_names():
    """Returns a list of all available location names."""
    return __locations


def load_saved_artifacts():
    """
    Loads the trained model and artifacts.
    The model itself is the source of truth for feature names.
    """
    print("Loading saved artifacts...start")
    global __data_columns, __locations, __model, __location_mapping

    # Load the trained model from the pickle file
    with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("Model loaded.")

    # Get feature names directly from the model (most reliable method)
    __data_columns = __model.feature_names_in_
    __locations = list(__data_columns[3:])  # First 3 are sqft, bath, bhk

    # Create a case-insensitive mapping for user-friendly lookup
    __location_mapping = {loc.lower(): loc for loc in __locations}

    print("Loading saved artifacts...done")


# --- Main execution block for testing ---
if __name__ == '__main__':
    load_saved_artifacts()

    print("\nAvailable locations:", len(get_location_names()))
    # print(get_location_names()) # Uncomment to see all locations

    print("\n--- Testing Predictions ---")

    # Test 1: Known location with correct casing
    price1 = get_estimated_price('1st Phase JP Nagar', 1000, 3, 3)
    print(f"Price for '1st Phase JP Nagar' (3 BHK): {price1} Lakhs")

    # Test 2: Known location with incorrect (lowercase) casing
    price2 = get_estimated_price('1st phase jp nagar', 1000, 2, 2)
    print(f"Price for '1st phase jp nagar' (2 BHK): {price2} Lakhs")

    # Test 3: Unknown location
    price3 = get_estimated_price('Kalhalli', 1000, 2, 2)
    print(f"Price for 'Kalhalli' (Unknown Location): {price3} Lakhs")

    # Test 4: Another known location
    price4 = get_estimated_price('Indira Nagar', 1000, 2, 2)
    print(f"Price for 'Indira Nagar' (2 BHK): {price4} Lakhs")
