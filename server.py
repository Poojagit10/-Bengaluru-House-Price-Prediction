# In server/server.py

from flask import Flask, request, jsonify
import util  # Imports your util.py

app = Flask(__name__)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """
    Endpoint to get all the available location names.
    """
    # FIX 1: Corrected jsonify and added () to the function call
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    """
    Endpoint to predict the home price.
    Expects form data: total_sqft, location, bhk, and bath.
    """
    try:
        # Extract data from the POST request form
        total_sqft = float(request.form['total_sqft'])
        location = request.form['location']
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])

        # FIX 2: Correct argument order: (location, sqft, bhk, bath)
        estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)

        # Create a successful JSON response
        response = jsonify({
            'estimated_price': estimated_price
        })
    except KeyError:
        return jsonify({'error': 'Missing form data. Please provide total_sqft, location, bhk, and bath.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    response.headers.add('Access-Control-Allow-Origin', '*')

    # FIX 3: Added the missing return statement
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    # FIX 4: Load artifacts when the server starts
    util.load_saved_artifacts()
    app.run()
