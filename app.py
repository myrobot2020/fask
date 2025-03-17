import pickle
import numpy as np
from flask import Flask, render_template, request

# Load the trained model BEFORE initializing Flask
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        try:
            # Get user input from form
            features = [float(request.form[key]) for key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
            
            # Convert to numpy array and reshape for model input
            input_data = np.array(features).reshape(1, -1)
            
            # Make a prediction
            predicted_class = model.predict(input_data)[0]
            
            # Mapping of class labels to species names
            species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            prediction = species[predicted_class]
            
        except ValueError:
            prediction = "Invalid input! Please enter numeric values."
    
    return render_template("index.html", prediction=prediction)

# Run Flask app safely
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Prevent auto-reloader issues
