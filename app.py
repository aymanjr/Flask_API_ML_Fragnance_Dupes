from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from models.fragrance_model import FragranceModel

app = Flask(__name__)

# Initialize the model
model = FragranceModel()

@app.route('/')
def index():
    """Render the main page with the search form."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle the search request and return fragrance dupes."""
    if request.method == 'POST':
        fragrance_name = request.form.get('fragrance_name', '')
        
        if not fragrance_name:
            return jsonify({'error': 'Please enter a fragrance name'}), 400
        
        # Get dupes from the model
        dupes = model.find_dupes(fragrance_name)
        
        return render_template('results.html', 
                              original_fragrance=fragrance_name, 
                              dupes=dupes)

@app.route('/api/dupes', methods=['GET'])
def api_dupes():
    """API endpoint to get fragrance dupes."""
    fragrance_name = request.args.get('name', '')
    
    if not fragrance_name:
        return jsonify({'error': 'Please provide a fragrance name'}), 400
    
    # Get dupes from the model
    dupes = model.find_dupes(fragrance_name)
    
    return jsonify({
        'original_fragrance': fragrance_name,
        'dupes': dupes
    })

if __name__ == '__main__':
    app.run(debug=True)
