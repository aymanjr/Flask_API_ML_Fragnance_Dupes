# Fragrance Dupes Finder

A Flask web application that uses machine learning to find similar fragrances (dupes) based on user input.

## Features

- Search for fragrance dupes by name
- ML-powered similarity matching based on fragrance notes and characteristics
- Responsive web interface
- API endpoint for programmatic access

## Project Structure

```
├── app.py                  # Main Flask application
├── models/
│   └── fragrance_model.py  # ML model for finding fragrance dupes
├── data/
│   └── fragrances.csv      # Sample fragrance dataset (auto-generated)
├── static/
│   └── css/
│       └── style.css       # CSS styles
└── templates/
    ├── index.html          # Main search page
    └── results.html        # Results page
```

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application locally:

```bash
python app.py
```

The application will be available at http://127.0.0.1:5000/

## API Usage

The application provides a simple API endpoint for programmatic access:

```
GET /api/dupes?name=<fragrance_name>
```

Example response:

```json
{
  "original_fragrance": "Aventus",
  "dupes": [
    {
      "name": "Club de Nuit Intense Man",
      "brand": "Armaf",
      "notes": "...",
      "price_range": "low"
    },
    ...
  ]
}
```

## Sample Data

The application comes with a sample dataset of popular fragrances. If no dataset is found, it will automatically generate sample data for demonstration purposes.

## Technologies Used

- Flask
- Pandas
- Scikit-learn
- HTML/CSS
