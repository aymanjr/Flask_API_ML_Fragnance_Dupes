# Fragrance Dupes Finder

A Flask web application that uses machine learning to find similar fragrances (dupes) based on user input. The application uses a comprehensive dataset of over 100,000 fragrances to provide accurate and diverse recommendations.

![Fragrance Dupes Finder](https://img.shields.io/badge/Fragrances-100K%2B-purple)
![ML Powered](https://img.shields.io/badge/ML-Powered-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)

## Features

- Search for fragrance dupes by name
- ML-powered similarity matching based on fragrance notes and characteristics
- Responsive web interface
- API endpoint for programmatic access
- Large dataset with over 100,000 fragrances

## Project Structure

```
├── app.py                          # Main Flask application
├── models/
│   └── fragrance_model.py          # ML model for finding fragrance dupes
├── data/
│   ├── all_merged_fragrances.csv   # Main dataset with 100,000+ fragrances
│   ├── fra_cleaned.csv             # Original dataset
│   ├── fra_perfumes.csv            # Original dataset
│   └── merged_cleaned_fragrances.csv # Intermediate dataset
├── static/
│   └── css/
│       └── style.css               # CSS styles
└── templates/
    ├── index.html                  # Main search page
    └── results.html                # Results page
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

## How to Use

1. **Enter a Fragrance Name**: Type the name of a fragrance you're interested in finding alternatives for. The search is flexible and will try to match partial names.

2. **View Results**: The application will display a list of similar fragrances based on their note composition. Each result includes:
   - Fragrance name and brand
   - Notes composition
   - Price range indicator (budget, mid-range, premium)
   - Gender classification

3. **Try Different Searches**: Experiment with different fragrances to discover new scents that match your preferences.

## Dataset

The application uses a comprehensive dataset of over 100,000 fragrances compiled from multiple sources. The dataset includes:

- Fragrance name and brand
- Gender classification (men, women, unisex)
- Rating information
- Perfumers
- Main accords
- Detailed notes (top, middle, base)
- URLs for more information

## Similarity Algorithm

The application uses natural language processing techniques to find similar fragrances:

1. **TF-IDF Vectorization**: Converts fragrance notes into numerical vectors that capture the importance of each note
2. **Cosine Similarity**: Measures the similarity between fragrances based on their note vectors
3. **Filtering Mechanism**: 
   - Removes the original fragrance from results
   - Prevents duplicate fragrances from appearing
   - Ensures diversity in the recommendations
   - Prioritizes fragrances with similar note profiles

This approach allows the application to find fragrances that share similar olfactory characteristics, even if they come from different brands or price points.

## Technologies Used

- **Flask**: Web framework for the application
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms for similarity calculation
- **TF-IDF Vectorization**: Text feature extraction for fragrance notes
- **Cosine Similarity**: Measuring similarity between fragrance vectors
- **HTML/CSS**: Frontend interface with responsive design

## Recent Updates

- Integrated a comprehensive dataset with over 100,000 fragrances
- Improved similarity algorithm to provide more diverse recommendations
- Enhanced filtering to prevent duplicate results
- Optimized search functionality for better matching
- Updated UI for better user experience

## Future Improvements

- Add price information for more accurate budget categorization
- Implement user accounts to save favorite fragrances
- Add filtering options (by gender, brand, price range)
- Incorporate user feedback to improve recommendations
- Add image support for fragrance bottles
