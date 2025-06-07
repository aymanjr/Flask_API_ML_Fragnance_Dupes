import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FragranceModel:
    def __init__(self):
        """Initialize the fragrance model with data."""
        self.data_path = os.path.join('data', 'fragrances.csv')
        self._load_data()
        self._prepare_model()
        
    def _load_data(self):
        """Load fragrance data from CSV file or create sample data if not exists."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} fragrances from {self.data_path}")
        except FileNotFoundError:
            print(f"Data file not found at {self.data_path}. Creating sample data.")
            self._create_sample_data()
            
    def _create_sample_data(self):
        """Create sample fragrance data for demonstration."""
        # Sample data with fragrance name, brand, notes, and price range
        data = {
            'name': [
                'Aventus', 'Green Irish Tweed', 'Sauvage', 'Bleu de Chanel', 
                'Acqua di Gio', 'La Nuit de L\'Homme', 'Eros', 'One Million',
                'Tobacco Vanille', 'Noir de Noir', 'Tuscan Leather', 'Oud Wood',
                'Baccarat Rouge 540', 'Grand Soir', 'Delina', 'Layton',
                'Black Orchid', 'Ombré Leather', 'Noir Extreme', 'Soleil Blanc'
            ],
            'brand': [
                'Creed', 'Creed', 'Dior', 'Chanel', 
                'Giorgio Armani', 'Yves Saint Laurent', 'Versace', 'Paco Rabanne',
                'Tom Ford', 'Tom Ford', 'Tom Ford', 'Tom Ford',
                'Maison Francis Kurkdjian', 'Maison Francis Kurkdjian', 'Parfums de Marly', 'Parfums de Marly',
                'Tom Ford', 'Tom Ford', 'Tom Ford', 'Tom Ford'
            ],
            'notes': [
                'pineapple, blackcurrant, apple, bergamot, birch, patchouli, ambergris, vanilla',
                'lemon, verbena, iris, violet leaf, sandalwood, ambergris',
                'bergamot, pepper, ambroxan, lavender, vetiver',
                'citrus, ginger, jasmine, sandalwood, amber, cedar',
                'bergamot, neroli, green tangerine, jasmine, rosemary, persimmon, patchouli, cedar, white musk',
                'cardamom, bergamot, lavender, cedar, coumarin, vetiver',
                'mint, green apple, lemon, tonka bean, vanilla, vetiver, cedar, oakmoss',
                'blood mandarin, grapefruit, mint, rose, cinnamon, spices, leather, woody notes, amber, patchouli',
                'tobacco leaf, vanilla, cocoa, tonka bean, tobacco blossom, dried fruits, wood',
                'rose, black truffle, vanilla, patchouli, oud, saffron',
                'leather, raspberry, thyme, olibanum, cedar, jasmine',
                'oud, sandalwood, rosewood, eastern spices, vanilla, amber, vetiver',
                'saffron, jasmine, ambergris, cedar, fir resin',
                'benzoin, amber, vanilla, tonka bean',
                'turkish rose, lychee, rhubarb, bergamot, vanilla, musk',
                'bergamot, lavender, apple, geranium, vanilla, cardamom, pink pepper',
                'black truffle, ylang-ylang, bergamot, blackcurrant, dark chocolate, patchouli, vanilla',
                'leather, cardamom, jasmine, patchouli, vetiver, cedar',
                'mandarin, neroli, saffron, kulfi, rose, jasmine, amber, sandalwood',
                'bergamot, cardamom, ylang-ylang, tuberose, coconut, amber, benzoin'
            ],
            'price_range': [
                'high', 'high', 'medium', 'medium',
                'medium', 'medium', 'medium', 'medium',
                'high', 'high', 'high', 'high',
                'high', 'high', 'high', 'high',
                'high', 'high', 'high', 'high'
            ],
            'dupes': [
                'Club de Nuit Intense Man, Explorer, Cedrat Boise',
                'Cool Water, Aspen, Tres Nuit',
                'Luna Rossa Carbon, Dylan Blue, Y Eau de Parfum',
                'Dylan Blue, Allure Homme Sport, Luna Rossa',
                'Perry Ellis 360 Red, Cool Water, Nautica Voyage',
                'CH Men Prive, The One, Bentley for Men Intense',
                'Joop Homme, 1 Million, Invictus',
                'Joop Homme, Eros, Invictus',
                'Vanilla Tobacco, Phaedon Tabac Rouge, Franck Boclet Tobacco',
                'Rose Oud, Velvet Rose & Oud, Cafe Rose',
                'Godolphin, Rasasi La Yuqawam, Montale Aoud Leather',
                'Versace Oud Noir, Commodity Oud, Montale Aoud Forest',
                'Ariana Grande Cloud, Burberry Her, Al Haramain Amber Oud Gold Edition',
                'Amber Oud Gold Edition, Amber Aoud, Ambre Nuit',
                'Roses Greedy, Rose Elixir, Roses Vanille',
                'Herod, Naxos, Pegasus',
                'Velvet Orchid, Orchid Soleil, Hypnotic Poison',
                'Godolphin, Tuscan Leather, La Yuqawam',
                'Spicebomb Extreme, Pure XS, Stronger With You',
                'Bronze Goddess, Terracotta, Eau des Lagons'
            ]
        }
        
        self.df = pd.DataFrame(data)
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        self.df.to_csv(self.data_path, index=False)
        print(f"Created sample data with {len(self.df)} fragrances and saved to {self.data_path}")
    
    def _prepare_model(self):
        """Prepare the TF-IDF vectorizer for similarity calculations."""
        # Combine relevant features for similarity calculation
        self.df['features'] = self.df['name'] + ' ' + self.df['brand'] + ' ' + self.df['notes']
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.features_matrix = self.vectorizer.fit_transform(self.df['features'])
    
    def find_dupes(self, fragrance_name, top_n=5):
        """Find dupes for a given fragrance name."""
        # First, try to find an exact match
        exact_matches = self.df[self.df['name'].str.lower() == fragrance_name.lower()]
        
        if not exact_matches.empty:
            # If we have an exact match, return its known dupes
            fragrance = exact_matches.iloc[0]
            dupes_list = fragrance['dupes'].split(', ')
            
            # Get additional info for each dupe
            dupes_info = []
            for dupe in dupes_list:
                dupe_match = self.df[self.df['name'].str.lower() == dupe.lower()]
                if not dupe_match.empty:
                    dupe_info = dupe_match.iloc[0].to_dict()
                else:
                    dupe_info = {
                        'name': dupe,
                        'brand': 'Unknown',
                        'notes': '',
                        'price_range': 'unknown'
                    }
                dupes_info.append(dupe_info)
            
            return dupes_info
        
        # If no exact match, use similarity search
        # Vectorize the query
        query_vec = self.vectorizer.transform([fragrance_name])
        
        # Calculate similarity
        similarity = cosine_similarity(query_vec, self.features_matrix).flatten()
        
        # Get indices of top similar fragrances
        indices = similarity.argsort()[:-top_n-1:-1]
        
        # Get the similar fragrances
        similar_fragrances = self.df.iloc[indices].to_dict('records')
        
        # For each similar fragrance, add its known dupes
        result = []
        for frag in similar_fragrances:
            result.append(frag)
            
            # Add known dupes of this similar fragrance
            dupes_list = frag['dupes'].split(', ')
            for dupe in dupes_list[:2]:  # Limit to 2 dupes per similar fragrance
                dupe_match = self.df[self.df['name'].str.lower() == dupe.lower()]
                if not dupe_match.empty:
                    result.append(dupe_match.iloc[0].to_dict())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_result = []
        for item in result:
            if item['name'] not in seen:
                seen.add(item['name'])
                unique_result.append(item)
        
        return unique_result[:top_n]
    
    def get_all_fragrances(self):
        """Return a list of all fragrance names for autocomplete."""
        return self.df['name'].tolist()
