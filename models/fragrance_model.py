import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

class FragranceModel:
    def __init__(self):
        """Initialize the FragranceModel by loading the dataset and preparing the vectorizer."""
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'all_merged_fragrances.csv')
        self.load_data()
        self.prepare_vectorizer()
        
    def load_data(self):
        """Load the fragrance dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} fragrances from dataset.")
            
            # Clean up data
            self.df['name'] = self.df['name'].str.lower()
            self.df['brand'] = self.df['brand'].str.lower()
            self.df['notes'] = self.df['notes'].fillna('').str.lower()
            
            # Create a combined field for better matching
            self.df['full_name'] = self.df['brand'] + ' ' + self.df['name']
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Create empty dataframe with required columns as fallback
            self.df = pd.DataFrame(columns=['name', 'brand', 'gender', 'rating_value', 'rating_count', 
                                           'perfumers', 'main_accords', 'notes', 'url', 'full_name'])
    
    def prepare_vectorizer(self):
        """Prepare the TF-IDF vectorizer for notes similarity."""
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), 
                                         min_df=2, stop_words='english')
        
        # Create a notes corpus and fit the vectorizer
        notes_corpus = self.df['notes'].tolist()
        if notes_corpus:
            self.notes_matrix = self.vectorizer.fit_transform(notes_corpus)
        else:
            self.notes_matrix = None
    
    def find_fragrance_index(self, query):
        """Find the index of a fragrance in the dataset based on a query string."""
        query = query.lower()
        
        # Try exact match on name first
        exact_matches = self.df[self.df['name'] == query]
        if not exact_matches.empty:
            return exact_matches.index[0]
        
        # Try exact match on full_name
        full_matches = self.df[self.df['full_name'] == query]
        if not full_matches.empty:
            return full_matches.index[0]
        
        # Try partial match on name
        name_matches = self.df[self.df['name'].str.contains(query, regex=False)]
        if not name_matches.empty:
            return name_matches.index[0]
        
        # Try partial match on full_name
        full_partial_matches = self.df[self.df['full_name'].str.contains(query, regex=False)]
        if not full_partial_matches.empty:
            return full_partial_matches.index[0]
        
        # If no matches, return None
        return None
    
    def find_dupes(self, fragrance_name, num_results=5):
        """
        Find similar fragrances (dupes) based on notes similarity.
        
        Args:
            fragrance_name: Name of the fragrance to find dupes for
            num_results: Number of similar fragrances to return
            
        Returns:
            List of dictionaries containing dupe information
        """
        if self.notes_matrix is None or self.df.empty:
            return []
        
        # Find the fragrance in our dataset
        fragrance_idx = self.find_fragrance_index(fragrance_name)
        
        if fragrance_idx is None:
            print(f"Fragrance '{fragrance_name}' not found in dataset.")
            return []
        
        # Get the original fragrance details
        original_fragrance = self.df.iloc[fragrance_idx]
        original_name = original_fragrance['name']
        original_brand = original_fragrance['brand']
        
        # Get the notes vector for this fragrance
        fragrance_vector = self.notes_matrix[fragrance_idx]
        
        # Calculate similarity with all other fragrances
        similarities = cosine_similarity(fragrance_vector, self.notes_matrix).flatten()
        
        # Create a list of (index, similarity) tuples
        similarity_tuples = [(i, similarities[i]) for i in range(len(similarities))]
        
        # Sort by similarity (descending)
        similarity_tuples.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out the original fragrance and any duplicates
        filtered_indices = []
        seen_names = set()
        
        for idx, sim in similarity_tuples:
            # Skip the original fragrance
            if idx == fragrance_idx:
                continue
                
            # Skip fragrances with the same name as the original
            current_name = self.df.iloc[idx]['name']
            current_brand = self.df.iloc[idx]['brand']
            
            # Skip exact matches to the original fragrance
            if current_name == original_name and current_brand == original_brand:
                continue
                
            # Skip if we've already seen this name (to avoid duplicates)
            if current_name in seen_names:
                continue
                
            # Add to our filtered list and mark as seen
            filtered_indices.append(idx)
            seen_names.add(current_name)
            
            # Stop once we have enough results
            if len(filtered_indices) >= num_results:
                break
        
        # If no similar fragrances found
        if not filtered_indices:
            return []
        
        # Prepare the results
        dupes = []
        for idx in filtered_indices:
            dupe = self.df.iloc[idx]
            similarity_score = similarities[idx]
            
            # Assign a price range (this is a placeholder - in a real app, you'd have actual price data)
            # Here we're just using rating_count as a proxy for popularity/price
            if dupe['rating_count'] > 1000:
                price_range = "premium"
            elif dupe['rating_count'] > 500:
                price_range = "mid-range"
            else:
                price_range = "budget"
                
            dupes.append({
                'name': dupe['name'].title(),
                'brand': dupe['brand'].title(),
                'notes': dupe['notes'],
                'similarity': f"{similarity_score:.2f}",
                'price_range': price_range,
                'gender': dupe['gender'],
                'dupes': original_fragrance['name'].title()
            })
        
        return dupes
