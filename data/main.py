import pandas as pd
import csv

# --------- Load fra_cleaned.csv ---------
fra_cleaned = pd.read_csv("fra_cleaned.csv", sep=";", encoding="ISO-8859-1")

fra_cleaned = fra_cleaned.rename(columns={
    "Perfume": "name",
    "Brand": "brand",
    "Country": "country",
    "Gender": "gender",
    "Rating Value": "rating_value",
    "Rating Count": "rating_count",
    "Year": "year",
    "Top": "top_notes",
    "Middle": "middle_notes",
    "Base": "base_notes",
    "Perfumer1": "perfumers",
    "Perfumer2": "perfumer2",
    "mainaccord1": "accord1",
    "mainaccord2": "accord2",
    "mainaccord3": "accord3",
    "mainaccord4": "accord4",
    "mainaccord5": "accord5",
    "url": "url"
})

# Combine perfumers
fra_cleaned["perfumers"] = fra_cleaned["perfumers"].fillna('') + ";" + fra_cleaned["perfumer2"].fillna('')
fra_cleaned.drop(columns=["perfumer2"], inplace=True)

# Combine accords
fra_cleaned["main_accords"] = fra_cleaned[["accord1", "accord2", "accord3", "accord4", "accord5"]].apply(
    lambda row: [a.strip() for a in row if pd.notnull(a)], axis=1)
fra_cleaned.drop(columns=["accord1", "accord2", "accord3", "accord4", "accord5"], inplace=True)

# Combine notes
fra_cleaned["notes"] = fra_cleaned[["top_notes", "middle_notes", "base_notes"]].fillna('').agg('; '.join, axis=1)
fra_cleaned.drop(columns=["top_notes", "middle_notes", "base_notes"], inplace=True)

# --------- Load fra_perfumes.csv with delimiter auto-detection ---------
with open("fra_perfumes.csv", "r", encoding="ISO-8859-1") as f:
    dialect = csv.Sniffer().sniff(f.read(1000))
    f.seek(0)

fra_perfumes = pd.read_csv("fra_perfumes.csv", delimiter=dialect.delimiter, encoding="ISO-8859-1")

# Rename columns manually
fra_perfumes.columns = [
    "name", "gender", "rating_value", "rating_count",
    "main_accords", "perfumers", "description", "url"
]

# Clean/convert types
fra_perfumes["main_accords"] = fra_perfumes["main_accords"].apply(
    lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [])

# Extract brand from name
fra_perfumes["brand"] = fra_perfumes["name"].str.extract(r"^(\w+)", expand=False)

# Extract top notes from description
fra_perfumes["notes"] = fra_perfumes["description"].str.extract(r"Top notes are (.*?)[.;]", expand=False)
fra_perfumes.drop(columns=["description"], inplace=True)

# --------- Merge and clean both ---------
common_cols = ["name", "brand", "gender", "rating_value", "rating_count", "perfumers", "main_accords", "notes", "url"]
fra_cleaned = fra_cleaned[common_cols]
fra_perfumes = fra_perfumes[common_cols]

merged = pd.concat([fra_cleaned, fra_perfumes], ignore_index=True)

# Deduplicate
merged = merged.drop_duplicates(subset=["url"], keep="first")
merged = merged.drop_duplicates(subset=["name", "brand"], keep="first")

# Cleanup
merged["rating_value"] = pd.to_numeric(merged["rating_value"], errors='coerce').fillna(0)
merged["rating_count"] = pd.to_numeric(merged["rating_count"], errors='coerce').fillna(0).astype(int)
merged["perfumers"] = merged["perfumers"].fillna("unknown")
merged["main_accords"] = merged["main_accords"].apply(lambda x: x if isinstance(x, list) else [])
merged["notes"] = merged["notes"].fillna("unknown")
merged["url"] = merged["url"].fillna("")

# --------- Save the final dataset ---------
merged.to_csv("merged_cleaned_fragrances.csv", index=False)
print("✅ Clean file saved as 'merged_cleaned_fragrances.csv'")
