import pandas as pd

# --------- Load CSV: merged_cleaned_fragrances.csv ---------
main_df = pd.read_csv("merged_cleaned_fragrances.csv")

# --------- Load Excel: perfume_database_cleaned.xlsx ---------
excel_df = pd.read_excel("perfume_database_cleaned.xlsx")

# --------- Rename Excel columns to match main dataset ---------
excel_df = excel_df.rename(columns={
    "perfume": "name",
    "brand": "brand",
    "notes": "notes"
})

# --------- Add missing columns with default values ---------
excel_df["gender"] = "unknown"
excel_df["rating_value"] = 0
excel_df["rating_count"] = 0
excel_df["perfumers"] = "unknown"
excel_df["main_accords"] = [[] for _ in range(len(excel_df))]
excel_df["url"] = ""

# --------- Reorder Excel columns to match main_df ---------
common_cols = ["name", "brand", "gender", "rating_value", "rating_count", "perfumers", "main_accords", "notes", "url"]
excel_df = excel_df[common_cols]
main_df = main_df[common_cols]

# --------- Concatenate both datasets ---------
merged_df = pd.concat([main_df, excel_df], ignore_index=True)

# --------- Remove duplicates by name + brand ---------
merged_df = merged_df.drop_duplicates(subset=["name", "brand"], keep="first")

# --------- Final cleanup ---------
merged_df["rating_value"] = pd.to_numeric(merged_df["rating_value"], errors='coerce').fillna(0)
merged_df["rating_count"] = pd.to_numeric(merged_df["rating_count"], errors='coerce').fillna(0).astype(int)
merged_df["perfumers"] = merged_df["perfumers"].fillna("unknown")
merged_df["main_accords"] = merged_df["main_accords"].apply(lambda x: x if isinstance(x, list) else [])
merged_df["notes"] = merged_df["notes"].fillna("unknown")
merged_df["url"] = merged_df["url"].fillna("")

# --------- Save the final merged dataset ---------
merged_df.to_csv("all_merged_fragrances.csv", index=False)
print("✅ Final file saved as 'all_merged_fragrances.csv'")
