import pandas as pd

# Load the Excel file
df = pd.read_excel("/srv/storage/custom-tools/Adhan/prayer_times_2026_raw.xlsx", header=None)

# Flatten the DataFrame row-wise
flattened_data = df.values.flatten()

# Convert it into a new DataFrame with a single column
df_flattened = pd.DataFrame(flattened_data, columns=["Values"])

# Save the modified file
df_flattened.to_excel("/srv/storage/custom-tools/Adhan/prayer_times_2026_flattened.xlsx", index=False)

print("Conversion complete! Saved as 'prayer_times_2026_flattened.xlsx'")
