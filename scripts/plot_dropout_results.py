import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing the .tsv files
tsv_dir = "models"
dropout_values = [0.0, 0.2, 0.3, 0.4, 0.5]

# Dictionary to store validation PPL from each dropout run
results = {}

for dropout in dropout_values:
    filename = f"dropout{dropout}.tsv"
    filepath = os.path.join(tsv_dir, filename)

    df = pd.read_csv(filepath, sep="\t")

    # Keep only numeric epochs (exclude "Test" row for now)
    df = df[df["Epoch"].apply(lambda x: str(x).isdigit())]
    df["Epoch"] = df["Epoch"].astype(int)
    df = df.drop_duplicates(subset="Epoch")
    results[f"Dropout {dropout}"] = df.set_index("Epoch")["ValidPPL"]

# Combine into a single DataFrame
all_epochs = sorted(set().union(*[r.index for r in results.values()]))
combined = pd.DataFrame(index=all_epochs)


for label, series in results.items():
    combined[label] = series
    
combined.index = [f"Epoch {e}" for e in combined.index]
combined.index.name = "Valid.perplexity"

# Print Markdown table (first 10 rows)
print("### 📊 Validation Perplexity per Epoch")
print(combined.head(10).to_markdown())

# Save table as CSV
combined.to_csv("dropout_perplexity_table.csv")

combined.index = combined.index.astype(str).str.extract(r'(\d+)').iloc[:, 0].astype(int)
# Plotting
plt.figure(figsize=(10, 6))
for column in combined.columns:
    plt.plot(combined.index, combined[column], label=column)
plt.xlabel("Epoch")
plt.ylabel("Validation Perplexity")
plt.title("Validation Perplexity per Dropout Setting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dropout_perplexity_plot.png")
print("\n✅ Plot saved as 'dropout_perplexity_plot.png'")

