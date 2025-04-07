import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load the dataset
file_path = "ClassificationModel/Extract/Extract_output/Plant2.csv"
df = pd.read_csv(file_path)

# ✅ Extract numeric features (exclude non-numeric and Label column)
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Label' in numeric_features:
    numeric_features.remove('Label')  # Just in case Label is numeric

# ✅ Auto-detect label column and treat as string
df['Label'] = df['Label'].astype(str)

# Init PDF path
Visualization_path = "ClassificationModel/Plant_Visualization2.pdf"

# Log summary
log_data = []
log_data.append("CSV Overview:")
log_data.append(f"Columns: {df.columns.tolist()}")
log_data.append(f"Number of rows: {len(df)}")
log_data.append("Data Types:")
log_data.append(str(df.dtypes))
log_data.append("First 5 rows:")
log_data.append(str(df.head()))

log_file = "ClassificationModel/VisualLog2.txt"
with open(log_file, "w", encoding="utf-8") as f:
    f.write("\n".join(log_data))
print(f"Log saved to {log_file}")

# 📌 Visualization
with PdfPages(Visualization_path) as pdf:
    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Label', data=df, palette="coolwarm", order=df['Label'].value_counts().index)
    plt.title("Class Distribution (Label)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    pdf.savefig()
    plt.close()

    # Boxplots for all features
    for feature in numeric_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Label', y=feature, data=df)
        plt.title(f"Feature: {feature} by Label")
        plt.xlabel("Label")
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        pdf.savefig()
        plt.close()

print(f"PDF report saved as {Visualization_path}")
