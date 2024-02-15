import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Load data from CSV file
file_path = 'combined_topic_predictions.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Create confusion matrix
conf_matrix = confusion_matrix(df['label'], df['prediction'])

# Calculate F1 score for each class
f1_score = f1_score(df['label'], df['prediction'], average='weighted')
print(f1_score)

# Normalize confusion matrix
conf_matrix_normalized = normalize(conf_matrix, axis=1, norm='l1')

# Plot normalized confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=sorted(df['prediction'].unique()), yticklabels=sorted(df['label'].unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")

# Save the normalized confusion matrix plot
plt.savefig('normalized_confusion_matrix_plot.png')  # You can change the filename and format as needed
plt.show()
