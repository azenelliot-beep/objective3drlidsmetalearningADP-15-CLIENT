# BLOCK 4 (UPDATED - COLAB): Load and explore the dataset from Google Drive
file_path = "/content/drive/MyDrive/Elham/TON_IoT_Train_Test_Network.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path} - check the Drive folder name and that Drive is mounted.")
print("Dataset file:", file_path)

df = pd.read_csv(file_path)
assert "type" in df.columns, f"{file_path} has no 'type' column - wrong file?"

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values (total):", int(df.isnull().sum().sum()))

# Class distribution - computed, not hardcoded
label_distribution = df['type'].value_counts()
print("\nClass distribution:\n", label_distribution)
normal_pct = 100 * (df['type'] == 'normal').mean()
print(f"\nNormal: {normal_pct:.1f}% | Attack: {100 - normal_pct:.1f}%")

plt.figure(figsize=(8, 5))
label_distribution.plot(kind='bar')
plt.title('Class Distribution (type)')
plt.xlabel('Class'); plt.ylabel('Count')
plt.tight_layout(); plt.show()
