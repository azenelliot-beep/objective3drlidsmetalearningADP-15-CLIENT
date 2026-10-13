# BLOCK 4 (UPDATED v4): Load and explore the dataset - finds the CSV wherever Kaggle mounted it
import glob
DATA_FILE = "TON_IoT_Train_Test_Network.csv"
_hits = sorted(glob.glob(f"/kaggle/input/**/{DATA_FILE}", recursive=True))
if not _hits:                                   # fall back: any CSV whose name looks like the ToN-IoT network file
    _hits = sorted(p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
                   if "ton" in os.path.basename(p).lower() and "network" in os.path.basename(p).lower())
if not _hits:
    raise FileNotFoundError(
        f"{DATA_FILE} not found under /kaggle/input. Add the dataset with 'Add Input' (search: ton-iot-train-test-network). "
        f"CSV files currently mounted: {glob.glob('/kaggle/input/**/*.csv', recursive=True)}")
file_path = _hits[0]
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
