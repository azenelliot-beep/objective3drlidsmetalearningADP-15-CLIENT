# BLOCK 4 (UPDATED v4): Load and explore the dataset - picks the ToN-IoT CSV by its COLUMNS, not by its file name
import glob
DATA_DIR = "/kaggle/input/datasets/fadiabuzwayed/ton-iot-train-test-network"   # dataset folder on Kaggle
NEEDED = {"type", "label", "proto", "conn_state"}                               # columns of the ToN-IoT network file

def _tables_under(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True) +
                  glob.glob(os.path.join(folder, "**", "*.CSV"), recursive=True))

def _is_ton_iot(path):
    try:
        return NEEDED <= set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False

# 1) the dataset folder above, 2) anything else mounted under /kaggle/input (in case Kaggle mounts it elsewhere)
candidates = _tables_under(DATA_DIR) or _tables_under("/kaggle/input")
matches = [p for p in candidates if _is_ton_iot(p)]
if not matches:
    print("Folder exists:", os.path.isdir(DATA_DIR))
    for root, _, files in os.walk("/kaggle/input"):
        print(root, files[:20])
    raise FileNotFoundError(
        f"No CSV with columns {sorted(NEEDED)} found in {DATA_DIR} or under /kaggle/input. "
        "Attach the dataset: right panel -> Add Input -> search 'ton-iot-train-test-network' -> Add.")
file_path = max(matches, key=os.path.getsize)            # the full network file if several splits are present
print("Dataset file:", file_path)

df = pd.read_csv(file_path)

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
