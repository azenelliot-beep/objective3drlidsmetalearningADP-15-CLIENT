# BLOCK 1 (COLAB): Environment - Google Colab (use Runtime -> Change runtime type -> T4 GPU)
!pip install -q "opacus==1.6.0"

import importlib, json, sys
versions = {"python": sys.version.split()[0]}
for pkg in ["numpy", "sklearn", "imblearn", "torch", "pandas", "opacus"]:
    versions[pkg] = importlib.import_module(pkg).__version__
print(json.dumps(versions, indent=2))
json.dump(versions, open("env_versions.json", "w"), indent=2)   # recorded for the thesis (Colab versions differ from Kaggle)
