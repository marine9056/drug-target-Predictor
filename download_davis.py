"""
Download Full Davis Dataset (29,444 pairs)
==========================================
Run this script to download the complete Davis kinase dataset.
"""

import pandas as pd
import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/dingyan20/Davis-Dataset-for-DTA-Prediction/main"

FILES = {
    "drugs.csv": f"{BASE_URL}/drugs.csv",
    "proteins.csv": f"{BASE_URL}/proteins.csv",
    "drug_protein_affinity.csv": f"{BASE_URL}/drug_protein_affinity.csv",
}

def download():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Downloading Davis Dataset (29,444 drug-target pairs)")
    print("=" * 50)

    # Download each file
    for filename, url in FILES.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"[OK] {filename} already exists")
            continue

        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"[OK] Saved to {filepath}")
        else:
            print(f"[FAIL] Could not download {filename}: HTTP {response.status_code}")
            return

    # Now combine into single training file
    print("\nCombining into single dataset...")

    drugs = pd.read_csv(data_dir / "drugs.csv")
    proteins = pd.read_csv(data_dir / "proteins.csv")
    affinity = pd.read_csv(data_dir / "drug_protein_affinity.csv")

    print(f"  Drugs: {len(drugs)}")
    print(f"  Proteins: {len(proteins)}")
    print(f"  Affinity pairs: {len(affinity)}")

    # Merge: get SMILES for each drug, sequence for each protein
    drug_smiles = drugs.set_index("Drug_Index")["Canonical_SMILES"].to_dict()
    protein_seq = proteins.set_index("Protein_Index")["Sequence"].to_dict()

    affinity["drug_smiles"] = affinity["Drug_Index"].map(drug_smiles)
    affinity["protein_sequence"] = affinity["Protein_Index"].map(protein_seq)

    # Affinity is already in pKd format (-log10(Kd in Molar))
    # Values typically range from 5.0 (weak) to 9.0 (strong binding)
    affinity["kd_value"] = affinity["Affinity"]

    # Keep only needed columns
    df = affinity[["drug_smiles", "protein_sequence", "kd_value"]].dropna()
    df = df.reset_index(drop=True)

    # Save
    output = data_dir / "davis_full.csv"
    df.to_csv(output, index=False)

    print(f"\n{'=' * 50}")
    print(f"SUCCESS! Dataset ready for training")
    print(f"{'=' * 50}")
    print(f"  Total pairs: {len(df)}")
    print(f"  Unique drugs: {df['drug_smiles'].nunique()}")
    print(f"  Unique proteins: {df['protein_sequence'].nunique()}")
    print(f"  pKd range: {df['kd_value'].min():.2f} - {df['kd_value'].max():.2f}")
    print(f"  Saved to: {output}")
    print(f"\nNext step: python src/train.py --config configs/default.yaml")


if __name__ == "__main__":
    download()
