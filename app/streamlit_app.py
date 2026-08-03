"""
Streamlit Web Application
=========================
Interactive interface for drug-target binding prediction.
Uses the trained GNN model for real predictions.
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DrugTargetPredictor
from src.featurization import DrugFeaturizer, ProteinFeaturizer
from src.predict import BindingPredictor

st.set_page_config(
    page_title="Drug-Target Binding Predictor",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "checkpoints", "best_model.pt")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = DrugTargetPredictor(
        drug_encoder_config=config["model"]["drug_encoder"],
        protein_encoder_config=config["model"]["protein_encoder"],
        fusion_type=config["model"]["fusion"]["type"],
        fusion_dim=config["model"]["fusion"]["hidden_dim"]
    )

    # Handle PyG GATConv version mismatch in both directions:
    #   older PyG uses a single "lin.weight"
    #   newer PyG uses split "lin_src.weight" / "lin_dst.weight" (tied to the same value
    #   for a standard, non-bipartite GATConv, so this conversion is exact, not approximate)
    state = checkpoint["model_state_dict"]
    new_state = {}
    for k, v in state.items():
        if ".lin_src.weight" in k:
            base = k.replace(".lin_src.weight", "")
            new_state[f"{base}.lin.weight"] = v
        elif ".lin_dst.weight" in k:
            continue  # duplicate of lin_src, skip
        elif ".lin.weight" in k and "lin_src" not in k and "lin_dst" not in k:
            base = k.replace(".lin.weight", "")
            new_state[f"{base}.lin_src.weight"] = v
            new_state[f"{base}.lin_dst.weight"] = v
        else:
            new_state[k] = v
    state = new_state

    model.load_state_dict(state)
    model.eval()

    predictor = BindingPredictor(model, device="cpu")
    return predictor


def draw_molecule(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        return None, f"SMILES parsing error: {e}"
    if mol is None:
        return None, "Invalid SMILES string"
    try:
        AllChem.Compute2DCoords(mol)
        return Draw.MolToImage(mol, size=(400, 300)), None
    except Exception as e:
        return None, f"Structure rendering failed: {e}"


def classify_binding(kd_value: float):
    # pKd = -log10(Kd in M). HIGHER pKd = STRONGER binding.
    return BindingPredictor._classify_binding(kd_value)


def create_gauge(value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Binding Affinity (pKd)"},
        gauge={
            'axis': {'range': [4, 11], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [4, 5.5], 'color': "#e74c3c"},
                {'range': [5.5, 7], 'color': "#e67e22"},
                {'range': [7, 8.5], 'color': "#f1c40f"},
                {'range': [8.5, 11], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def main():
    st.markdown('<div class="main-header">Drug-Target Binding Predictor</div>', unsafe_allow_html=True)
    st.markdown("**GNN model trained on the Davis Kinase dataset (29,444 drug-target pairs)**")
    st.markdown("---")

    predictor = load_model()

    with st.sidebar:
        st.header("Settings")
        st.info("""
        **How to use:**
        1. Enter a drug SMILES string
        2. Enter a protein sequence
        3. Click **Predict**
        """)

        st.subheader("Example Drugs")
        examples = {
            "Imatinib (Gleevec)": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "Sorafenib": "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
            "Erlotinib": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
            "Gefitinib": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        }
        selected_example = st.selectbox("Quick fill:", list(examples.keys()))
        if st.button("Load Example"):
            st.session_state.drug_smiles = examples[selected_example]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Drug Input")
        if 'drug_smiles' not in st.session_state:
            st.session_state.drug_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"

        drug_smiles = st.text_area(
            "Enter drug SMILES:",
            value=st.session_state.drug_smiles,
            height=100
        )

        if drug_smiles:
            img, error_reason = draw_molecule(drug_smiles)
            if img:
                st.image(img, caption="Molecular Structure")
            elif error_reason == "Invalid SMILES string":
                st.warning("Could not parse SMILES.")
            else:
                st.info(f"Structure preview unavailable ({error_reason}), but prediction will still run normally.")

    with col2:
        st.subheader("Protein Input")
        if 'protein_seq' not in st.session_state:
            st.session_state.protein_seq = "MELLATGPQGASSCVPAAGQHFVVILGQGYGKVYKGEWVADANHLDFRESEQFQAFQEAELMAALGLHPHIVKIFHFYCGDLITMLVFEYCEMGSLDSYLHRKRRGALQDPYLVPTQGICKILSTILSQLKGHNLENPIDNLLDFGCRFEVQSSQSRGQSEVSEEFDEFNQACCSQSFQELWQTEEYGFGG"

        protein_sequence = st.text_area(
            "Enter protein sequence:",
            value=st.session_state.protein_seq,
            height=150
        )
        if protein_sequence:
            st.info(f"Length: {len(protein_sequence)} aa")

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("Predict Binding Affinity", type="primary", use_container_width=True)

    if predict_button:
        if not BindingPredictor._valid_protein(protein_sequence):
            valid_chars = sum(1 for c in protein_sequence if c in "ACDEFGHIKLMNPQRSTVWY")
            st.warning(
                f"Protein sequence has too few valid amino-acid characters "
                f"({valid_chars} of {len(protein_sequence)}). Expected 20 standard "
                f"amino acids (ACDEFGHIKLMNPQRSTVWY), so this prediction is not meaningful."
            )
            return
        with st.spinner("Running GNN model..."):
            result = predictor.predict(drug_smiles, protein_sequence)

        if "error" in result:
            st.error(result["error"])
            return

        kd_value = result["kd_value"]
        binding_label = classify_binding(kd_value)

        st.markdown("---")
        st.subheader("Prediction Results")

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Predicted pKd", f"{kd_value:.2f}")
        with col_r2:
            st.metric("Binding Strength", binding_label)
        with col_r3:
            st.metric("Protein Length", f"{len(protein_sequence)} aa")

        fig = create_gauge(kd_value)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Interpretation")
        kd_nM = 10 ** (9 - kd_value)
        if kd_value >= 8.5:
            interp = f"**Strong binding predicted** (pKd={kd_value:.2f}, ~{kd_nM:.2f} nM). The drug shows high affinity for this target."
        elif kd_value >= 7.0:
            interp = f"**Moderate binding predicted** (pKd={kd_value:.2f}, ~{kd_nM:.2f} nM). Promising candidate, may need optimization."
        elif kd_value >= 5.5:
            interp = f"**Weak binding predicted** (pKd={kd_value:.2f}, ~{kd_nM:.2f} nM). Structural modifications may improve affinity."
        else:
            interp = f"**Poor binding predicted** (pKd={kd_value:.2f}, ~{kd_nM:.2f} nM). Consider alternative scaffolds."
        st.markdown(interp)

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol:
                st.subheader("Drug Properties")
                pc1, pc2, pc3, pc4 = st.columns(4)
                with pc1:
                    st.metric("MW", f"{Descriptors.MolWt(mol):.0f}")
                with pc2:
                    st.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")
                with pc3:
                    st.metric("HBD", Descriptors.NumHDonors(mol))
                with pc4:
                    st.metric("HBA", Descriptors.NumHAcceptors(mol))
        except Exception:
            pass

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>Drug-Target Binding Predictor | GNN on Davis Kinase Dataset | Built by Haseeb Ur Rehman</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
