
import streamlit as st
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBClassifier, XGBRegressor

# ===============================

# CONFIG

# ===============================

st.set_page_config(layout="wide")
st.title("🧪 AI Drug Discovery Platform")

# Use a direct path for model loading in Colab, assuming models are in Google Drive
BASE_DIR = os.path.dirname(file)

# ===============================

# SAFE MODEL LOADER

# ===============================

def load_xgb_model(filename, model_type="classifier"):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        st.warning(f"Missing: {filename} at {path}")
        return None

    if model_type == "classifier":
        model = XGBClassifier()
    else:
        model = XGBRegressor()

    model.load_model(path)
    return model

# ===============================

# FINGERPRINT + DESCRIPTORS

# ===============================

morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_descriptors(mol):
    from rdkit.Chem import Descriptors
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol)
    ]

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.array(morgan.GetFingerprint(mol))
    desc = np.array(get_descriptors(mol))
    return np.concatenate([fp, desc])

# ===============================

# LOAD MODELS

# ===============================

tox_model = load_xgb_model("tox_model.json", "classifier")
compat_model = load_xgb_model("compatibility_xgb.json", "classifier")

targets = ["SERT","DAT","D2","D3","D4","5HT1A","5HT6","5HT7"]
hybrid_models = {}

for name in targets:
    reg = load_xgb_model(f"{name}_reg.json", "regressor")
    clf = load_xgb_model(f"{name}_clf.json", "classifier")

    if reg and clf:
        hybrid_models[name] = (reg, clf)

# ===============================

# EXCIPIENTS

# ===============================

excipients = {
    "Lactose": "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](CO)O[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O",
    "PEG": "OCCO",
    "PVP": "C=CC(=O)N1CCCC1",
    "HPMC": "COC1=CC=CC=C1O",
    "Ethanol": "CCO",
    "Sodium Benzoate": "C1=CC=C(C=C1)C(=O)[O-].[Na+]",
    "Sucrose": "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](CO)O[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O",
    "Glycerol": "C(C(CO)O)O",
    "Propylene Glycol": "CC(CO)O",
    "Mannitol": "C(C(C(C(C(CO)O)O)O)O)O",
    "Citric Acid": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
    "Polysorbate 80": "CC(C)OC(=O)CCCCCCCCCCCCC",
    "Starch": "C(C1C(C(C(C(O1)O)O)O)O)O"
}

# ===============================

# PREDICTIONS

# ===============================

def predict_ic50(smiles):
    features = smiles_to_features(smiles)
    if features is None:
        return None

    features = features.reshape(1,-1)
    results = {}

    for name, (reg, clf) in hybrid_models.items():
        # Check if models for the target are loaded
        if reg is None or clf is None:
            results[name] = {"Active": False, "Confidence": 0.0, "Error": "Model not loaded"}
            continue

        prob = clf.predict_proba(features)[0][1]

        if prob > 0.5:
            pic50 = reg.predict(features)[0]
            ic50 = 10**(-pic50) * 1e9
            results[name] = {"Active": True, "Confidence": round(prob,2), "pIC50": round(pic50,2), "IC50_nM": round(ic50,2)}
        else:
            results[name] = {"Active": False, "Confidence": round(prob,2)}

    return results

def predict_toxicity(smiles):
    if tox_model is None:
        return "Model missing"

    features = smiles_to_features(smiles)
    if features is None:
        return "Invalid SMILES"

    pred = tox_model.predict(features.reshape(1,-1))[0]
    return "High" if pred==1 else "Low"

def predict_compatibility(drug, excipient):
    if compat_model is None:
        return "Model missing", 0

    f1 = smiles_to_features(drug)
    f2 = smiles_to_features(excipient)

    if f1 is None or f2 is None:
        return "Invalid", 0

    features = np.concatenate([f1,f2]).reshape(1,-1)
    prob = compat_model.predict_proba(features)[0][1]

    return ("Compatible" if prob>0.5 else "Incompatible"), prob

def best_excipient(smiles):
    best_name = None
    best_score = -1

    for name, smi in excipients.items():
        _, prob = predict_compatibility(smiles, smi)
        if prob > best_score:
            best_score = prob
            best_name = name

    return best_name, best_score

# ===============================

# UI

# ===============================

smiles = st.text_input("Enter Drug SMILES")
excipient = st.selectbox("Select Excipient", list(excipients.keys()))

if st.button("Run Prediction"):

    st.subheader("📊 Results")

    # IC50
    st.write("### IC50 (Multi-target)")
    ic50 = predict_ic50(smiles)

    if ic50:
        for t,res in ic50.items():
            if res.get("Active"):
                st.success(f"{t}: Active | pIC50={res['pIC50']} | IC50={res['IC50_nM']} nM | Confidence={res['Confidence']}")
            elif res.get("Error"):
                st.error(f"{t}: {res['Error']}")
            else:
                st.error(f"{t}: Inactive | Confidence={res['Confidence']}")

    # Toxicity
    st.write("### Toxicity")
    st.success(predict_toxicity(smiles))

    # Compatibility
    st.write("### Compatibility")
    comp, prob = predict_compatibility(smiles, excipients[excipient])
    st.write(f"{comp} (Confidence: {prob:.2f})")

    # Best excipient
    st.write("### Best Excipient")
    best, score = best_excipient(smiles)
    st.success(f"{best} (Score: {score:.2f})")
    