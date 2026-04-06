import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Group-level index (20K groups)
INDEX_FILE = "faiss_index.bin"
MAPPING_FILE = "group_mapping.pkl"

# Code-level index (71K specific codes)
CODE_INDEX_FILE = "code_index.bin"
CODE_MAPPING_FILE = "code_mapping.pkl"

MODEL_NAME = "FremyCompany/BioLORD-2023"


def load_data(csv_path="ICD10codes.csv"):
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["Group_Code", "Extension", "Specific_Code", "Specific_Description", "Specific_Description_2", "Group_Description"],
        dtype=str
    )
    df["Group_Description"] = df["Group_Description"].fillna("")
    return df


def get_unique_groups(df):
    unique_groups = df[["Group_Code", "Group_Description"]].drop_duplicates().reset_index(drop=True)
    return unique_groups


def build_or_load_index(csv_path="ICD10codes.csv"):
    df = load_data(csv_path)
    unique_groups = get_unique_groups(df)
    group_list = unique_groups.to_dict(orient="records")

    if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
        print("Loading existing FAISS group index and mappings...")
        index = faiss.read_index(INDEX_FILE)
        with open(MAPPING_FILE, "rb") as f:
            group_list = pickle.load(f)

        if index.d == 768:
            return index, group_list, df
        else:
            print("Detected old vector dimension. Forcing rebuild for local model...")

    print(f"Building new FAISS group index for {len(unique_groups)} groups using SentenceTransformers...")

    model = SentenceTransformer(MODEL_NAME)
    texts_to_embed = [f"{row['Group_Code']}: {row['Group_Description']}" for _, row in unique_groups.iterrows()]

    print("Computing group embeddings locally...")
    embeddings_np = model.encode(texts_to_embed, show_progress_bar=True, batch_size=256, convert_to_numpy=True).astype("float32")

    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)

    print(f"Saving FAISS group index of shape {embeddings_np.shape}...")
    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(group_list, f)

    return index, group_list, df


def build_or_load_code_index(csv_path="ICD10codes.csv"):
    """Build or load FAISS index over 71K specific ICD-10-CM codes.
    Uses cosine similarity (IndexFlatIP with normalized vectors)."""
    df = load_data(csv_path)

    if os.path.exists(CODE_INDEX_FILE) and os.path.exists(CODE_MAPPING_FILE):
        print("Loading existing FAISS code index...")
        code_index = faiss.read_index(CODE_INDEX_FILE)
        with open(CODE_MAPPING_FILE, "rb") as f:
            code_list = pickle.load(f)

        if code_index.d == 768:
            return code_index, code_list, df
        else:
            print("Detected old code index dimension. Forcing rebuild...")

    print(f"Building new FAISS code index for {len(df)} specific codes...")

    model = SentenceTransformer(MODEL_NAME)
    texts_to_embed = [
        f"{row['Specific_Code']}: {row['Specific_Description']}"
        for _, row in df.iterrows()
    ]

    print("Computing code-level embeddings (this may take a few minutes)...")
    embeddings_np = model.encode(
        texts_to_embed, show_progress_bar=True, batch_size=256,
        convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine similarity (IndexFlatIP on normalized vectors = cosine sim)
    faiss.normalize_L2(embeddings_np)

    dim = embeddings_np.shape[1]
    code_index = faiss.IndexFlatIP(dim)
    code_index.add(embeddings_np)

    # Build mapping list
    code_list = df[["Specific_Code", "Specific_Description", "Group_Code", "Group_Description"]].to_dict(orient="records")

    print(f"Saving FAISS code index of shape {embeddings_np.shape}...")
    faiss.write_index(code_index, CODE_INDEX_FILE)
    with open(CODE_MAPPING_FILE, "wb") as f:
        pickle.dump(code_list, f)

    return code_index, code_list, df


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    index, group_list, df = build_or_load_index()
    print("Group indexing complete! Total unique groups:", len(group_list))

    code_index, code_list, _ = build_or_load_code_index()
    print("Code indexing complete! Total specific codes:", len(code_list))
