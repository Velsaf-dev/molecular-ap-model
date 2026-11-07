"""
Module: ChemBERTa Embedding Extractor.
Loads a pretrained molecular transformer (DeepChem/ChemBERTa-77M-MTR or seyonec/ChemBERTa-zinc-base-v1)
and extracts 768-dimensional dense representations for SMILES strings using GPU acceleration.
"""

import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
FALLBACK_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBERTaExtractor:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, batch_size=32, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"ChemBERTaExtractor initialized on device: {self.device}")

        logger.info(f"Loading tokenizer and model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            logger.warning(f"Failed loading {self.model_name}: {e}. Trying fallback: {FALLBACK_MODEL_NAME}")
            self.model_name = FALLBACK_MODEL_NAME
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        self.model.eval()

    def extract_embeddings(self, smiles_list, pooling="mean"):
        """
        Extracts 768-dim embeddings for a list of SMILES.
        pooling: 'mean' (attention-weighted average) or 'cls' ([CLS] token)
        """
        embeddings = []
        n_total = len(smiles_list)
        logger.info(f"Extracting ChemBERTa embeddings for {n_total} molecules (pooling={pooling}, batch_size={self.batch_size})...")

        for i in range(0, n_total, self.batch_size):
            batch_smiles = [str(s).strip() if pd.notnull(s) else "C" for s in smiles_list[i : i + self.batch_size]]
            inputs = self.tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # (B, L, 768)

                if pooling == "cls":
                    batch_embeds = last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    # Mean pooling weighted by attention mask
                    mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    batch_embeds = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings.append(batch_embeds)

        embeddings_matrix = np.vstack(embeddings).astype(np.float32)
        logger.info(f"ChemBERTa feature matrix shape: {embeddings_matrix.shape}")
        return embeddings_matrix


def run_embedding_extraction(
    train_csv="data/processed/train_clean.csv",
    test_csv="data/processed/test_clean.csv",
    out_dir="data/processed"
):
    os.makedirs(out_dir, exist_ok=True)
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    extractor = ChemBERTaExtractor()

    logger.info("Extracting embeddings for train set...")
    train_embeds = extractor.extract_embeddings(df_train["clean_smiles"].tolist())
    train_out_path = os.path.join(out_dir, "train_chemberta_embeddings.npy")
    np.save(train_out_path, train_embeds)
    logger.info(f"Saved {train_out_path}")

    logger.info("Extracting embeddings for test set...")
    test_embeds = extractor.extract_embeddings(df_test["clean_smiles"].tolist())
    test_out_path = os.path.join(out_dir, "test_chemberta_embeddings.npy")
    np.save(test_out_path, test_embeds)
    logger.info(f"Saved {test_out_path}")

    return train_out_path, test_out_path


if __name__ == "__main__":
    run_embedding_extraction()
