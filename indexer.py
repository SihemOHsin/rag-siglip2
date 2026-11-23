"""Indexation : PDF -> images + fiches -> embeddings SigLIP2 -> base Chroma.

PDF -> chunking (une image = un chunk) -> embeddings SigLIP2 -> Chroma

Utilisation :
    uv run indexer.py

À relancer seulement si le catalogue change.
"""

import os
import re

import pymupdf
import torch
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
)


# =========================================================
# PARAMÈTRES
# =========================================================

PDF_PATH = "catalogue_EcomNet.pdf"
IMG_DIR = "catalogue_images"
CHROMA_DIR = "chroma_db"
COLLECTION = "catalogue"

MODEL_ID = "google/siglip2-base-patch16-224"

MIN_SIDE = 80
DPI = 150

REF_RE = re.compile(r"\b\d{7}\b")


# =========================================================
# FONCTIONS UTILISÉES PAR LE CHUNKING
# =========================================================

def nettoyer(texte):
    return re.sub(r"\s+", " ", texte).strip()


def a_du_texte(doc, i):
    """Vrai si la page i contient du texte
    en plus des références.
    """

    return (
        0 <= i < len(doc)
        and len(
            nettoyer(
                REF_RE.sub(
                    "",
                    doc[i].get_text()
                )
            )
        ) > 80
    )


def page_fiche(doc, i):
    """Page qui décrit le produit.

    La page elle-même si elle contient du texte,
    sinon l'autre page de la double page.
    """

    if a_du_texte(doc, i):
        return i

    partenaire = (
        i + 1
        if i % 2 == 1
        else i - 1
    )

    for k in (
        partenaire,
        i + 1,
        i - 1
    ):
        if a_du_texte(doc, k):
            return k

    return i


def titre_produit(page):
    """Le texte en plus gros caractères
    = nom du produit.
    """

    spans = [
        s
        for b in page.get_text("dict")["blocks"]
        for l in b.get("lines", [])
        for s in l["spans"]
        if s["text"].strip()
    ]

    if not spans:
        return ""

    taille_max = max(
        s["size"]
        for s in spans
    )

    return nettoyer(
        " ".join(
            s["text"]
            for s in spans
            if s["size"] >= 0.6 * taille_max
        )
    )


def extraire_images(
    doc,
    img_dir=IMG_DIR
):
    """PDF ouvert -> liste de chunks.

    Un chunk =
    une image + sa fiche produit.
    """

    os.makedirs(
        img_dir,
        exist_ok=True
    )

    chunks = []

    for i, page in enumerate(doc):

        for j, img in enumerate(
            page.get_images(full=True)
        ):

            try:
                bbox = (
                    page.get_image_bbox(img)
                    & page.rect
                )

            except Exception:
                continue

            if (
                bbox.width < MIN_SIDE
                or bbox.height < MIN_SIDE
            ):
                continue

            chemin = (
                f"{img_dir}/"
                f"p{i + 1:03d}_{j:02d}.png"
            )

            page.get_pixmap(
                clip=bbox,
                dpi=DPI
            ).save(
                chemin
            )

            autour = page.get_text(
                clip=bbox + (
                    -30,
                    -30,
                    30,
                    30
                )
            )

            k = page_fiche(
                doc,
                i
            )

            chunks.append(
                {
                    "image": chemin,

                    "page": i + 1,

                    "references": ", ".join(
                        sorted(
                            set(
                                REF_RE.findall(
                                    autour
                                )
                            )
                        )
                    ),

                    "texte_autour":
                        nettoyer(autour)[:300],

                    "page_fiche": k + 1,

                    "produit":
                        titre_produit(
                            doc[k]
                        ),

                    "fiche":
                        nettoyer(
                            doc[k].get_text()
                        )[:1500],
                }
            )

    return chunks


# =========================================================
# EMBEDDINGS SIGLIP2
# =========================================================

class SigLIP2Embeddings(
    Embeddings
):
    """Transforme images et textes
    en vecteurs SigLIP2 normalisés.
    """

    def __init__(
        self,
        model_id=MODEL_ID
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = (
            AutoModel
            .from_pretrained(
                model_id
            )
            .to(
                self.device
            )
            .eval()
        )

        self.processor = (
            AutoImageProcessor
            .from_pretrained(
                model_id
            )
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_id
            )
        )


    @staticmethod
    def _en_liste(sortie):
        """Normalise le vecteur."""

        v = (
            sortie
            if torch.is_tensor(sortie)
            else sortie.pooler_output
        )

        return (
            torch.nn.functional
            .normalize(
                v,
                dim=-1
            )
            .cpu()
            .tolist()
        )


    @torch.no_grad()
    def embed_image(
        self,
        uris,
        batch=16
    ):
        """Chemins d'images -> vecteurs."""

        vecteurs = []

        for k in range(
            0,
            len(uris),
            batch
        ):

            lot = [
                (
                    Image.open(u)
                    if isinstance(u, str)
                    else u
                ).convert("RGB")

                for u
                in uris[
                    k:k + batch
                ]
            ]

            entrees = (
                self.processor(
                    images=lot,
                    return_tensors="pt"
                )
                .to(
                    self.device
                )
            )

            vecteurs += self._en_liste(
                self.model
                .get_image_features(
                    **entrees
                )
            )

        return vecteurs


    @torch.no_grad()
    def embed_documents(
        self,
        texts
    ):
        """Textes -> vecteurs dans
        le même espace que les images.
        """

        entrees = (
            self.tokenizer(
                texts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt"
            )
            .to(
                self.device
            )
        )

        return self._en_liste(
            self.model
            .get_text_features(
                **entrees
            )
        )


    def embed_query(
        self,
        text
    ):
        return self.embed_documents(
            [text]
        )[0]


# =========================================================
# OUVRIR LA BASE CHROMA
# =========================================================

def ouvrir_base(
    embeddings
):
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # 1. LOAD PDF
    # =====================================================

    doc = pymupdf.open(
        PDF_PATH
    )

    print(
        "Pages chargées :",
        len(doc)
    )

    print(
        "Extrait :",
        nettoyer(
            doc[0].get_text()
        )[:200]
    )


    # =====================================================
    # 2. CHUNKING
    # =====================================================

    chunks = extraire_images(
        doc
    )

    avec_ref = sum(
        1
        for c in chunks
        if c["references"]
    )

    print(
        f"Chunks : "
        f"{len(chunks)} images "
        f"({avec_ref} avec référence)"
    )


    # =====================================================
    # 3. EMBEDDINGS
    # =====================================================

    embeddings = (
        SigLIP2Embeddings()
    )


    # =====================================================
    # 4. VECTOR STORE
    # =====================================================

    vectorstore = ouvrir_base(
        embeddings
    )

    vectorstore.reset_collection()

    lot = 32

    for k in tqdm(
        range(
            0,
            len(chunks),
            lot
        ),
        desc="Indexation"
    ):

        morceau = chunks[
            k:k + lot
        ]

        vectorstore.add_images(
            uris=[
                c["image"]
                for c in morceau
            ],

            metadatas=morceau,

            ids=[
                os.path.basename(
                    c["image"]
                )
                for c in morceau
            ]
        )


    print(
        "Base Chroma créée ✔"
    )


    # =====================================================
    # 5. SAUVEGARDE LOCALE
    # =====================================================

    total = len(
        vectorstore.get(
            include=[]
        )["ids"]
    )

    print(
        f"Base Chroma : "
        f"{total} images indexées "
        f"-> {CHROMA_DIR}/"
    )


if __name__ == "__main__":
    main()