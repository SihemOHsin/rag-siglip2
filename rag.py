"""RAG visuel : la base Chroma (créée par indexer.py) pour retrouver,
Gemini pour répondre.

Photo ou phrase -> embedding SigLIP2 -> Chroma -> images proches -> Gemini
-> réponse

Utilisation (après uv run indexer.py) :
    uv run rag.py image.jpg
    uv run rag.py image.jpg --gemini
    uv run rag.py image.jpg "Est-ce que c'est en promotion ?"
    uv run rag.py --texte "une bouteille d'huile d'olive"

Depuis un autre programme :
    from rag import rechercher, rechercher_texte, repondre
"""

import argparse
import base64
import io

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

from indexer import SigLIP2Embeddings, ouvrir_base


load_dotenv(override=True)


# =========================================================
# PARAMÈTRES
# =========================================================

LLM = "gemini-3.5-flash-lite"

SEUIL_EXISTE = 0.90
SEUIL_RESSEMBLE = 0.75


CONSIGNES = """Tu es l'assistant d'un magasin. On te donne :
- IMAGE 1 : la photo envoyée par le client ;
- IMAGE 2 : l'image du catalogue retrouvée par le moteur de recherche ;
- la fiche du catalogue qui accompagne l'image 2.

Réponds en français, en suivant ce plan :

1. Description :
   décris le produit de l'image 2
   (type de produit, emballage, couleurs, textes et marque lisibles).

2. Correspondance :
   dis si l'image 1 montre bien le même produit que l'image 2,
   ou seulement un produit proche, et pourquoi
   (marque, format, couleur, contenance).

3. Informations du catalogue :
   nom, référence, prix, remise et période de l'offre,
   uniquement d'après la fiche.

4. Si le client pose une question, réponds-y à la fin.

Règles :
prix, références et offres viennent uniquement de la fiche,
jamais de l'image.

Si une information manque, dis-le.
N'invente rien.
"""


# Chargés une seule fois
_embeddings = None
_vectorstore = None
_llm = None


# =========================================================
# 1. CHARGEMENT DE LA BASE CHROMA
# =========================================================

def charger_base():
    """Retourne embeddings + vectorstore."""

    global _embeddings, _vectorstore

    if _vectorstore is None:

        _embeddings = SigLIP2Embeddings()

        _vectorstore = ouvrir_base(
            _embeddings
        )

    return (
        _embeddings,
        _vectorstore
    )


# =========================================================
# 2. RETRIEVER
# =========================================================

def _chercher(
    vecteur,
    k
):
    """Vecteur -> k entrées les plus proches
    avec leur similarité cosinus.
    """

    _, vectorstore = charger_base()

    trouves = (
        vectorstore
        .similarity_search_by_vector_with_relevance_scores(
            vecteur,
            k=k
        )
    )

    if not trouves:
        raise SystemExit(
            "La base Chroma est vide : "
            "lancez d'abord uv run indexer.py"
        )

    return [
        dict(
            doc.metadata,
            score=1 - distance
        )
        for doc, distance
        in trouves
    ]


def verdict(score):
    """Score du meilleur résultat -> verdict."""

    if score >= SEUIL_EXISTE:

        return (
            "OUI : l'image existe dans le catalogue"
        )

    if score >= SEUIL_RESSEMBLE:

        return (
            "PROCHE : image absente, "
            "mais un produit similaire existe"
        )

    return (
        "NON : image introuvable dans le catalogue"
    )


def rechercher(
    image,
    k=3
):
    """Image -> verdict + résultats."""

    embeddings, _ = charger_base()

    vecteur = embeddings.embed_image(
        [image]
    )[0]

    resultats = _chercher(
        vecteur,
        k
    )

    return (
        verdict(
            resultats[0]["score"]
        ),
        resultats
    )


def rechercher_texte(
    phrase,
    k=5
):
    """Phrase -> k images proches."""

    embeddings, _ = charger_base()

    vecteur = embeddings.embed_query(
        phrase
    )

    return _chercher(
        vecteur,
        k
    )


# =========================================================
# 3. PROMPT RAG
# =========================================================

def image_en_url(
    image,
    taille_max=1024
):
    """Image -> URL data JPEG pour Gemini."""

    image = (
        Image.open(image)
        if isinstance(image, str)
        else image
    ).convert("RGB")

    image.thumbnail(
        (
            taille_max,
            taille_max
        )
    )

    tampon = io.BytesIO()

    image.save(
        tampon,
        "JPEG",
        quality=90
    )

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            tampon.getvalue()
        ).decode()
    )


def construire_messages(
    photo,
    meilleur,
    le_verdict,
    question=""
):
    """Construit les messages envoyés à Gemini."""

    fiche = (
        f"Verdict de la recherche : "
        f"{le_verdict} "
        f"(score {meilleur['score']:.2f})\n"

        f"Produit : "
        f"{meilleur['produit']}\n"

        f"Références : "
        f"{meilleur['references'] or 'aucune'}\n"

        f"Page du catalogue : "
        f"{meilleur['page']}\n"

        f"Fiche : "
        f"{meilleur['fiche']}"
    )

    return [

        SystemMessage(
            content=CONSIGNES
        ),

        HumanMessage(
            content=[

                {
                    "type": "text",
                    "text":
                        "IMAGE 1 "
                        "(photo du client) :"
                },

                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                            image_en_url(
                                photo
                            )
                    }
                },

                {
                    "type": "text",
                    "text":
                        "IMAGE 2 "
                        "(image du catalogue) :"
                },

                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                            image_en_url(
                                meilleur["image"]
                            )
                    }
                },

                {
                    "type": "text",
                    "text": fiche
                },

                {
                    "type": "text",
                    "text":
                        f"Question du client : "
                        f"{question or 'aucune'}"
                }
            ]
        )
    ]


# =========================================================
# 4. GEMINI + RÉPONSE
# =========================================================

def charger_llm():
    global _llm

    if _llm is None:

        _llm = ChatGoogleGenerativeAI(
            model=LLM,
            temperature=0
        )

    return _llm


def repondre(
    photo,
    question="",
    recherche=None
):
    """Photo + question -> réponse Gemini."""

    # R : Retrieval
    le_verdict, resultats = (
        recherche
        or rechercher(photo)
    )

    meilleur = resultats[0]

    # Si rien de suffisamment proche :
    # pas d'appel Gemini
    if (
        meilleur["score"]
        < SEUIL_RESSEMBLE
    ):

        return (
            "Aucun produit correspondant "
            "dans le catalogue : "
            "rien à décrire."
        )

    # A : augmentation
    messages = construire_messages(
        photo,
        meilleur,
        le_verdict,
        question
    )

    # G : génération
    response = charger_llm().invoke(
        messages
    )

    return response.content


# =========================================================
# 5. AFFICHAGE TERMINAL
# =========================================================

def afficher(
    le_verdict,
    resultats
):
    m = resultats[0]

    print(
        f"{le_verdict} "
        f"(score {m['score']:.3f})\n"
    )

    print(
        "Page :",
        m["page"]
    )

    print(
        "Références :",
        m["references"]
        or "aucune détectée"
    )

    print(
        "Produit :",
        m["produit"]
    )

    print(
        "Page fiche :",
        m["page_fiche"]
    )

    print(
        "Fiche (début) :",
        m["fiche"][:400]
    )

    print(
        "\nAutres candidats :"
    )

    for r in resultats[1:]:

        print(
            f" page {r['page']:3d} "
            f"{r['references'] or '-':20s} "
            f"score {r['score']:.3f}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    p = argparse.ArgumentParser(
        description=(
            "RAG visuel : "
            "SigLIP2 + Chroma + Gemini"
        )
    )

    p.add_argument(
        "image",
        nargs="?",
        help="photo à rechercher"
    )

    p.add_argument(
        "question",
        nargs="?",
        default="",
        help="question pour Gemini"
    )

    p.add_argument(
        "--gemini",
        action="store_true",
        help="faire répondre Gemini"
    )

    p.add_argument(
        "--texte",
        help=(
            "rechercher avec une phrase "
            "au lieu d'une image"
        )
    )

    args = p.parse_args()


    # -----------------------------------------------------
    # RECHERCHE TEXTE
    # -----------------------------------------------------

    if args.texte:

        print(
            "\n=== RÉSULTATS ==="
        )

        for r in rechercher_texte(
            args.texte
        ):

            print(
                f"{r['score']:.3f} "
                f"page {r['page']:3d} "
                f"{r['references'] or '-':9s} "
                f"{r['produit'][:60]}"
            )


    # -----------------------------------------------------
    # RECHERCHE IMAGE
    # -----------------------------------------------------

    elif args.image:

        recherche = rechercher(
            args.image
        )

        print(
            "\n=== RÉSULTATS ==="
        )

        afficher(
            *recherche
        )

        if (
            args.gemini
            or args.question
        ):

            print(
                "\n=== RÉPONSE LLM ==="
            )

            print(
                repondre(
                    args.image,
                    args.question,
                    recherche
                )
            )

    else:

        p.print_help()


if __name__ == "__main__":
    main()