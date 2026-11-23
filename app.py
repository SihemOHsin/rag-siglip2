"""Interface Streamlit : envoyer une photo, retrouver le produit,
lire la réponse de Gemini.

Utilisation :
    uv run streamlit run app.py
"""

import tempfile

import streamlit as st
from PIL import Image

from rag import rechercher, repondre


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Recherche dans le catalogue",
    page_icon="🔎",
    layout="wide"
)

st.title(
    "🔎 Ce produit est-il dans le catalogue ?"
)


# =========================================================
# ENTRÉES UTILISATEUR
# =========================================================

photo = st.file_uploader(
    "Photo du produit",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)

question = st.text_input(
    "Question (facultatif)",
    placeholder=(
        "Ex. : Est-ce que c'est en promotion ?"
    )
)


# =========================================================
# RECHERCHE
# =========================================================

if photo and st.button(
    "Rechercher",
    type="primary"
):

    # Streamlit donne un fichier en mémoire.
    # On l'enregistre temporairement sur le disque.

    image = Image.open(
        photo
    ).convert("RGB")

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    ) as f:

        image.save(
            f,
            "PNG"
        )

        chemin = f.name


    # =====================================================
    # RETRIEVAL : SIGLIP2 + CHROMA
    # =====================================================

    with st.spinner(
        "Recherche dans le catalogue "
        "(SigLIP2 + Chroma)..."
    ):

        recherche = rechercher(
            chemin
        )


    verdict, resultats = recherche

    m = resultats[0]


    # =====================================================
    # VERDICT
    # =====================================================

    if verdict.startswith(
        "OUI"
    ):

        st.success(
            f"{verdict} "
            f"(score {m['score']:.3f})",
            icon="✅"
        )

    elif verdict.startswith(
        "PROCHE"
    ):

        st.warning(
            f"{verdict} "
            f"(score {m['score']:.3f})",
            icon="⚠️"
        )

    else:

        st.error(
            f"{verdict} "
            f"(score {m['score']:.3f})",
            icon="❌"
        )


    # =====================================================
    # AFFICHAGE DES IMAGES
    # =====================================================

    gauche, droite = st.columns(2)

    gauche.image(
        image,
        caption="Votre photo",
        width=200
    )

    droite.image(
        m["image"],
        caption=(
            f"Catalogue, page "
            f"{m['page']}"
        ),
        width=300
    )


    # =====================================================
    # INFORMATIONS PRODUIT
    # =====================================================

    st.subheader(
        m["produit"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Références",
        m["references"]
        or "aucune"
    )

    c2.metric(
        "Page",
        m["page"]
    )

    c3.metric(
        "Similarité",
        f"{m['score']:.0%}"
    )


    # =====================================================
    # FICHE DU CATALOGUE
    # =====================================================

    with st.expander(
        "Fiche du catalogue"
    ):

        st.write(
            m["fiche"]
        )


    # =====================================================
    # RÉPONSE GEMINI
    # =====================================================

    if not verdict.startswith(
        "NON"
    ):

        st.subheader(
            "Réponse de l'assistant"
        )

        with st.spinner(
            "Gemini décrit le produit..."
        ):

            reponse = repondre(
                chemin,
                question,
                recherche
            )


        # Gemini peut renvoyer une liste de blocs.
        # On garde uniquement le texte.

        if isinstance(
            reponse,
            list
        ):

            reponse = "".join(
                b["text"]
                for b in reponse
                if (
                    isinstance(
                        b,
                        dict
                    )
                    and b.get(
                        "type"
                    ) == "text"
                )
            )


        with st.container(
            border=True
        ):

            st.markdown(
                reponse
            )


    # =====================================================
    # AUTRES CANDIDATS
    # =====================================================

    if len(
        resultats
    ) > 1:

        with st.expander(
            "Autres candidats"
        ):

            colonnes = st.columns(
                len(
                    resultats
                ) - 1
            )

            for col, r in zip(
                colonnes,
                resultats[1:]
            ):

                col.image(
                    r["image"],
                    width=150
                )

                col.caption(
                    f"Page {r['page']} · "
                    f"{r['references'] or '-'} · "
                    f"score {r['score']:.3f}"
                )