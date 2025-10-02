import streamlit as st
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
import openai
import textwrap
import re

# Configuration de la page
st.set_page_config(
    page_title="Recherche Jurisprudence",
    page_icon="⚖️",
    layout="wide"
)

# Configuration des constantes
COLLECTION_NAME = "blocs_motivation"
VECTOR_SIZE = 256
DEFAULT_LIMIT = 10

# Récupération des clés API depuis les secrets Streamlit
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
CLUSTER_QDRANT_URL = st.secrets["CLUSTER_QDRANT_URL"]

class MotivationBlocksSearcher:
    """Classe pour rechercher dans la collection blocs_motivation."""

    def __init__(self):
        """Initialise le client Qdrant et vérifie la collection."""
        self.client = QdrantClient(url=CLUSTER_QDRANT_URL, api_key=QDRANT_API_KEY)
        self._check_collection_exists()

    def _check_collection_exists(self):
        """Vérifie que la collection existe."""
        collections = self.client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if COLLECTION_NAME not in collection_names:
            raise Exception(f"Collection '{COLLECTION_NAME}' introuvable.")

    def generate_embedding(self, text: str) -> List[float]:
        """Génère un embedding pour le texte donné."""
        # Initialiser le client OpenAI ici
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=VECTOR_SIZE
        )
        return response.data[0].embedding

    def search_similar_decisions(self, query: str, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
        """Recherche des décisions similaires."""
        query_vector = self.generate_embedding(query)

        search_result = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )

        results = []
        for hit in search_result:
            result = {
                'score': hit.score,
                'decision_id': hit.payload.get('unique_ID'),
                'number': hit.payload.get('number'),
                'ecli': hit.payload.get('ecli'),
                'jurisdiction': hit.payload.get('jurisdiction'),
                'chamber': hit.payload.get('chamber'),
                'formation': hit.payload.get('formation'),
                'type': hit.payload.get('type'),
                'decision_date': hit.payload.get('decision_date'),
                'localisation': hit.payload.get('localisation'),
                'solution': hit.payload.get('solution'),
                'theme': hit.payload.get('theme'),
                'themes': hit.payload.get('themes'),
                'publication': hit.payload.get('publication'),
                'summary': hit.payload.get('summary'),
                'zones': hit.payload.get('zones'),
                'visa': hit.payload.get('visa'),
                'rapprochements': hit.payload.get('rapprochements'),
                'files': hit.payload.get('files')
            }
            results.append(result)

        return results


def format_themes_inline(themes) -> str:
    """Formate les thèmes sur une seule ligne."""
    if not themes:
        return "Non renseigné"
    if isinstance(themes, list):
        if len(themes) == 0:
            return "Non renseigné"
        return ", ".join(themes)
    return str(themes)


def create_decision_url(decision_id: str) -> str:
    """Crée l'URL vers une décision de la Cour de cassation."""
    if decision_id and decision_id != "Non renseigné":
        return f"https://www.courdecassation.fr/decision/{decision_id}"
    return None


def extract_decision_id_from_title(title: str) -> str:
    """Extrait un ID de décision depuis un titre de rapprochement."""
    patterns = [
        r'pourvoi n°\s*(\d{2}-\d{2}\.\d{3})',
        r'n°\s*(\d{2}-\d{2}\.\d{3})',
        r'(\d{2}-\d{2}\.\d{3})',
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


# Initialisation du moteur de recherche
@st.cache_resource
def get_searcher():
    """Initialise et met en cache le moteur de recherche."""
    return MotivationBlocksSearcher()


# Interface Streamlit
def main():
    # En-tête
    st.title("⚖️ Recherche de Jurisprudence")
    st.markdown("### 🔍 Trouvez des motivations juridiques répondant au moyen de droit soulevé")
    
    st.markdown("---")
    
    # Zone de saisie de la requête
    query = st.text_area(
        "📝 Votre question juridique :",
        height=150,
        placeholder="Exemple : L'employeur qui manque à son obligation de sécurité prévue à l'article L. 4121-1 du Code du travail...",
        help="Posez votre question juridique en langage naturel"
    )
    
    # Bouton de recherche
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        search_button = st.button("🔍 Rechercher", type="primary", use_container_width=True)
    
    # Exécution de la recherche
    if search_button and query:
        with st.spinner("🔄 Recherche en cours..."):
            try:
                searcher = get_searcher()
                results = searcher.search_similar_decisions(query, limit=10)
                
                if results:
                    st.success(f"✅ {len(results)} décision(s) trouvée(s)")
                    st.markdown("---")
                    
                    # Affichage des résultats
                    for i, result in enumerate(results, 1):
                        with st.expander(f"�️ **Décision #{i}** - Score: {result['score']:.4f} | {result.get('decision_date', 'Date non renseignée')}", expanded=(i==1)):
                            # Résumé
                            if result.get('summary'):
                                st.markdown("#### 📄 Résumé")
                                st.write(result['summary'])
                            else:
                                st.info("📄 Résumé non disponible")
                            
                            st.markdown("---")
                            
                            # Métadonnées essentielles
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**📅 Date de décision**")
                                st.write(result.get('decision_date', 'Non renseigné'))
                                
                                st.markdown("**⚖️ Chambre**")
                                st.write(result.get('chamber', 'Non renseigné'))
                                
                                st.markdown("**🏷️ Thèmes**")
                                st.write(format_themes_inline(result.get('themes')))
                            
                            with col_b:
                                st.markdown("**🔢 Numéro**")
                                st.write(result.get('number', 'Non renseigné'))
                                
                                st.markdown("**📋 Solution**")
                                st.write(result.get('solution', 'Non renseigné'))
                                
                                # Lien vers la décision
                                decision_url = create_decision_url(result.get('decision_id'))
                                if decision_url:
                                    st.markdown(f"**🔗 [Consulter la décision]({decision_url})**")
                            
                            # Rapprochements
                            rapprochements = result.get('rapprochements')
                            if rapprochements and isinstance(rapprochements, list) and len(rapprochements) > 0:
                                st.markdown("---")
                                st.markdown("#### 📚 Jurisprudences proches")
                                for rapp in rapprochements:
                                    if isinstance(rapp, dict):
                                        title = rapp.get('title', 'Sans titre')
                                        st.write(f"• {title}")
                                    else:
                                        st.write(f"• {str(rapp)}")
                
                else:
                    st.warning("⚠️ Aucun résultat trouvé")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de la recherche : {str(e)}")
    
    elif search_button and not query:
        st.warning("⚠️ Veuillez saisir une question juridique")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 *Cette application recherche dans la base de jurisprudence de la Cour de cassation*")


if __name__ == "__main__":
    main()
