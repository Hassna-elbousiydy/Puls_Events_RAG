# Trame de soutenance — 12 diapositives à mettre en forme

1. Puls-Events : besoin métier et objectifs du POC.
2. Mission : région choisie, un an d'historique et événements à venir.
3. OpenAgenda : acquisition réelle, provenance et fréquence de mise à jour.
4. Prétraitement : valeurs manquantes, UID, créneaux et exclusions culturelles.
5. Chunking : contexte, overlap et métadonnées.
6. Embeddings Mistral et FAISS : dimension 1024, IndexFlatL2 et persistance.
7. Retrieval : recherche sémantique, ville, période et déduplication.
8. RAG : sources, prompt français, règles anti-invention et réponse.
9. Tests : distinguer offline et intégration ; afficher les résultats enregistrés.
10. Évaluation : références vérifiables, métriques UID, grille sémantique et limites.
11. Démo : afficher événements avant génération ; montrer honnêtement un blocage API.
12. Bilan : limites, quota, fraîcheur, supervision et industrialisation.

## Déroulé de démonstration

- Question historique : Quelle exposition à Nantes parle de la Libération et de la Seconde Guerre mondiale ?
- Recherche culturelle : Quels concerts sont proposés à Nantes ?
- Filtre explicite : même question avec --city Angers.
- Période : utiliser --start-date et --end-date avec des dates réellement présentes.
- Absence dans ce corpus régional : rechercher à Paris avec --city Paris.
- Ambiguïté : Que me conseilles-tu ? Expliquer la nécessité de préciser ville/période.

Vérifier chaque exemple avec le snapshot courant avant la soutenance : une
exposition historique peut sortir de la fenêtre. Ne pas annoncer un succès
Mistral si l'API répond 429. Présenter le diagnostic minimal et les sources récupérées.
