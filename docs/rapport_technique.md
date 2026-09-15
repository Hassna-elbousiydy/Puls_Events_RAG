# Rapport technique préparatoire

Ce document prépare le rapport de soutenance. Il ne constitue pas un rapport final
PDF/Word de 5–10 pages et ne certifie aucune génération non obtenue.

## Besoin et périmètre

Puls-Events recommande en français des événements culturels OpenAgenda des Pays
de la Loire. Le POC utilise un an d'historique et les annonces futures disponibles.
Les événements historiques doivent être distingués des sorties encore accessibles.

## Architecture conservée

L'acquisition exporte les données publiques, Pandas nettoie les champs et créneaux,
le découpage conserve les UID et les lieux, mistral-embed produit 1024 dimensions,
FAISS IndexFlatL2 persiste les vecteurs. Le retrieval regroupe les chunks par UID,
filtre ville et période, puis un prompt LangChain transmet les sources au chat
Mistral configurable. Aucun historique conversationnel n'est nécessaire.

## Audit et corrections

Le corpus initial contenait 13 380 événements et 15 237 chunks. Le contrôle de la
fenêtre à la date actuelle a identifié des événements devenus trop anciens.
L'actualisation réutilise les vecteurs uniquement si textes et métadonnées sources
correspondent au docstore et à l'empreinte du rapport. Les données d'origine sont
conservées en sauvegarde. Cette maintenance n'est pas une nouvelle acquisition.

La construction FAISS écrit maintenant dans une zone temporaire puis valide avant
activation. Un refus API ne doit pas détruire l'index utilisé par la démo.
Le client chat n'effectue aucun retry automatique. Les erreurs authentification,
quota, réseau, timeout et réponse vide sont explicites. Les tests offline bloquent
les connexions réseau et utilisent des clients contrôlés, identifiés comme tels.

## Mesures et limites

Lire reports/evidence pour les résultats réellement sauvegardés, avec leur date,
leur statut et leur nombre de cas complétés. Ne pas assimiler un fichier partiel
à une évaluation terminée. Le diagnostic direct Mistral a retourné 429 code 1300
alors que le modèle était listé. Ce résultat concerne une limite fournisseur,
pas une erreur du retrieval. La nature exacte du quota n'est pas visible ici.

Le jeu de 25 cas copie les faits vérifiables du corpus, sans réponse générée par
un LLM. Une validation humaine est encore attendue. L'évaluation distingue UID
pertinents récupérés et qualité sémantique des réponses. Les scores sémantiques
restent indisponibles tant que la génération réelle n'est pas validée.

## Suite avant soutenance

Restaurer l'accès au dépôt dans le connecteur GitHub, examiner le quota Mistral,
faire relire les annotations, exécuter les tests d'intégration, exporter les
résultats réels et mettre ce contenu au format de rapport final demandé.
En production : renouvellement planifié des données, contrôle des annulations,
provenance des artefacts, suivi de latence/coûts/quotas, tests de charge et revue
humaine des hallucinations. Ne désérialiser que les index produits et vérifiés
par le projet ; le format pickle n'est pas sûr pour un fichier non fiable.
