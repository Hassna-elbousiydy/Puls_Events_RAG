# Point de reprise du 15 septembre 2026

Travail non finalisé sur la branche fix/complete-rag-poc, à partir du ZIP utilisateur
et du commit 2ba6c5f. Le dépôt distant retourne 404 via le connecteur GitHub.
Aucune modification de main, aucun push et aucune PR effectués.

Mission et cours lus intégralement. Les nouvelles pièces jointes sont identiques.
Audit initial : 44 tests réussis sur la date historique 2026-09-08.
Après actualisation au 2026-09-14 : 59 tests offline réussis.
Après ajout des tests complémentaires : 19 tests unitaires réussis.
La dernière adaptation du client HTTP pour prendre en compte les certificats du
proxy TLS reste à retester : l'environnement virtuel temporaire a disparu à la reprise.
Ne pas confondre ces résultats avec une validation finale du code actuel.

Diagnostic direct réel : modèles HTTP 200, mistral-small-2603 présent,
chat minimal HTTP 429 code 1300. LangChain rencontrait aussi une erreur de
certificat TLS dans Work. La correction conserve la vérification TLS active.
Aucune génération réussie n'est certifiée.

Données actualisées au 14 septembre : 13 329 événements, 15 168 vecteurs,
51 événements et 69 chunks périmés retirés, vecteurs inchangés réutilisés.
Le 15 septembre nécessite une nouvelle validation de fraîcheur.
Anciennes données/index conservés dans les dossiers .previous-* ignorés par Git.
Les fichiers reports/evidence sont des copies exactes des rapports disponibles,
y compris les checkpoints incomplets, et ne doivent pas être présentés comme
une évaluation terminée si status vaut running.

Restent : README complet, matrice des quatre étapes et cinq rubriques,
validation de reconstruction, contrôles finaux, résultats d'évaluation consolidés,
revue humaine des annotations, rapport et présentation, push et PR dès accès GitHub.
Le jeu de 25 références provient du CSV ; human_reviewed reste false.

Reprise : installer requirements.txt dans un venv ; avec un proxy SOCKS, installer
httpx[socks] dans cet environnement. Relancer pytest et les diagnostics explicites.
La maintenance sans nouvel embedding est dans scripts/refresh_snapshot.py ;
la reconstruction complète dans scripts/rebuild_pipeline.py.
