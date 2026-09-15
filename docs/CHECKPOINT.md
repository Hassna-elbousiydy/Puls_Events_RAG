# Point de reprise du 15 septembre 2026

Branche locale fix/complete-rag-poc, base 2ba6c5f. Le connecteur GitHub retourne
404 pour Hassna-elbousiydy/Puls_Events_RAG. Aucun push, aucune PR, main inchangée.
Le bundle inclus dans la sauvegarde permet de restaurer les commits locaux.

Mission et cours lus intégralement, dernières pièces jointes identiques.
Voir CONFORMITE.md pour la matrice complète des quatre étapes et cinq rubriques.

Validation : 63 tests offline passent sur le snapshot actualisé au 15 septembre.
Le journal final est dans reports/evidence/pytest_final.txt. pip check sans conflit.
Docker absent de Work, image non construite ici.

Index actif : 13 312 événements, 15 149 chunks, 1024 dimensions, IndexFlatL2.
68 événements et 88 chunks exclus depuis le snapshot initial par vieillissement.
Aucun nouvel embedding pour cette maintenance. Données originales conservées
localement dans les dossiers previous ignorés et dans le ZIP fourni par l’utilisateur.
La sauvegarde du code n’inclut pas les gros corpus/index : pour les reproduire,
reprendre les données de ce ZIP puis lancer refresh_snapshot au 2026-09-15.

Diagnostic HTTP direct : modèles 200 et modèle présent, chat minimal 429 code 1300.
Le client LangChain corrigé atteint également Mistral et reçoit 429 : le problème
TLS local est résolu pour ce test. Aucune génération réussie n’est annoncée.
Évaluation génération : bloquée dès le premier cas, 0/25 complétés.
Évaluation retrieval réelle : 25/25 complétés le 15 septembre, avant activation
du snapshot du 15 (index du 14 : 13 329 événements, 15 168 chunks).
Sur 22 cas avec UID attendus : précision moyenne 0,2151515, rappel 1,0,
rang réciproque moyen 0,8787879. Deux cas d’absence corrects, un cas ambigu
réservé au jugement humain. Les labels ne sont pas exhaustifs et les filtres
ville/période sont passés explicitement : ces scores ont une portée limitée.

Restent : accès GitHub pour publier, limite du workspace Mistral pour génération
et évaluation sémantique, revue humaine des annotations, mise en forme finale
du rapport et du PowerPoint. Les trames sont dans docs/.
