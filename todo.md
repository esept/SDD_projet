# Prétraitement des données [ FINI ]
- Convertir tous les formats de données en formats utilisables
- Supprimer toutes les colonnes vides

# Traitement des données
- Nettoyer les données aberrantes
- Découper le format des données (temps, …)

# Analyse des données
- Relation avec la langue (langs, NB_langs, ONE-HOT_langs[NB_en,NB_ar,… les six dernières colonnes])
- Relation avec le framework (framework_…)
- Relation avec le nombre de publications (nb_arxiv), le nombre de jeux de données (nb_dataset), …
- Relation avec les modèles de base (base-models) : Les modèles de base sont similaires aux logiciels open source, la principale préoccupation est la publication et la maintenance de nouveaux modèles
- Relation avec les tâches (tâche principale : pipeline_tag, tâches secondaires : tasks)
- ONE-HOT_… : les tags suivants sont les plus fréquents dans les statistiques `ONEHOT_endpoints_compatible`, `ONEHOT_autotrain_compatibl`, `ONEHOT_safetensors`, `ONEHOT_tensorboard`, `ONEHOT_has_space`

# Application de visualisation
Utiliser `streamlit` pour créer une page de visualisation en temps réel
