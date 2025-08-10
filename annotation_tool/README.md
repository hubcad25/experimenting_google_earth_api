# Outil d'Annotation de Piscines

Outil visuel pour annoter manuellement les piscines sur images satellites.

## Installation

```bash
cd annotation_tool
pip install opencv-python
```

## Utilisation

### Annoter toutes les images
```bash
python pool_annotator.py
```

### Annoter une image spécifique  
```bash
python pool_annotator.py --image ../images/satellite_xyz.jpg
```

### Personnaliser les dossiers
```bash
python pool_annotator.py --images ../images --output ../validation_dataset
```

## Instructions

- **Clic gauche + glisser** : Dessiner une bounding box autour d'une piscine
- **Clic droit** : Supprimer la piscine la plus proche
- **Touche 's'** : Sauvegarder les annotations
- **Touche 'n'** : Passer à l'image suivante
- **Touche 'q'** : Quitter

## Format de sortie

Les annotations sont sauvées en JSON :

```json
{
  "image": "satellite_xyz.jpg",
  "image_size": [640, 640],
  "pools": [
    {
      "id": 0,
      "bbox": [120, 150, 45, 30],
      "center": [142, 165],
      "type": "residential"
    }
  ],
  "total_pools": 1
}
```

## Workflow recommandé

1. Annoter 10-15 images diverses
2. Utiliser `validation_evaluator.py` pour tester les algorithmes  
3. Comparer précision/rappel de chaque algorithme