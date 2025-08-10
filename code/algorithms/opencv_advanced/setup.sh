#!/bin/bash
# Setup script pour l'algorithme OpenCV Advanced

echo "🔧 Configuration environnement OpenCV Advanced"

# Créer et activer l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "Création du venv..."
    python3 -m venv venv
fi

echo "Activation du venv..."
source venv/bin/activate

echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environnement OpenCV Advanced prêt!"
echo "Pour l'activer: source code/algorithms/opencv_advanced/venv/bin/activate"