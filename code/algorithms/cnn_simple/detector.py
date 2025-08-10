#!/usr/bin/env python3
"""
CNN Simple Pool Detector
Détecteur de piscines utilisant un CNN à 3 couches
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import argparse
from pathlib import Path

class PoolCNNDetector:
    def __init__(self, model_path=None, img_size=128):
        self.img_size = img_size
        self.model = None
        self.model_path = model_path or "pool_cnn_model.h5"
        
    def build_model(self):
        """Construit un CNN simple à 3 couches pour la classification binaire"""
        model = keras.Sequential([
            # Première couche convolutionnelle
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            
            # Deuxième couche convolutionnelle
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Troisième couche convolutionnelle
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Couches denses
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Classification binaire
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image_path):
        """Préprocesse une image pour la prédiction"""
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
        else:
            img = image_path
            
        # Redimensionner
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convertir BGR vers RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normaliser
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_model(self):
        """Charge un modèle pré-entraîné"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Modèle chargé depuis {self.model_path}")
            return True
        else:
            print(f"Aucun modèle trouvé à {self.model_path}")
            return False
    
    def save_model(self):
        """Sauvegarde le modèle"""
        if self.model:
            self.model.save(self.model_path)
            print(f"Modèle sauvegardé à {self.model_path}")
    
    def detect_pool(self, image_path, threshold=0.5):
        """Détecte la présence d'une piscine dans une image"""
        if self.model is None:
            if not self.load_model():
                print("Erreur: Aucun modèle disponible pour la prédiction")
                return False, 0.0
        
        # Préprocesser l'image
        img = self.preprocess_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Prédiction
        prediction = self.model.predict(img_batch, verbose=0)[0][0]
        
        has_pool = prediction >= threshold
        confidence = float(prediction)
        
        return has_pool, confidence
    
    def create_synthetic_data(self, n_samples=1000):
        """Crée des données synthétiques pour l'entraînement de démonstration"""
        print(f"Génération de {n_samples} échantillons synthétiques...")
        
        X = []
        y = []
        
        for i in range(n_samples):
            # Créer une image synthétique
            img = np.random.rand(self.img_size, self.img_size, 3).astype(np.float32)
            
            # Simuler une piscine (zone bleue rectangulaire)
            has_pool = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% ont une piscine
            
            if has_pool:
                # Ajouter une zone bleue (piscine)
                h, w = np.random.randint(20, 40), np.random.randint(30, 50)
                y_start = np.random.randint(0, self.img_size - h)
                x_start = np.random.randint(0, self.img_size - w)
                
                # Zone bleue
                img[y_start:y_start+h, x_start:x_start+w, 0] = 0.1  # Peu de rouge
                img[y_start:y_start+h, x_start:x_start+w, 1] = 0.3  # Peu de vert
                img[y_start:y_start+h, x_start:x_start+w, 2] = 0.8  # Beaucoup de bleu
                
                # Ajouter du bruit de végétation autour (vert)
                for _ in range(3):
                    gh, gw = np.random.randint(10, 30), np.random.randint(10, 30)
                    gy_start = np.random.randint(0, self.img_size - gh)
                    gx_start = np.random.randint(0, self.img_size - gw)
                    img[gy_start:gy_start+gh, gx_start:gx_start+gw, 1] = 0.6  # Vert
            
            X.append(img)
            y.append(has_pool)
        
        return np.array(X), np.array(y)
    
    def train(self, X=None, y=None, epochs=20, validation_split=0.2):
        """Entraîne le modèle"""
        if X is None or y is None:
            print("Génération de données synthétiques pour l'entraînement...")
            X, y = self.create_synthetic_data(2000)
        
        if self.model is None:
            self.build_model()
        
        print("Début de l'entraînement...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32,
            verbose=1
        )
        
        self.save_model()
        return history

def main():
    parser = argparse.ArgumentParser(description='Détecteur de piscines CNN simple')
    parser.add_argument('--image', type=str, help='Chemin vers l\'image à analyser')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle')
    parser.add_argument('--epochs', type=int, default=20, help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--threshold', type=float, default=0.5, help='Seuil de détection')
    
    args = parser.parse_args()
    
    detector = PoolCNNDetector()
    
    if args.train:
        print("=== Entraînement du modèle CNN ===")
        detector.train(epochs=args.epochs)
        print("Entraînement terminé!")
    
    if args.image:
        print(f"=== Analyse de l'image: {args.image} ===")
        if not os.path.exists(args.image):
            print(f"Erreur: Image non trouvée: {args.image}")
            return
        
        has_pool, confidence = detector.detect_pool(args.image, args.threshold)
        
        print(f"Piscine détectée: {'Oui' if has_pool else 'Non'}")
        print(f"Confiance: {confidence:.2%}")
        
        # Afficher l'image avec le résultat
        img = cv2.imread(args.image)
        if img is not None:
            # Ajouter le texte du résultat
            result_text = f"Pool: {'YES' if has_pool else 'NO'} ({confidence:.2%})"
            color = (0, 255, 0) if has_pool else (0, 0, 255)
            cv2.putText(img, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Afficher
            cv2.imshow('Pool Detection - CNN Simple', img)
            print("Appuyez sur une touche pour fermer la fenêtre...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    if not args.train and not args.image:
        print("Usage:")
        print("  python detector.py --train --epochs 30")
        print("  python detector.py --image /path/to/image.jpg")
        print("  python detector.py --image /path/to/image.jpg --threshold 0.7")

if __name__ == "__main__":
    main()