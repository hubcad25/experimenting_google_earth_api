#!/usr/bin/env python3
"""
Détecteur de piscines basé sur la couleur HSV (approche simple)
Basé sur le blog post de danielcorcoranssql
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import json

class OpenCVHSVPoolDetector:
    def __init__(self):
        # Plages de couleur pour détecter les piscines bleues en HSV
        # Format HSV: Hue (0-179), Saturation (0-255), Value (0-255)
        
        # Vraies couleurs de piscines calibrées
        self.lower_blue1 = np.array([100, 100, 180])   # Piscines claires
        self.upper_blue1 = np.array([110, 180, 255])
        
        # Piscines avec plus de variation
        self.lower_blue2 = np.array([95, 80, 150])
        self.upper_blue2 = np.array([115, 200, 255])
        
        # Paramètres de filtrage
        self.min_contour_area = 100      # Aire minimale pour éliminer le bruit
        self.max_contour_area = 50000    # Aire maximale (éviter océan/lac)
        self.min_aspect_ratio = 0.3      # Ratio largeur/hauteur minimum
        self.max_aspect_ratio = 3.0      # Ratio largeur/hauteur maximum
    
    def detect_pools(self, image_path: str) -> Dict:
        """
        Détecte les piscines dans une image satellite
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            Dict: Résultats de la détection
        """
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Impossible de charger l'image", "pools": []}
            
            # Obtenir les dimensions
            height, width = image.shape[:2]
            
            # Convertir en HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Créer des masques pour les différentes nuances de bleu
            mask1 = cv2.inRange(hsv, self.lower_blue1, self.upper_blue1)
            mask2 = cv2.inRange(hsv, self.lower_blue2, self.upper_blue2)
            
            # Combiner les masques
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Opérations morphologiques pour nettoyer le masque
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Enlever le bruit
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fermer les trous
            
            # Trouver les contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrer et analyser les contours
            pools = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filtrer par aire
                if area < self.min_contour_area or area > self.max_contour_area:
                    continue
                
                # Calculer le rectangle englobant
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filtrer par ratio d'aspect (éviter les formes trop allongées)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue
                
                # Calculer des métriques supplémentaires
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Informations sur la piscine détectée
                pool_info = {
                    "id": i,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "center": [int(x + w/2), int(y + h/2)],
                    "area_pixels": float(area),
                    "aspect_ratio": float(aspect_ratio),
                    "circularity": float(circularity),
                    "confidence": self._calculate_confidence(area, aspect_ratio, circularity)
                }
                
                pools.append(pool_info)
            
            # Trier par confiance décroissante
            pools.sort(key=lambda x: x["confidence"], reverse=True)
            
            result = {
                "image_path": image_path,
                "image_size": [width, height],
                "pools_detected": len(pools),
                "pools": pools,
                "algorithm": "opencv_hsv",
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {
                "image_path": image_path,
                "error": str(e),
                "pools": [],
                "algorithm": "opencv_hsv",
                "success": False
            }
    
    def _calculate_confidence(self, area: float, aspect_ratio: float, circularity: float) -> float:
        """
        Calcule un score de confiance basé sur les caractéristiques géométriques
        """
        confidence = 0.0
        
        # Score basé sur l'aire (piscines typiques entre 500-10000 pixels)
        if 500 <= area <= 10000:
            confidence += 0.4
        elif 100 <= area <= 500 or 10000 <= area <= 20000:
            confidence += 0.2
        
        # Score basé sur le ratio d'aspect (piscines souvent rectangulaires ou carrées)
        if 0.5 <= aspect_ratio <= 2.0:
            confidence += 0.3
        elif 0.3 <= aspect_ratio <= 3.0:
            confidence += 0.1
        
        # Score basé sur la circularité (formes régulières)
        if circularity > 0.3:
            confidence += 0.3
        elif circularity > 0.1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def save_visualization(self, image_path: str, result: Dict, output_path: str):
        """
        Sauvegarde une visualisation avec les piscines détectées
        """
        if not result["success"] or result["pools_detected"] == 0:
            print("Aucune piscine à visualiser")
            return
        
        # Charger l'image originale
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Dessiner les détections
        for pool in result["pools"]:
            x, y, w, h = pool["bbox"]
            confidence = pool["confidence"]
            
            # Couleur basée sur la confiance
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))  # Rouge->Vert
            thickness = 2
            
            # Rectangle englobant
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Texte avec confiance
            label = f"Pool {pool['id']}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        # Sauvegarder
        cv2.imwrite(output_path, image)
        print(f"✅ Visualisation sauvegardée: {output_path}")

def main():
    """Test de l'algorithme"""
    detector = OpenCVHSVPoolDetector()
    
    # Exemple d'utilisation
    image_path = "../images/test_satellite.jpg"  # À adapter
    
    print("🔍 Test OpenCV HSV Pool Detector")
    print("-" * 35)
    
    result = detector.detect_pools(image_path)
    
    if result["success"]:
        print(f"✅ Analyse terminée")
        print(f"   Piscines détectées: {result['pools_detected']}")
        
        for pool in result["pools"]:
            print(f"   - Piscine {pool['id']}: confiance {pool['confidence']:.2f}")
    else:
        print(f"❌ Erreur: {result.get('error', 'Inconnue')}")

if __name__ == "__main__":
    main()