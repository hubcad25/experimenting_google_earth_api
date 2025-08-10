#!/usr/bin/env python3
"""
Détecteur de piscines OpenCV avancé avec analyse de contours sophistiquée
Basé sur le repo danielc92/cv2-pool-detection avec améliorations
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from scipy import ndimage
import json

class OpenCVAdvancedPoolDetector:
    def __init__(self):
        # Paramètres de détection HSV calibrés pour vraies piscines
        self.pool_hsv_ranges = [
            # Piscines bleu clair (#86c5f8, #579de1)
            {'lower': np.array([100, 100, 200]), 'upper': np.array([110, 180, 255])},
            # Piscines bleu moyen (#5296eb, #4986cc)
            {'lower': np.array([105, 120, 180]), 'upper': np.array([108, 170, 240])},
            # Piscines avec variations (reflets, ombres légères)
            {'lower': np.array([95, 80, 150]), 'upper': np.array([115, 200, 255])}
        ]
        
        # Filtres géométriques stricts
        self.min_area = 200          # Aire minimale plus stricte
        self.max_area = 15000        # Aire maximale plus stricte
        self.min_perimeter = 50      # Périmètre minimal
        self.max_perimeter = 500     # Périmètre maximal
        
        # Ratios de forme pour piscines
        self.min_aspect_ratio = 0.4
        self.max_aspect_ratio = 2.5
        self.min_extent = 0.3        # Ratio aire/aire rectangle englobant
        self.min_solidity = 0.7      # Ratio aire/aire enveloppe convexe
        self.min_circularity = 0.3   # Plus strict sur la circularité
        
        # Paramètres morphologiques
        self.morph_kernel_size = 3
        self.blur_kernel_size = 5

    def detect_pools(self, image_path: str) -> Dict:
        """
        Détection avancée des piscines avec analyse multi-critères
        """
        try:
            # Charger et préprocesser l'image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Impossible de charger l'image", "pools": []}
            
            height, width = image.shape[:2]
            
            # Préprocessing pour améliorer la détection
            processed_image = self._preprocess_image(image)
            
            # Détecter les candidats avec multiple plages HSV
            candidates = self._detect_blue_regions(processed_image)
            
            # Analyser et filtrer les candidats
            pools = self._analyze_candidates(candidates, image)
            
            # Post-processing pour éliminer les doublons et faux positifs
            pools = self._post_process_detections(pools, width, height)
            
            # Trier par score de confiance
            pools.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "image_path": image_path,
                "image_size": [width, height],
                "pools_detected": len(pools),
                "pools": pools,
                "algorithm": "opencv_advanced",
                "success": True
            }
            
        except Exception as e:
            return {
                "image_path": image_path,
                "error": str(e),
                "pools": [],
                "algorithm": "opencv_advanced",
                "success": False
            }
    
    def _preprocess_image(self, image):
        """Préprocessing avancé de l'image"""
        # Réduction du bruit avec filtre bilatéral
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Amélioration du contraste avec CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Léger flou gaussien pour lisser les textures
        blurred = cv2.GaussianBlur(enhanced, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        return blurred
    
    def _detect_blue_regions(self, image):
        """Détection des régions bleues avec multiple plages HSV"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Combiner les masques de différentes plages de bleu
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for hsv_range in self.pool_hsv_ranges:
            mask = cv2.inRange(hsv, hsv_range['lower'], hsv_range['upper'])
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Opérations morphologiques avancées
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.morph_kernel_size, self.morph_kernel_size))
        
        # Opening pour enlever le bruit
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Closing pour fermer les trous
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Dilatation légère pour reconnecter les parties fragmentées
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def _analyze_candidates(self, contours, original_image):
        """Analyse approfondie des candidats avec critères géométriques"""
        pools = []
        
        for i, contour in enumerate(contours):
            # Calculs géométriques de base
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filtrage de base par aire et périmètre
            if (area < self.min_area or area > self.max_area or 
                perimeter < self.min_perimeter or perimeter > self.max_perimeter):
                continue
            
            # Rectangle englobant et ses propriétés
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filtrage par aspect ratio
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Calculs de forme avancés
            extent = area / rect_area if rect_area > 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Circularité améliorée
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Filtrage par propriétés géométriques
            if (extent < self.min_extent or 
                solidity < self.min_solidity or 
                circularity < self.min_circularity):
                continue
            
            # Analyse de la régularité du contour
            contour_regularity = self._calculate_contour_regularity(contour)
            
            # Analyse du contexte (éviter les bords d'image)
            edge_penalty = self._calculate_edge_penalty(x, y, w, h, original_image.shape[:2])
            
            # Score de confiance composite
            confidence = self._calculate_advanced_confidence(
                area, aspect_ratio, circularity, extent, solidity, 
                contour_regularity, edge_penalty
            )
            
            # Seuil minimal de confiance
            if confidence < 0.3:
                continue
            
            # Informations détaillées sur la piscine
            pool_info = {
                "id": i,
                "bbox": [int(x), int(y), int(w), int(h)],
                "center": [int(x + w/2), int(y + h/2)],
                "area_pixels": float(area),
                "perimeter": float(perimeter),
                "aspect_ratio": float(aspect_ratio),
                "circularity": float(circularity),
                "extent": float(extent),
                "solidity": float(solidity),
                "contour_regularity": float(contour_regularity),
                "edge_penalty": float(edge_penalty),
                "confidence": float(confidence)
            }
            
            pools.append(pool_info)
        
        return pools
    
    def _calculate_contour_regularity(self, contour):
        """Calcule la régularité du contour (piscines ont des formes régulières)"""
        # Approximation polygonale du contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Plus le contour est proche d'un polygone simple, plus il est régulier
        regularity = len(contour) / len(approx) if len(approx) > 0 else 0
        
        # Normaliser entre 0 et 1 (valeurs typiques entre 10-50 pour piscines)
        return min(regularity / 50.0, 1.0)
    
    def _calculate_edge_penalty(self, x, y, w, h, image_shape):
        """Pénalité pour les détections près des bords (souvent faux positifs)"""
        height, width = image_shape
        margin = 20  # Pixels de marge
        
        penalty = 0.0
        
        # Pénalité si trop proche des bords
        if x < margin or y < margin or (x + w) > (width - margin) or (y + h) > (height - margin):
            penalty = 0.3
        
        # Pénalité si touche exactement le bord (souvent des artefacts)
        if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
            penalty = 0.6
        
        return penalty
    
    def _calculate_advanced_confidence(self, area, aspect_ratio, circularity, 
                                     extent, solidity, regularity, edge_penalty):
        """Score de confiance multicritères avancé"""
        confidence = 0.0
        
        # Score basé sur l'aire (piscines typiques 1000-8000 pixels)
        if 1000 <= area <= 8000:
            confidence += 0.25
        elif 500 <= area <= 1000 or 8000 <= area <= 12000:
            confidence += 0.15
        elif 200 <= area <= 500:
            confidence += 0.1
        
        # Score basé sur l'aspect ratio (piscines souvent rectangulaires)
        if 0.6 <= aspect_ratio <= 1.8:
            confidence += 0.20
        elif 0.4 <= aspect_ratio <= 0.6 or 1.8 <= aspect_ratio <= 2.5:
            confidence += 0.1
        
        # Score basé sur la circularité
        if circularity > 0.5:
            confidence += 0.20
        elif circularity > 0.3:
            confidence += 0.15
        elif circularity > 0.2:
            confidence += 0.1
        
        # Score basé sur l'extent et la solidité
        if extent > 0.6 and solidity > 0.8:
            confidence += 0.15
        elif extent > 0.4 and solidity > 0.7:
            confidence += 0.1
        
        # Score basé sur la régularité du contour
        if regularity > 0.3:
            confidence += 0.1
        elif regularity > 0.2:
            confidence += 0.05
        
        # Application de la pénalité des bords
        confidence = max(0, confidence - edge_penalty)
        
        return min(confidence, 1.0)
    
    def _post_process_detections(self, pools, width, height):
        """Post-processing pour éliminer doublons et améliorer la précision"""
        if len(pools) <= 1:
            return pools
        
        # Éliminer les détections qui se chevauchent (Non-Maximum Suppression simplifié)
        filtered_pools = []
        pools_sorted = sorted(pools, key=lambda x: x["confidence"], reverse=True)
        
        for pool in pools_sorted:
            overlap_found = False
            
            for existing_pool in filtered_pools:
                if self._calculate_overlap(pool, existing_pool) > 0.5:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered_pools.append(pool)
        
        return filtered_pools
    
    def _calculate_overlap(self, pool1, pool2):
        """Calcule le chevauchement entre deux détections"""
        x1, y1, w1, h1 = pool1["bbox"]
        x2, y2, w2, h2 = pool2["bbox"]
        
        # Intersection
        inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter_area = inter_x * inter_y
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def save_visualization(self, image_path: str, result: Dict, output_path: str):
        """Visualisation avancée avec informations détaillées"""
        if not result["success"] or result["pools_detected"] == 0:
            print("Aucune piscine à visualiser")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Dessiner les détections avec codes couleur par confiance
        for pool in result["pools"]:
            x, y, w, h = pool["bbox"]
            confidence = pool["confidence"]
            
            # Couleur basée sur la confiance (rouge -> jaune -> vert)
            if confidence >= 0.8:
                color = (0, 255, 0)      # Vert (haute confiance)
                thickness = 3
            elif confidence >= 0.6:
                color = (0, 255, 255)    # Jaune (confiance moyenne)
                thickness = 2
            else:
                color = (0, 165, 255)    # Orange (faible confiance)
                thickness = 2
            
            # Rectangle englobant
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Centre de la piscine
            center_x, center_y = pool["center"]
            cv2.circle(image, (center_x, center_y), 3, color, -1)
            
            # Labels avec informations détaillées
            label = f"Pool {pool['id']}: {confidence:.2f}"
            detail_label = f"A:{pool['area_pixels']:.0f} C:{pool['circularity']:.2f}"
            
            # Texte principal
            cv2.putText(image, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
            # Texte détaillé
            cv2.putText(image, detail_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, color, 1)
        
        # Sauvegarder
        cv2.imwrite(output_path, image)
        print(f"✅ Visualisation avancée sauvegardée: {output_path}")

def main():
    """Test de l'algorithme avancé"""
    detector = OpenCVAdvancedPoolDetector()
    
    print("🔍 Test OpenCV Advanced Pool Detector")
    print("-" * 40)
    
    # Test sur une image d'exemple
    image_path = "../../images/test_satellite.jpg"
    
    result = detector.detect_pools(image_path)
    
    if result["success"]:
        print(f"✅ Analyse terminée")
        print(f"   Piscines détectées: {result['pools_detected']}")
        
        for pool in result["pools"][:5]:  # Afficher top 5
            print(f"   - Pool {pool['id']}: conf={pool['confidence']:.2f}, "
                  f"aire={pool['area_pixels']:.0f}, circ={pool['circularity']:.2f}")
    else:
        print(f"❌ Erreur: {result.get('error', 'Inconnue')}")

if __name__ == "__main__":
    main()