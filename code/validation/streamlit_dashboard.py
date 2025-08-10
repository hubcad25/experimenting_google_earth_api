#!/usr/bin/env python3
"""
Dashboard Streamlit pour visualiser les résultats de validation des algorithmes de détection de piscines
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

class PoolValidationDashboard:
    def __init__(self, results_path="../../validation_dataset/validation_results.json", 
                 validation_dir="../../validation_dataset",
                 images_dir="../../images"):
        self.results_path = Path(results_path)
        self.validation_dir = Path(validation_dir)
        self.images_dir = Path(images_dir)
        self.load_data()
        self.load_validation_annotations()
    
    def load_data(self):
        """Charge les données de validation"""
        if not self.results_path.exists():
            st.error(f"Fichier de résultats non trouvé: {self.results_path}")
            st.stop()
        
        with open(self.results_path, 'r') as f:
            self.data = json.load(f)
    
    def load_validation_annotations(self):
        """Charge les annotations manuelles"""
        self.annotations = {}
        for json_file in self.validation_dir.glob("*.json"):
            if json_file.name == "validation_results.json":
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    label = json_file.stem  # chapleau, bousquet, etc.
                    self.annotations[label] = data
            except Exception as e:
                st.warning(f"Erreur lecture {json_file}: {e}")
    
    def get_image_path(self, label):
        """Trouve le chemin de l'image pour un label donné"""
        for image_file in self.images_dir.glob(f"{label}_*.jpg"):
            return image_file
        return None
    
    def create_global_metrics_chart(self):
        """Graphique des métriques globales"""
        algorithms = []
        precision = []
        recall = []
        f1_score = []
        
        for algo_name, algo_data in self.data.items():
            metrics = algo_data['global_metrics']
            algorithms.append(algo_name.replace('opencv_', '').title())
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1_score.append(metrics['f1_score'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Précision',
            x=algorithms,
            y=precision,
            marker_color='#FF6B6B'
        ))
        
        fig.add_trace(go.Bar(
            name='Rappel',
            x=algorithms,
            y=recall,
            marker_color='#4ECDC4'
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=algorithms,
            y=f1_score,
            marker_color='#45B7D1'
        ))
        
        fig.update_layout(
            title='Comparaison des Métriques Globales',
            xaxis_title='Algorithmes',
            yaxis_title='Score',
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_per_image_analysis(self):
        """Analyse détaillée par image"""
        image_data = []
        
        for algo_name, algo_data in self.data.items():
            for image_name, image_results in algo_data['per_image_results'].items():
                # Simplifier le nom de l'image
                simple_name = self.simplify_image_name(image_name)
                
                image_data.append({
                    'Algorithme': algo_name.replace('opencv_', '').title(),
                    'Image': simple_name,
                    'GT_Pools': image_results['ground_truth_pools'],
                    'Predicted_Pools': image_results['predicted_pools'],
                    'Matches': image_results['matches'],
                    'Précision': image_results['metrics']['precision'],
                    'Rappel': image_results['metrics']['recall'],
                    'F1-Score': image_results['metrics']['f1_score'],
                    'IoU_Moyen': image_results['avg_iou']
                })
        
        return pd.DataFrame(image_data)
    
    def simplify_image_name(self, image_name):
        """Simplifie le nom d'image pour l'affichage"""
        coord_to_label = {
            "45.5780309414347": "chapleau",
            "45.53280102910986": "bousquet",
            "45.58021382961247": "gilbertdionne",
            "45.55994802880481": "sthilairerandom",
            "45.58319990832722": "beloeil",
            "46.71361046403241": "levis"
        }
        
        for coord, label in coord_to_label.items():
            if coord in image_name:
                return label.title()
        
        return image_name[:20] + "..."
    
    def create_confusion_matrix_data(self):
        """Données pour matrice de confusion"""
        confusion_data = []
        
        for algo_name, algo_data in self.data.items():
            metrics = algo_data['global_metrics']
            confusion_data.append({
                'Algorithme': algo_name.replace('opencv_', '').title(),
                'Vrais Positifs': metrics['true_positives'],
                'Faux Positifs': metrics['false_positives'],
                'Faux Négatifs': metrics['false_negatives']
            })
        
        return pd.DataFrame(confusion_data)
    
    def create_image_performance_heatmap(self, df):
        """Heatmap des performances par image"""
        pivot_data = df.pivot(index='Image', columns='Algorithme', values='F1-Score')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            text=[[f'{val:.3f}' for val in row] for row in pivot_data.values],
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Performance F1-Score par Image et Algorithme',
            xaxis_title='Algorithme',
            yaxis_title='Image',
            height=400
        )
        
        return fig
    
    def create_detection_analysis_chart(self, df):
        """Graphique d'analyse des détections"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Piscines Prédites vs Réelles', 'Matches par Image'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Graphique 1: Prédites vs Réelles
        for algo in df['Algorithme'].unique():
            algo_data = df[df['Algorithme'] == algo]
            fig.add_trace(
                go.Scatter(
                    x=algo_data['GT_Pools'],
                    y=algo_data['Predicted_Pools'],
                    mode='markers',
                    name=f'{algo} - Prédites',
                    marker=dict(size=8),
                    text=algo_data['Image'],
                    hovertemplate='%{text}<br>GT: %{x}<br>Pred: %{y}'
                ),
                row=1, col=1
            )
        
        # Ligne parfaite y=x
        max_pools = max(df['GT_Pools'].max(), df['Predicted_Pools'].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_pools],
                y=[0, max_pools],
                mode='lines',
                name='Prédiction Parfaite',
                line=dict(dash='dash', color='gray')
            ),
            row=1, col=1
        )
        
        # Graphique 2: Matches par image
        for algo in df['Algorithme'].unique():
            algo_data = df[df['Algorithme'] == algo]
            fig.add_trace(
                go.Bar(
                    x=algo_data['Image'],
                    y=algo_data['Matches'],
                    name=f'{algo} - Matches',
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        fig.update_layout(height=500, showlegend=True)
        fig.update_xaxes(title_text="Piscines Réelles", row=1, col=1)
        fig.update_yaxes(title_text="Piscines Prédites", row=1, col=1)
        fig.update_xaxes(title_text="Image", row=1, col=2)
        fig.update_yaxes(title_text="Nombre de Matches", row=1, col=2)
        
        return fig
    
    def get_algorithm_predictions(self, label, algorithm):
        """Récupère les prédictions de l'algorithme depuis les fichiers de résultats"""
        results_files = list(Path("../../results").glob("test_results_*.json"))
        
        for results_file in sorted(results_files, reverse=True):  # Le plus récent d'abord
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                for result in results:
                    if (result.get("algorithm") == algorithm and 
                        result.get("success", False) and
                        label.lower() in result.get("image_path", "").lower()):
                        return result.get("pools", [])
            except Exception as e:
                continue
        
        return []

    def create_simple_cnn_annotation(self, label, algorithm, predictions, ground_truth):
        """Crée une annotation simple pour CNN (classification seulement)"""
        # Trouver l'image
        image_path = self.get_image_path(label)
        if not image_path or not image_path.exists():
            return None, f"Image non trouvée pour {label}"
        
        # Charger l'image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, f"Impossible de charger l'image {image_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Récupérer la confiance CNN
        confidence = predictions[0].get('confidence', 0.0) if predictions else 0.0
        has_pool_pred = len(predictions) > 0
        has_pool_gt = len(ground_truth) > 0
        
        # Déterminer le type de prédiction
        if has_pool_pred and has_pool_gt:
            result_type = "TP"  # Vrai positif
            color = (0, 255, 0)  # Vert
        elif has_pool_pred and not has_pool_gt:
            result_type = "FP"  # Faux positif
            color = (255, 255, 0)  # Jaune
        elif not has_pool_pred and has_pool_gt:
            result_type = "FN"  # Faux négatif
            color = (255, 0, 0)  # Rouge
        else:
            result_type = "TN"  # Vrai négatif
            color = (128, 128, 128)  # Gris
        
        # Dessiner les ground truth en contours fins
        for gt_pool in ground_truth:
            x, y, w, h = gt_pool["bbox"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rouge fin pour GT
        
        # Ajouter le texte des résultats en haut de l'image
        result_text = f"CNN Prediction: {result_type}"
        confidence_text = f"Confidence: {confidence:.2%}"
        gt_text = f"GT Pools: {len(ground_truth)}"
        pred_text = f"Predicted: {'YES' if has_pool_pred else 'NO'}"
        
        # Background noir pour le texte
        cv2.rectangle(image, (5, 5), (400, 120), (0, 0, 0), -1)
        
        cv2.putText(image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, confidence_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, gt_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, pred_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Note explicative
        note_text = "Note: CNN fait de la classification globale (pas de localisation précise)"
        cv2.putText(image, note_text, (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Convertir en PIL Image pour Streamlit
        pil_image = Image.fromarray(image)
        
        return pil_image, None

    def create_annotated_image(self, label, algorithm):
        """Crée une image annotée avec ground truth et prédictions incluant les FP"""
        # Trouver l'image
        image_path = self.get_image_path(label)
        if not image_path or not image_path.exists():
            return None, f"Image non trouvée pour {label}"
        
        # Charger l'image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, f"Impossible de charger l'image {image_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Obtenir les annotations et prédictions
        if label not in self.annotations:
            return None, f"Annotations non trouvées pour {label}"
        
        ground_truth = self.annotations[label]["pools"]
        
        # Récupérer les prédictions complètes de l'algorithme
        predictions = self.get_algorithm_predictions(label, algorithm)
        if not predictions:
            return None, f"Prédictions non trouvées pour {algorithm} sur {label}"
        
        # Pour CNN simple, les prédictions peuvent ne pas avoir de bbox précises
        if algorithm == "cnn_simple" and predictions:
            # Vérifier si on a une prédiction simple sans bbox
            first_pred = predictions[0]
            if "bbox" not in first_pred or first_pred["bbox"] == [200, 200, 400, 300]:
                # C'est une prédiction CNN simple, on va créer une annotation simple
                return self.create_simple_cnn_annotation(label, algorithm, predictions, ground_truth)
        
        # Trouver les résultats de validation pour les matches
        algo_results = None
        matched_pairs = []
        
        for algo_name, algo_data in self.data.items():
            if algo_name == algorithm:
                for img_name, img_results in algo_data['per_image_results'].items():
                    if label.lower() in self.simplify_image_name(img_name).lower():
                        algo_results = img_results
                        matched_pairs = img_results.get('matched_pairs', [])
                        break
        
        if not algo_results:
            return None, f"Résultats de validation non trouvés pour {algorithm} sur {label}"
        
        # Créer des couleurs pour les annotations
        colors = {
            'tp': (0, 255, 0),      # Vert pour TP
            'fp': (255, 255, 0),    # Jaune pour FP  
            'fn': (255, 0, 0)       # Rouge pour FN
        }
        
        # Identifier les prédictions matchées (TP)
        matched_pred_indices = set()
        matched_gt_indices = set()
        
        # Dessiner les matches (TP) en vert
        for match in matched_pairs:
            gt_idx, pred_idx, iou = match
            
            # Dessiner ground truth en vert (TP)
            if gt_idx < len(ground_truth):
                gt_pool = ground_truth[gt_idx]
                x, y, w, h = gt_pool["bbox"]
                
                cv2.rectangle(image, (x, y), (x + w, y + h), colors['tp'], 3)
                cv2.putText(image, f"TP IoU:{iou:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['tp'], 2)
                
                matched_gt_indices.add(gt_idx)
            
            # Marquer la prédiction comme matchée
            matched_pred_indices.add(pred_idx)
        
        # Dessiner les FN (ground truth non matchées) en rouge
        for i, gt_pool in enumerate(ground_truth):
            if i not in matched_gt_indices:
                x, y, w, h = gt_pool["bbox"]
                cv2.rectangle(image, (x, y), (x + w, y + h), colors['fn'], 3)
                cv2.putText(image, "FN", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['fn'], 2)
        
        # Dessiner les FP (prédictions non matchées) en jaune
        for i, pred_pool in enumerate(predictions):
            if i not in matched_pred_indices:
                x, y, w, h = pred_pool["bbox"]
                
                # Dessiner la prédiction FP en jaune
                cv2.rectangle(image, (x, y), (x + w, y + h), colors['fp'], 3)
                cv2.putText(image, f"FP conf:{pred_pool.get('confidence', 0):.2f}", 
                           (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['fp'], 2)
        
        # Ajouter le texte des statistiques en haut de l'image
        stats_text = f"TP: {algo_results['matches']}, FP: {algo_results['metrics']['false_positives']}, FN: {algo_results['metrics']['false_negatives']}"
        
        # Background noir pour le texte
        cv2.rectangle(image, (5, 5), (600, 85), (0, 0, 0), -1)
        
        cv2.putText(image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Precision: {algo_results['metrics']['precision']:.3f}, Rappel: {algo_results['metrics']['recall']:.3f}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"F1-Score: {algo_results['metrics']['f1_score']:.3f}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convertir en PIL Image pour Streamlit
        pil_image = Image.fromarray(image)
        
        return pil_image, None

def show_overview_page(dashboard):
    """Page de vue d'ensemble"""
    st.title("📊 Vue d'Ensemble")
    
    # Métriques globales
    col1, col2, col3 = st.columns(3)
    
    # Extraire les meilleures performances
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_algo_precision = ""
    best_algo_recall = ""
    best_algo_f1 = ""
    
    for algo_name, algo_data in dashboard.data.items():
        metrics = algo_data['global_metrics']
        algo_display = algo_name.replace('opencv_', '').title()
        
        if metrics['precision'] > best_precision:
            best_precision = metrics['precision']
            best_algo_precision = algo_display
        
        if metrics['recall'] > best_recall:
            best_recall = metrics['recall']
            best_algo_recall = algo_display
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_algo_f1 = algo_display
    
    with col1:
        st.metric("🎯 Meilleure Précision", f"{best_precision:.3f}", f"{best_algo_precision}")
    
    with col2:
        st.metric("🔍 Meilleur Rappel", f"{best_recall:.3f}", f"{best_algo_recall}")
    
    with col3:
        st.metric("⚖️ Meilleur F1-Score", f"{best_f1:.3f}", f"{best_algo_f1}")
    
    st.plotly_chart(dashboard.create_global_metrics_chart(), use_container_width=True)

def show_confusion_matrix_page(dashboard):
    """Page matrice de confusion"""
    st.title("🔢 Matrice de Confusion")
    
    confusion_df = dashboard.create_confusion_matrix_data()
    st.dataframe(confusion_df, use_container_width=True)
    
    # Graphique en barres des TP/FP/FN
    fig = go.Figure()
    
    for _, row in confusion_df.iterrows():
        algo = row['Algorithme']
        fig.add_trace(go.Bar(name=f'{algo} - TP', x=[algo], y=[row['Vrais Positifs']], marker_color='green'))
        fig.add_trace(go.Bar(name=f'{algo} - FP', x=[algo], y=[row['Faux Positifs']], marker_color='orange'))
        fig.add_trace(go.Bar(name=f'{algo} - FN', x=[algo], y=[row['Faux Négatifs']], marker_color='red'))
    
    fig.update_layout(title='Distribution des TP/FP/FN par Algorithme', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def show_performance_heatmap_page(dashboard):
    """Page heatmap des performances"""
    st.title("🔥 Heatmap des Performances")
    
    df_images = dashboard.create_per_image_analysis()
    st.plotly_chart(dashboard.create_image_performance_heatmap(df_images), use_container_width=True)

def show_detection_analysis_page(dashboard):
    """Page analyse des détections"""
    st.title("📈 Analyse des Détections")
    
    df_images = dashboard.create_per_image_analysis()
    st.plotly_chart(dashboard.create_detection_analysis_chart(df_images), use_container_width=True)

def show_detailed_results_page(dashboard):
    """Page résultats détaillés"""
    st.title("📋 Résultats Détaillés")
    
    df_images = dashboard.create_per_image_analysis()
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        selected_algo = st.multiselect(
            "Filtrer par algorithme:",
            df_images['Algorithme'].unique(),
            default=df_images['Algorithme'].unique()
        )
    
    with col2:
        selected_images = st.multiselect(
            "Filtrer par image:",
            df_images['Image'].unique(),
            default=df_images['Image'].unique()
        )
    
    # Appliquer filtres
    filtered_df = df_images[
        (df_images['Algorithme'].isin(selected_algo)) &
        (df_images['Image'].isin(selected_images))
    ]
    
    # Formater pour l'affichage
    display_df = filtered_df.copy()
    for col in ['Précision', 'Rappel', 'F1-Score', 'IoU_Moyen']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Statistiques rapides
    st.subheader("📈 Statistiques")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_gt_pools = df_images['GT_Pools'].sum() // 2  # Diviser par 2 car 2 algorithmes
        st.metric("🏊 Total Piscines Réelles", int(total_gt_pools))
    
    with col2:
        avg_precision = df_images['Précision'].mean()
        st.metric("📊 Précision Moyenne", f"{avg_precision:.3f}")
    
    with col3:
        avg_recall = df_images['Rappel'].mean()
        st.metric("📊 Rappel Moyen", f"{avg_recall:.3f}")
    
    with col4:
        avg_f1 = df_images['F1-Score'].mean()
        st.metric("📊 F1-Score Moyen", f"{avg_f1:.3f}")

def show_annotated_images_page(dashboard):
    """Page visualisation images annotées"""
    st.title("🖼️ Images Annotées")
    
    # Layout en deux colonnes: contrôles à gauche, images à droite
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        **Légende:**
        - 🟢 **Vert**: Vrai Positif (TP) - Piscine correctement détectée
        - 🔴 **Rouge**: Faux Négatif (FN) - Piscine ratée par l'algorithme
        - 🟡 **Jaune**: Faux Positif (FP) - Détection incorrecte
        """)
        
        # Sélecteur d'image
        available_labels = list(dashboard.annotations.keys())
        if not available_labels:
            st.error("Aucune annotation trouvée")
            return
        
        selected_label = st.selectbox("Choisir une image:", available_labels)
    
    with col2:
        if selected_label:
            st.subheader(f"📍 Image: {selected_label.title()}")
            
            # Afficher les trois algorithmes en colonnes
            subcol1, subcol2, subcol3 = st.columns(3)
            algorithms = ['opencv_hsv', 'opencv_advanced', 'cnn_simple']
            
            for i, algo in enumerate(algorithms):
                column = subcol1 if i == 0 else (subcol2 if i == 1 else subcol3)
                with column:
                    st.subheader(f"🤖 {algo.replace('opencv_', '').replace('cnn_', '').title()}")
                    
                    try:
                        annotated_image, error = dashboard.create_annotated_image(selected_label, algo)
                        
                        if error:
                            st.error(error)
                        elif annotated_image:
                            st.image(annotated_image, use_container_width=True)
                        else:
                            st.error(f"Impossible de créer l'image annotée pour {algo}")
                            
                    except Exception as e:
                        st.error(f"Erreur: {e}")

def main():
    st.set_page_config(
        page_title="🏊 Dashboard Validation Piscines",
        page_icon="🏊",
        layout="wide"
    )
    
    # Initialiser le dashboard
    try:
        dashboard = PoolValidationDashboard()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        st.stop()
    
    # Sidebar pour navigation
    st.sidebar.title("🏊 Navigation")
    
    pages = {
        "🖼️ Images Annotées": show_annotated_images_page,
        "📊 Vue d'Ensemble": show_overview_page,
        "🔢 Matrice de Confusion": show_confusion_matrix_page,
        "🔥 Heatmap Performances": show_performance_heatmap_page,
        "📈 Analyse Détections": show_detection_analysis_page,
        "📋 Résultats Détaillés": show_detailed_results_page
    }
    
    selected_page = st.sidebar.selectbox("Choisir une page:", list(pages.keys()))
    
    # Afficher la page sélectionnée
    if selected_page in pages:
        pages[selected_page](dashboard)

if __name__ == "__main__":
    main()