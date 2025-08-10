# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive pool detection system that combines Google Maps Static API for satellite imagery acquisition with multiple computer vision algorithms to identify swimming pools in residential areas. The project has evolved into a complete testing and validation framework.

## Project Structure

```
/
├── code/
│   ├── fetch_satellite_image.py          # Google Maps API image fetching
│   ├── test_all_algorithms.py            # Main testing framework
│   ├── algorithms/                       # Algorithm implementations
│   │   ├── opencv_hsv/                   # HSV color-based detection
│   │   ├── opencv_advanced/              # Advanced OpenCV techniques  
│   │   ├── resnet50/                     # ResNet50 CNN approach
│   │   └── clip_kmeans/                  # CLIP + K-means clustering
│   └── validation/
│       ├── streamlit_dashboard.py        # Interactive validation dashboard
│       └── validation_evaluator.py       # Metrics calculation
├── images/                               # Satellite imagery dataset
├── validation_dataset/                   # Manual annotations (ground truth)
├── results/                              # Algorithm outputs and test results
├── annotation_tool/
│   └── pool_annotator.py                # Manual annotation tool
└── pool_algorithms_analysis.md          # Algorithm comparison analysis
```

## Core Objectives (Implemented)

✅ **Image Acquisition**: Google Maps Static API integration with configurable zoom levels and coordinates  
✅ **Multiple Detection Algorithms**: OpenCV HSV, Advanced OpenCV, CNN approaches  
✅ **Testing Framework**: Automated testing across multiple algorithms and images  
✅ **Manual Annotation System**: Interactive tool for ground truth creation  
✅ **Validation & Metrics**: Precision, Recall, F1-Score, IoU calculations  
✅ **Interactive Dashboard**: Streamlit-based visualization and analysis  

## Key Technologies (Current)

- **API Integration**: Google Maps Static API for satellite imagery
- **Computer Vision**: OpenCV for image processing and feature detection
- **Machine Learning**: TensorFlow/Keras for CNN approaches, potential ResNet50 integration
- **Testing Framework**: Custom multi-algorithm comparison system
- **Validation**: Manual annotation tool with JSON output format
- **Visualization**: Streamlit dashboard with Plotly charts
- **Data Management**: JSON-based annotation and results storage
- **Environment Management**: Individual virtual environments per algorithm

## Algorithm Implementations

### Currently Ready
- `opencv_hsv/`: HSV color space filtering for blue pool detection
- `opencv_advanced/`: Enhanced OpenCV with morphological operations
- Basic CNN infrastructure in place

### Planned/In Progress
- `resnet50/`: Transfer learning approach with heatmap generation
- `clip_kmeans/`: Modern CLIP model with K-means segmentation

## Test Locations (Quebec)

The system has been tested on 6 residential areas:
- Chapleau, Bousquet, Gilbertdionne (Montreal suburbs)
- St-Hilaire Random, Beloeil (Montérégie)
- Lévis (Quebec City area)

## Development Notes

### API Configuration
- Google Maps API key required in `.env` file as `GOOGLE_API_KEY`
- Images fetched at zoom level 19 for optimal pool detection
- 640x640 pixel resolution (Google's free tier limit)

### Algorithm Testing
- Each algorithm runs in isolated virtual environment
- Automated testing via `test_all_algorithms.py`
- Results saved with timestamps in JSON format
- Visual outputs saved for successful detections

### Validation Process
- Manual annotation via `pool_annotator.py` with bounding box creation
- Ground truth stored in JSON format per image
- Automated metrics calculation (Precision, Recall, F1, IoU)
- Interactive dashboard for result visualization

### Performance Considerations
- Image processing requires sufficient memory for OpenCV operations
- CNN algorithms need adequate computational resources
- Consider batch processing for large datasets
- Virtual environments prevent dependency conflicts between algorithms

## Commands Reference

### Setup
```bash
cd code/algorithms/opencv_hsv && ./setup.sh    # Setup HSV algorithm
cd code/algorithms/opencv_advanced && ./setup.sh  # Setup advanced OpenCV
```

### Testing
```bash
python code/test_all_algorithms.py             # Run all algorithm tests
python code/validation/validation_evaluator.py  # Calculate validation metrics
```

### Annotation
```bash
cd annotation_tool && python pool_annotator.py  # Annotate all images
```

### Dashboard
```bash
cd code/validation && streamlit run streamlit_dashboard.py  # Launch validation dashboard
```