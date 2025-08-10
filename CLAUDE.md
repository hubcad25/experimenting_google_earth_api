# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental project focused on combining Google Earth/Maps API with image recognition algorithms to identify houses with swimming pools within a specified perimeter.

## Core Objectives

- Experiment with Google Earth Engine API or Google Maps API for satellite/aerial imagery
- Implement computer vision algorithms for pool detection in residential areas
- Create a system to analyze houses within defined geographical boundaries
- Develop image classification capabilities to distinguish pools from other features

## Key Technologies Expected

- Google Earth Engine API or Google Maps Static API for imagery acquisition
- Image processing libraries (likely OpenCV, PIL/Pillow, or similar)
- Machine learning frameworks for image recognition (TensorFlow, PyTorch, or scikit-image)
- Geospatial libraries for coordinate handling and area definition
- API authentication and rate limiting management
- Algos open-source de détection de piscine:
    - https://github.com/yacine-benbaccar/Pool-Detection
    - https://danielcorcoranssql.wordpress.com/2019/01/13/detecting-pools-from-aerial-imagery-using-cv2/
    - https://github.com/Jonas1312/swimming-pool-detection
    - https://github.com/danielc92/cv2-pool-detection
    - https://github.com/AlexisBaladon/Pool-Analyzer

## Development Notes

- API keys for Google services should be stored in environment variables or config files (never committed)
- Image processing may require significant computational resources
- Consider batch processing for large geographical areas
- Implement proper error handling for API rate limits and failures
- Test with various image qualities and lighting conditions