# Phoenix – AI-Powered Urban Damage Assessment & Reconstruction

## Overview

Phoenix is an AI-driven system designed to support post-disaster urban damage assessment and recovery planning using satellite imagery and geospatial intelligence.

The project applies deep learning and computer vision to analyze pre-event and post-event satellite images, identify building damage levels, and generate spatial outputs to support urban recovery assessment.

## Key Features

- **Satellite Image Analysis:** Processes paired pre-disaster and post-disaster satellite imagery.
- **Deep Learning-Based Damage Assessment:** Uses a ResNet34-based U-Net architecture to estimate building damage severity.
- **Geospatial Processing:** Integrates satellite imagery with geographic data for spatial analysis and visualization.
- **Urban Recovery Support:** Converts damage assessment outputs into structured spatial information to support post-disaster analysis.

## Methodology

The system follows an integrated workflow:

1. Preprocessing and alignment of pre-event and post-event satellite imagery.
2. Deep learning-based building damage segmentation.
3. Extraction of spatial damage information.
4. Geospatial processing and visualization of assessment results.

## Study Area

The project includes a case study focused on urban damage assessment in Rafah and the Al-Rimal area of Gaza.

Model development and evaluation use the xBD building damage assessment dataset, with a separate application of the pipeline to the study area.

## Technologies

- **Programming:** Python
- **Deep Learning:** PyTorch
- **Computer Vision:** Image Processing, U-Net, ResNet34
- **Geospatial Analysis:** Rasterio, Google Earth Engine, GIS, OpenStreetMap
- **Data Processing & Visualization:** NumPy, Matplotlib

## Project Objective

Phoenix aims to bridge deep learning-based satellite image analysis and geospatial intelligence to provide useful information for post-disaster urban assessment and reconstruction planning.
