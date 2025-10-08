# Musical Note Recognition and Classification System

## Overview

This project implements a comprehensive machine learning system for automatic musical note recognition and classification. The system is designed to identify and classify musical notes from audio input, including both pre-recorded audio files and real-time microphone input.

## Project Description

The Musical Note Recognition and Classification System utilizes advanced audio signal processing techniques combined with multiple machine learning algorithms to accurately identify musical notes across the chromatic scale (A, A#, B, C, C#, D, D#, E, F, F#, G, G#). The system processes audio data through sophisticated feature extraction methods and employs an ensemble of trained classifiers to provide robust and accurate predictions.

## Features

### Audio Processing Capabilities
- Support for MP3 audio file formats
- Real-time microphone input processing
- Comprehensive audio feature extraction using librosa library
- Sample rate normalization and audio preprocessing

### Feature Extraction
The system extracts a comprehensive set of audio features including:
- **MFCC (Mel-Frequency Cepstral Coefficients)**: 13 coefficients with mean and standard deviation statistics
- **Chroma Features**: Pitch class profiles representing harmonic content
- **Spectral Features**: 
  - Spectral centroid (brightness measure)
  - Spectral rolloff (spectral shape descriptor)
  - Spectral bandwidth (spectral spread measure)
- **Temporal Features**:
  - Zero crossing rate (time-domain feature)
  - RMS energy (amplitude measure)
- **Rhythmic Features**: Tempo estimation
- **Pitch Analysis**: Fundamental frequency extraction and statistics

### Machine Learning Models
The system implements an ensemble of 10 different machine learning algorithms:
1. **K-Nearest Neighbors (KNN)**: Instance-based learning with k=5
2. **Support Vector Machine (SVM)**: RBF kernel with probability estimates
3. **Random Forest**: Ensemble of 200 decision trees with controlled depth
4. **Logistic Regression**: One-vs-Rest multi-class approach
5. **Decision Tree**: Single tree with pruning parameters
6. **Naive Bayes**: Gaussian assumption for continuous features
7. **Neural Network**: Multi-layer perceptron with 3 hidden layers (200-100-50 neurons)
8. **Gradient Boosting**: Sequential ensemble with 200 estimators
9. **AdaBoost**: Adaptive boosting with SAMME algorithm
10. **Linear Discriminant Analysis**: Linear dimensionality reduction classifier

### Dataset Processing
- Automated audio snippet generation with overlapping windows
- 3-second audio segments with 50% overlap ratio
- Stratified data splitting (70% training, 10% validation, 20% testing)
- Feature standardization using StandardScaler
- Class imbalance detection and handling

### Model Evaluation and Analysis
- Comprehensive performance metrics including accuracy, precision, recall, and F1-score
- Cross-validation with stratified k-fold approach
- Overfitting detection through training vs validation accuracy comparison
- Model ranking and performance comparison
- Ensemble prediction using majority voting

### Real-time Recognition Capabilities
- Live microphone input processing
- 5-second audio recording and analysis
- Real-time feature extraction and classification
- Confidence scoring for predictions
- Interactive Jupyter notebook interface with recording controls

## Technical Implementation

### Dependencies
- **Audio Processing**: librosa, sounddevice, IPython.display
- **Machine Learning**: scikit-learn (multiple algorithms)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Interactive Interface**: ipywidgets

### System Architecture
1. **Data Loading Module**: Handles MP3 file discovery and loading with flexible naming patterns
2. **Feature Extraction Engine**: Comprehensive audio analysis using librosa
3. **Dataset Generator**: Creates training datasets from audio snippets with proper labeling
4. **Model Training Pipeline**: Automated training and evaluation of multiple classifiers
5. **Prediction System**: Ensemble-based classification with confidence measures
6. **Real-time Interface**: Microphone integration with live audio processing

### Performance Characteristics
- Multi-class classification across 12 musical note categories
- Ensemble approach provides robust predictions through majority voting
- Feature vector dimensionality: 75 features per audio sample
- Real-time processing capability with configurable audio chunk sizes
- Comprehensive evaluation metrics for model selection and optimization

## Usage

The system operates through a Jupyter notebook environment (`tmrtAC.ipynb`) with sequential cell execution for:
1. Audio data loading and preprocessing
2. Feature extraction and dataset creation
3. Model training and evaluation
4. Performance analysis and comparison
5. Real-time microphone-based note recognition

## Applications

This system can be utilized for:
- Music education and training applications
- Automatic music transcription preprocessing
- Musical instrument tuning assistance
- Audio content analysis and classification
- Research in music information retrieval

## Technical Notes

The implementation emphasizes modularity and extensibility, allowing for easy integration of additional machine learning models or feature extraction methods. The ensemble approach provides robustness against individual model limitations, while the comprehensive evaluation framework ensures reliable performance assessment across different musical contexts.

