# Cataract Surgery Skill Assessment

This repository explores video-based surgical skill assessment for cataract surgery using deep learning models trained on microscope video data.

## Current Baseline
- **EfficientNet-B0 + LSTM**  
  A clip-based video model that encodes short frame sequences with a CNN backbone and models temporal dynamics using a multi-layer LSTM, with predictions aggregated at the video level.

## Baseline Models in Development
- **Clip-indexed video models**  
  Expanding training from one clip per video to many clips per case to improve temporal coverage and stability.
- **Phase-aware skill models**  
  Incorporating surgical phase information to condition or guide skill prediction.
- **Attention-based video models**  
  Adding spatial and/or temporal attention to focus on critical surgical actions.
- **Alternative temporal backbones**  
  Exploring 3D CNNs and transformer-based video models as complementary baselines.

## Evaluation
- Skill assessment is evaluated at the video level by aggregating predictions from multiple clips per surgery.

This work is intended to establish strong, reproducible baselines for future research in automated surgical skill assessment.
