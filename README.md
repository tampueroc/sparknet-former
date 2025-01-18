# SparkNet+Former

## Purpose and General Overview of the Model Architecture

**SparkNet+Former** is a hybrid deep learning architecture designed to predict fire dynamics by integrating **spatial, temporal, and environmental data**. The model combines the local feature extraction capabilities of **Convolutional Neural Networks (CNNs)** with the long-range sequence modeling power of **Transformers**, enabling accurate modeling of fire behavior over time.

The architecture processes three primary inputs:
1. **Fire State Bitmask Sequence**: A sequence of fire spread snapshots.
2. **Static Landscape Data**: Geographical and environmental data that remain constant.
3. **Wind Inputs**: Time-varying scalar inputs for wind direction and magnitude.

The model generates predictions for the next timestep’s fire state, enabling real-time monitoring and forecasting in fire management systems.

---

## Architecture

The architecture is modular and consists of the following components:

### 1. **FireStateEncoder (CNN Encoder)**
- **Purpose**: Extracts spatial features from fire state bitmask sequences at each timestep.
- **Key Features**:
  - Captures localized patterns like fire spread clusters.
  - Outputs spatial embeddings for further temporal modeling.

### 2. **StaticLandscapeEncoder (CNN Encoder)**
- **Purpose**: Encodes static landscape data (e.g., vegetation, topography) into spatial embeddings.
- **Key Features**:
  - Provides static context to enhance fire spread predictions.
  - Outputs fixed embeddings aligned with the fire bitmask grid.

### 3. **TemporalTransformerEncoder**
- **Purpose**: Models temporal dependencies across the sequence of spatial embeddings.
- **Key Features**:
  - Captures long-range relationships between timesteps.
  - Integrates wind inputs into the temporal sequence.

### 4. **FeatureFusion**
- **Purpose**: Combines embeddings from multiple sources (fire bitmask, static landscape, and wind inputs).
- **Key Features**:
  - Supports concatenation or projection-based fusion methods.
  - Produces a unified spatiotemporal representation for decoding.

### 5. **TransposedConvDecoder**
- **Purpose**: Reconstructs the predicted fire state bitmask for the next timestep.
- **Key Features**:
  - Upsamples the fused representation into a full-resolution prediction.
  - Incorporates skip connections from the encoders to preserve spatial details.

