# Continual Learning Project

This repository extends and reorganizes the [Space-AI](https://github.com/continualist/space-ai) project by Continual-IST, keeping its core structure while adding new models and tools for studying **anomaly detection** in *continual learning* scenarios.  
In the original version, experiments with **Echo State Network (ESN)** and **LSTM** on NASA datasets were already available.

The task of this dataset is **anomaly detection in telemetry signals** from the **SMAP** and **MSL** missions, provided by the [NASA Frontier Development Lab](https://github.com/nasa/telemanom).  
Each channel is treated as a separate task: the model learns to predict the next value, and anomalies are detected by comparing the prediction error against a dynamic threshold.  

The method used is *Non-Parametric Dynamic Thresholding* from Telemanom, which applies an **exponentially weighted moving average (EWMA)** to the errors and estimates a non-parametric threshold that adapts to signal variations.

---

## Extension with PNN and LSTM backbone

This work introduces a **Progressive Neural Network (PNN)** with an **LSTM backbone**, enabling the investigation of potential knowledge transfer between channels, which was not supported in the original setup.  
The implementation of the pnn in spaceai/models builds on [Avalanche](https://avalanche.continualai.org/) but has been adapted to:

- handle a **regression** task instead of classification, replacing the `MultiTaskClassifier` with a `MultiHeadRegressor`;  
- use an LSTM *encoder* in each column to capture temporal dependencies in telemetry data.  

---

## Training Workflow

The script [`examples/run_nasa_pnn_lstm.py`](examples/run_nasa_pnn_lstm.py) implements the complete training pipeline:

1. **Channel Grouping**: Telemetry channels from the NASA SMAP/MSL datasets are grouped by sensor type, identified by their anonymized prefixes (e.g., “P-” for power, “R-” for radiation).  
   Channels in the same group share similar measurement dynamics, making them more suitable for joint modeling.  
   This grouping also avoids the overhead of training a single PNN across all channels, which would be computationally heavy and less effective due to heterogeneity.


2. **Per-group PNN instantiation**: For each sensor group, a new Progressive Neural Network (PNN) is created, with each channel in the group processed sequentially.

3. **CLTrainer handles training**:
   - Includes **early stopping** and additional utilities tailored for continual learning workflows.

4. **Benchmark automation**:
   - The new method of the nasa benchmark `run_pnn` orchestrates training, validation, and metric computation for each new channel (experience).

---

## Results and Outlook

- At the end of the experiments, performance indicators (**precision, recall, F1**) are aggregated in `results.csv`, allowing quick comparison of different configurations.  
- Since F1 also depends on the Telemanom thresholding module and tuning its hyperparameters is out of scope, model performance is not the primary focus here.  
- Instead, this project concentrates on efficiency-related metrics such as evaluation loss, training time, and the number of epochs before early stopping.  
- The goal of the experiment is to investigate whether the PNN with adapters brings improvements in efficiency or effectiveness compared to the standard method.  
- In particular, it can be hypothesized that the use of lateral adapters allows knowledge reuse across channels, leading on average to faster convergence when training a new channel.  
- Results can be inspected by running the script [`examples/read_results.py`](examples/read_results.py).
