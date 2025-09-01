<p align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/continualist/space-ai/refs/heads/main/docs/_static/images/logo.jpeg?"><img width=450 alt="spaceai-logo" src="https://raw.githubusercontent.com/continualist/space-ai/refs/heads/main/docs/_static/images/logo.jpeg"/>
</picture>
</p>

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Feclypse-org%2Feclypse%2Fmain%2Fpyproject.toml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

[![Import sorted with isort](https://img.shields.io/badge/isort-checked-brightgreen)](https://pycqa.github.io/isort/)
[![IMport cleaned with pycln](https://img.shields.io/badge/pycln-checked-brightgreen)](https://github.com/hadialqattan/pycln)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Doc style: docformatter](https://img.shields.io/badge/doc%20style-docformatter-black)](https://github.com/PyCQA/docformatter)

SpaceAI is a comprehensive library designed for space mission data analysis and machine learning model benchmarking. It provides tools for data preprocessing, model training, and evaluation, specifically tailored for space-related datasets. The library includes implementations of various machine learning models, such as ESNs (Echo State Networks) and LSTMs (Long Short-Term Memory networks), and offers a range of utilities to facilitate the development and testing of these models. With SpaceAI, researchers and engineers can streamline their workflow and focus on deriving insights from space mission data.

Here's the link to the documentation: [https://space-ai.readthedocs.io/en/latest/](https://spaceai.readthedocs.io/en/latest/)

## Installation

To install SpaceAI and all its dependencies, you can run the following commands:
```bash

pip install spaceai

```

## Report del progetto

Questo repository estende e riorganizza il progetto [Space-AI](https://github.com/continualist/space-ai) di Continual-IST, mantenendone la struttura di base ma aggiungendo nuovi modelli e strumenti per lo studio dell'**anomaly detection** in scenari di *continual learning*. Nella versione originale erano già disponibili esperimenti con **Echo State Network (ESN)** e **LSTM** sui dataset NASA, insieme a notebook dimostrativi e a un'interfaccia a riga di comando per riprodurre gli studi.

### Task e dataset NASA

Il focus rimane l'individuazione di anomalie nei segnali di telemetria delle missioni **SMAP** e **MSL**, forniti dal [NASA Frontier Development Lab](https://github.com/nasa/telemanom). Ogni canale è trattato come un task distinto: il modello apprende a prevedere il valore successivo e l'anomalia viene rilevata confrontando l'errore di previsione con una soglia dinamica. Il metodo utilizzato è il *Non‑Parametric Dynamic Thresholding* proposto da Telemanom, che applica una **media mobile esponenziale** agli errori e stima una soglia non parametrica capace di adattarsi alle variazioni del segnale.

### Estensione con PNN e backbone LSTM

In questo lavoro è stato introdotto un modello di **Progressive Neural Network (PNN)** con backbone **LSTM** per migliorare il trasferimento tra canali. L'implementazione prende spunto da quella disponibile in [Avalanche](https://avalanche.continualai.org/) ma è stata adattata per:

* gestire un compito di **regressione** anziché classificazione, sostituendo il `MultiTaskClassifier` con un `MultiHeadRegressor`;
* utilizzare un *encoder* LSTM in ciascuna colonna, così da modellare le dipendenze temporali presenti nei dati di telemetria.

La PNN è composta da colonne che, una volta addestrate su un sensore, vengono congelate e utilizzate tramite **adattatori laterali** (lineari o MLP) per fornire conoscenza alle colonne successive. In uscita ogni colonna espone teste indipendenti per i diversi canali, realizzando un `MultiTaskModule` specializzato nella previsione multivariata.

### Workflow di training

Lo script `examples/run_nasa_pnn_lstm.py` mostra il workflow completo: per ogni gruppo di canali (derivato dal prefisso del sensore) viene istanziata una nuova PNN e, all'interno del gruppo, i canali vengono processati in sequenza. L'addestramento di ciascun canale è gestito dal `CLTrainer`, estensione della strategia *Naive* di Avalanche con **early stopping** e utility per il *continual learning*. Il benchmark è stato ampliato con il metodo `run_pnn`, che automatizza le fasi di training, validazione e calcolo delle metriche per ogni esperienza.

### Telemanom e confronto con Space-AI

Come nel progetto originario, la rilevazione delle anomalie è effettuata dal modello **Telemanom**, basato sul *Non‑Parametric Dynamic Thresholding* dell'errore di previsione. L'algoritmo analizza i residui filtrati con EWMA in finestre scorrevoli e segnala sequenze che superano la soglia dinamica, includendo un filtro di *pruning* per rimuovere segnalazioni troppo ravvicinate. Rispetto alla versione in Space-AI, qui Telemanom è integrato nel ciclo di training della PNN e utilizzato per valutare le prestazioni su più sensori senza necessità di riaddestramento completo.

### Risultati e prospettive

Al termine delle esperienze gli indicatori (precision, recall, F1) vengono aggregati in `results.csv`, consentendo di confrontare rapidamente configurazioni diverse. La modularità di questa estensione permette di sperimentare nuove architetture e strategie di training mantenendo compatibilità con l'infrastruttura di Space-AI.

## NASA PNN-LSTM (Continual Learning)

See `docs/experiments/nasa_pnn_lstm.md` for a walkthrough using a Progressive
Neural Network with LSTM encoders on the NASA SMAP/MSL datasets.

## Credits
We thank [eclypse-org](https://github.com/eclypse-org) and [Jacopo Massa](https://github.com/jacopo-massa) for the structure and the template of the documentation!
