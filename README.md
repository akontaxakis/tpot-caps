

# CAPS–TPOT Integration

This repository contains the integration of **CAPS** with **TPOT**. 
This if forked repository from EpistasisLab/tpot

The main CAPS repository, including instructions on how to integrate CAPS with other AutoML tools, can be found here:  
*<insert link>*.

---

## Overview

**CAPS** acts as a middleware between the **generation** and **evaluation** stages of an AutoML system.  
Its purpose is to select which pipelines should be evaluated by using a **weighted ratio between expected performance and expected cost**, controlled by a parameter **λ (lambda)**.

### How CAPS Works

1. **Cost & Performance Estimation**  
   CAPS trains models that predict:
   - the expected **execution cost** of a generated pipeline  
   - the expected **performance** of the same pipeline  

2. **Pipeline Selection (NP-hard problem)**  
   Selecting the optimal set of pipelines is NP-hard (see our paper for details).  
   CAPS provides two approximation algorithms:
   - **Greedy**
   - **Beam Search**

### Baseline Algorithms Included

This repository also includes two additional selection approaches:

- **flaml-like**  
  Inspired by FLAML; uses *observed* performance and cost from previous iterations (no predicted values).

- **ratio**  
  Uses CAPS’ predicted values; selects based on  
  **expected_performance / expected_cost**.

---

## Testing CAPS–TPOT

An example run using the **Dionis** dataset is included in this repository.

All datasets used in our experiments can be found here:  
*https://automl.chalearn.org/data*.

---

## CAPS–TPOT Parameterization

Below is a typical configuration snippet for CAPS inside TPOT:

```python
mode = "CAPS"                        # CAPS-specific parameter
sel_algo = "caps-greedy"           # options: caps-greedy, caps-beam_search, flaML-like, ratio
lamda = 0.5                      # used with caps-greedy and caps-beam_search
selection = 100              # CAPS-specific parameter
data_id = "dionis"                           # unique identifier for history graph + logging files
```

## Minimal example

```python
from AutoML_data_manager.data_manager import DataManager
from tpot import TPOTClassifier

if __name__ == "__main__":
    data_id = "dionis"

    dm = DataManager(
        data_id,
        "datasets",
        replace_missing=True,
        verbose=3,
    )
    X = dm.data["X_train"]
    y = dm.data["Y_train"]

    tpot = TPOTClassifier(
        mode="CAPS",
        sel_algo="caps_greedy",
        lamda=0.5,
        selection=50,
        data_id=data_id,
        population_size=100,
        generations=10,
        cv=2,
        template="Transformer-Classifier",
        random_state=7777,
        config_dict="TPOT light",
    )

    tpot.fit(X, y)
```
### Contact

For any questions don't hesitate to ask:

Antonios Kontaxakis, antonios.kontaxakis-ATNOSPAM-ulb.be
