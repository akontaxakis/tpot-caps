

# CAPS–TPOT Integration

This repository contains the integration of **CAPS** with **TPOT**.

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
*<insert link>*.

---

## CAPS–TPOT Parameterization

Below is a typical configuration snippet for CAPS inside TPOT:

```python
mode = mode                        # CAPS-specific parameter
sel_algo = "caps-greedy"           # options: caps-greedy, caps-beam_search, flaML-like, ratio
lamda = lamda                      # used with caps-greedy and caps-beam_search
selection = selection              # CAPS-specific parameter
id = ...                           # unique identifier for history graph + logging files
