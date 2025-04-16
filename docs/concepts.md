# Ocean4DVarNet

---
## **Core Concept**

4DVarNet blends:

- **4D-Variational data assimilation (4D-Var)**: A method used to optimally combine a numerical model with observations over a time window.
- **Deep learning**: Neural networks learn to model complex dynamics and correct for model errors.
  
It learns to **reconstruct missing data** over both space and time (the "4D" stands for 3D space + time).

---
## **How It Works**

- It uses a neural network **within a variational optimization loop**.
- Given **partial observations** (like satellite data), it:
    1. Predicts the full state over time.
    2. Minimizes a **loss function** that balances between:
        - Staying close to the observations.
        - Being consistent with a learned dynamical prior (the network).
    3. Refines its predictions iteratively (like traditional 4D-Var).

---
## Applications

- **Oceanographic reconstructions** (e.g. Sea Surface Height fields).
- **Climate modeling**.
- **Atmospheric reanalysis**.
- Any setting where data is sparse, noisy, and spatiotemporal.


---
## 4DVarNet papers

- Fablet, R.; Amar, M. M.; Febvre, Q.; Beauchamp, M.; Chapron, B. END-TO-END PHYSICS-INFORMED REPRESENTATION LEARNING FOR SA℡LITE OCEAN REMOTE SENSING DATA: APPLICATIONS TO SA℡LITE ALTIMETRY AND SEA SURFACE CURRENTS. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 2021, V-3–2021, 295–302. [https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021](https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021).
- Fablet, R.; Chapron, B.; Drumetz, L.; Mmin, E.; Pannekoucke, O.; Rousseau, F. Learning Variational Data Assimilation Models and Solvers. Journal of Advances in Modeling Earth Systems n/a (n/a), e2021MS002572. [https://doi.org/10.1029/2021MS002572](https://doi.org/10.1029/2021MS002572).
- Fablet, R.; Beauchamp, M.; Drumetz, L.; Rousseau, F. Joint Interpolation and Representation Learning for Irregularly Sampled Satellite-Derived Geophysical Fields. Frontiers in Applied Mathematics and Statistics 2021, 7. [https://doi.org/10.3389/fams.2021.655224](https://doi.org/10.3389/fams.2021.655224).

