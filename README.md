# Welcome to ocean4DVarNet ! 


The full documentation of the package available at : [https://cia-oceanix.github.io/ocean4dvarnet/](https://cia-oceanix.github.io/ocean4dvarnet/)

---
## About ocean4DVarNet

**4DVarNet** (short for **4D-Variational Network**) is a deep learning-based framework that combines ideas from data assimilation (specifically **4D-Var**) with neural networks to reconstruct high-resolution spatiotemporal data from incomplete and noisy observations, commonly applied in **Earth sciences** like oceanography and meteorology.

*Get more informations about [Ocean4DVarnet Concepts](concepts.md).*

---
## Installing

To install the package, you can use the following command:
``` bash
pip install ocean4dvarnet
```

*Get more information in the [installing](./installing.md) section*

---
##  Usage

``` python
import ocean4dvarnet
``` 

*Project getting started : [https://github.com/CIA-Oceanix/4dvarnet-starter](https://github.com/CIA-Oceanix/4dvarnet-starter)*

---
## Ocean4DVarnet Extension

This package contains the model's main functions. 

Another package is available, containing extended functionality : models, dataloader, etc. ... : Ocean4DVarnet-contrib : [https://github.com/CIA-Oceanix/ocean4dvarnet-contrib](https://github.com/CIA-Oceanix/ocean4dvarnet-contrib)


---
## Contributing

If you want to add specific models, data loaders, utilities, etc., the best way to do so is to contribute to the [Ocean4DVarnet-contrib](https://github.com/CIA-Oceanix/ocean4dvarnet-contrib) project.

If you want to contribute to the core Ocean4DVarnet Package, please follow the informations in the [contributing](./contributing.md) section.

---
## Useful links

- Project getting started : [https://github.com/CIA-Oceanix/4dvarnet-starter](https://github.com/CIA-Oceanix/4dvarnet-starter)
- 4DVarNet papers:
	- Fablet, R.; Amar, M. M.; Febvre, Q.; Beauchamp, M.; Chapron, B. END-TO-END PHYSICS-INFORMED REPRESENTATION LEARNING FOR SA℡LITE OCEAN REMOTE SENSING DATA: APPLICATIONS TO SA℡LITE ALTIMETRY AND SEA SURFACE CURRENTS. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 2021, V-3–2021, 295–302. [https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021](https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021).
	- Fablet, R.; Chapron, B.; Drumetz, L.; Mmin, E.; Pannekoucke, O.; Rousseau, F. Learning Variational Data Assimilation Models and Solvers. Journal of Advances in Modeling Earth Systems n/a (n/a), e2021MS002572. [https://doi.org/10.1029/2021MS002572](https://doi.org/10.1029/2021MS002572).
	- Fablet, R.; Beauchamp, M.; Drumetz, L.; Rousseau, F. Joint Interpolation and Representation Learning for Irregularly Sampled Satellite-Derived Geophysical Fields. Frontiers in Applied Mathematics and Statistics 2021, 7. [https://doi.org/10.3389/fams.2021.655224](https://doi.org/10.3389/fams.2021.655224).

---
## License

**Copyright IMT Atlantique/OceaniX, contributor(s) : M. Beauchamp, R. Fablet, Q. Febvre, D. Zhu (IMT Atlantique)**

**Contact person: ronan.fablet@imt-atlantique.fr**.

This software is a computer program whose purpose is to apply deep learning schemes to dynamical systems and ocean remote sensing data.

This software is governed by the CeCILL-C license under French law and abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the terms of the CeCILL-C license as circulated by CEA, CNRS and INRIA at the following URL "http://www.cecill.info". As a counterpart to the access to the source code and rights to copy, modify and redistribute granted by the license, users are provided only with a limited warranty and the software's author, the holder of the economic rights, and the successive licensors have only limited liability.

In this respect, the user's attention is drawn to the risks associated with loading, using, modifying and/or developing or reproducing the software by the user in light of its specific status of free software, that may mean that it is complicated to manipulate, and that also therefore means that it is reserved for developers and experienced professionals having in-depth computer knowledge. Users are therefore encouraged to load and test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or data to be ensured and, more generally, to use and operate it in the same conditions as regards security. The fact that you are presently reading this means that you have had knowledge of the CeCILL-C license and that you accept its terms.



