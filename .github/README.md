# Implémentation de Github Actions


---
## Workflows

- **release** : publish a code release
  - create a relesae on Github
  - publish a releas on pypi.org
- **update-doc** : update documentation

---
## Actions

- build-package : buid pacakge with python build module
- create-github-release : create a release of the code on github. 
- publish-to-pypi : publish e package to pypi with twine

---
## Secrets à configurer dans GitHub

Dans Settings > Secrets and Variables > Actions : Repository secrets

- PYPI_API_TOKEN : le token PyPI ou TestPyPI (selon le contexte)