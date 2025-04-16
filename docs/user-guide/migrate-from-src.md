# Migrating old code that doesn't use the package

Migrating old code that doesn't use the Ocean4DVarNet package is not a major problem.

Version 1.0.x of the package is the code that was available in the starter repository.

All you need to do is :

- Install the package in the 1.0.x version :
``` bash
pip install "ocean4dvarnet<=1.1"
```
- Delete (or rename for backup) the src directory
``` bash
mv src src.backup
```
- Then replace in the code the function/class calls "src/xxx" with "oceandvarnet/xxx". For example, `src.models.Lit4dVarNet` becomes `oceandvarnet.models.Lit4dVarNet`.

After validation, you can delete the src directory
``` bash
rm src
```
