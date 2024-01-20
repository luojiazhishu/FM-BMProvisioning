# Bluetooth Mesh Provisioning
A formal verification of the Bluetooth Mesh provisioning protocol using Tamarin Prover.

# Usage
Use m4 macro processor to generate the Makefile.
Submodels will be generated and verified by `make`.
```
m4 Makefile.m4 > Makefile
make ALL    # Or other labels, see the Makefile
```
The location of results can be modified in the Makefile.

Use the following command to collect the results in an HTML file.
```
python3 collect.py Results/Fixed    # or Results/Unfixed
```
