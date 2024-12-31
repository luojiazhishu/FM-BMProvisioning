# Bluetooth Mesh Provisioning
A formal verification of the Bluetooth Mesh provisioning protocol using Tamarin Prover.

# Results Explore

## Verification results for Unfixed BM provisioning protocol
https://luojiazhishu.github.io/FM-BMProvisioning/Results/Unfixed/results.html

## Verification results for Fixed BM provisioning protocol
https://luojiazhishu.github.io/FM-BMProvisioning/Results/Fixed/results.html


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
