The examples can be run as 

`python script_filename.py`

If called with arguments, 
* The first is interpreted as path to a json file to load the calculation config from. 
If no arguments are given, the configuration is loaded from the provided file with the same stem as the `.py` file.
I.e. the call
`python examples/tJ_spin_half_ladder/all_ks_eigvals.py examples/tJ_spin_half_ladder/rung_current_spectrum.json`
calculates the eigenpairs, but not the current operator.

* The optional second argument is interpreted as a json logging configuration file passed onto `logging.config.dictConfig` in the standard library.
The package does not configure the loggers, only the examples do.

By default, `calculations/` and `figures/` folders are created in the current working directory 
to save the results into. The folders are specified as variables on the `Hamiltonian_Config` class.

The default logging creates `logs/` folder in the CWD.
