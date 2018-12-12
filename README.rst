=============================================================================
Bayesian inference of chromatin structure ensembles from population Hi-C data
=============================================================================

What is this about?
-------------------
This Python package implements a Bayesian approach to infer explicit chromatin structure ensembles from population-averaged Hi-C data (Carstens et al., submitted to biorxiv). It is based on the Inferential Structure Determination (ISD, `Rieping et al., Science 2005) <http://science.sciencemag.org/content/309/5732/303>`_) approach and extends our previous work, in which we used ISD to infer chromosome structures from single-cell data (`Carstens et al., PLOS Comput. Biol. 2016 <http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005292>`_).

The code comprises classes implementing a likelihood for population-averaged contact data, prior distributions for chromatin bead coordinates and nuisance parameters, scripts to run simulations, example config files, unit tests, documentation and various utility functions and classes.

Setting up
---------------
The ensemble_hic package has several dependencies:

  - Python 2.7
  - numpy (`download <https://pypi.python.org/pypi/numpy>`_)
  - the Computational Structural Biology Toolbox (CSB, `download <https://github.com/csb-toolbox/CSB>`_)
  - my binf (`download <http://bitbucket.org/simeon_carstens/binf>`_) package
  - my rexfw (`download <http://bitbucket.org/simeon_carstens/rexfw>`_) package
  - an OpenMPI implementation

That's it, I believe.
      
Install these and you should be able to install ensemble_hic by typing::

    $ python setup.py install
    
possibly with the ``--user`` option, if you don't have administrator privileges.

Tests / documentation
---------------------
For most classes, there are unit tests to make sure things work as they're supposed to. You can run the tests by typing::

    $ cd tests
    $ python run_test.py
    
The API is fully documented. I tried to pay attention to some coding practices, so hopefully it is not too bad to read / use / adapt. There are also config files for the simulations we performed in out preprint. They can be found in ``config_files``. In them, paths to the input data file, the replica schedule and the simulation output folder have to be set. Then, you can launch a simulation with::

$ cd scripts/
$ mpirun -n 100 python run_simulation /path/to/config_file.cfg
$ python example_script.py

The ``-n`` argument (here, 100) specifies the number of processes, which equals to ``# replicas + 1``. You can find the number of replicas in the config files. You most certainly want to run the simulations on a HPC system, as the simulations all require at least around 50 processes.

In ``scripts/`` there's a colorful assortment of analysis and test scripts, which I still have to tidy up. Feel free to play around (some of that requires ``matplotlib``).

Contact
-------
If you have questions, don't hesitate to drop me a message.

Reference
---------
If you discuss / use / extend this software, please cite our preprint (or, at some later point, the peer-reviewed paper), as follows:

``Carstens, S., Nilges, M., Habeck, M. Bayesian inference of chromatin structure ensembles from population Hi-C data. bioRxiv, tba``

License
-------
ensemble_hic is open source and distributed under the OSI-approved MIT license::

    Copyright (c) 2018 Simeon Carstens

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE 
