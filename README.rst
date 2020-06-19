=============================================================================
Bayesian inference of chromatin structure ensembles from population Hi-C data
=============================================================================

What is this package about?
---------------------------
This Python package implements a Bayesian approach to infer explicit chromatin structure ensembles from population-averaged Hi-C data (`Carstens et al., PNAS 2020 <https://www.pnas.org/content/117/14/7824.short>`_). It is based on the Inferential Structure Determination (ISD, `Rieping et al., Science 2005) <http://science.sciencemag.org/content/309/5732/303>`_) approach and extends our previous work, in which we used ISD to infer chromosome structures from single-cell data (`Carstens et al., PLOS Comput. Biol. 2016 <http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005292>`_).

The code comprises classes implementing a likelihood for population-averaged contact data, prior distributions for chromatin bead coordinates and nuisance parameters, scripts to run simulations, example config files, unit tests, documentation and various utility functions and classes.

Setting up
---------------
The ensemble_hic package has several dependencies:

- Python 2.7
- numpy (`download <https://pypi.python.org/pypi/numpy>`_)
- mpi4py (`download <https://pypi.org/project/mpi4py/>`_)
- the Computational Structural Biology Toolbox (CSB, `download <https://github.com/csb-toolbox/CSB>`_)
- my binf (`download <http://bitbucket.org/simeon_carstens/binf>`_) package
- my rexfw (`download <http://bitbucket.org/simeon_carstens/rexfw>`_) package
- an MPI (Message Passing Interface) implementation. I use OpenMPI.

That's it, I believe.
      
For all the Python dependencies, it's best to use ``pip`` in a virtual environment so as not to mess up your system Python installation.
With it, the required dependencies can be installed by running::

    $ pip install -r requirements.txt

Once this is done, install the ``ensemble_hic`` package by running::

    $ pip install .
    
possibly with the ``--user`` option, if you don't have administrator privileges.

Note: to obtain the version of the code with which the results in our PNAS paper were produced, check out the following commit: ``cfeef4a``. In later commits, I did some changes in the Cython code to improve speed, but which don't change the calculations in any way. 

Another option is to run the provided Nix shell (``shell.nix``). `Nix <https://nixos.org>`_ is a purely functional package manager. You can install Nix according to the `documentation <https://nixos.org/download.html>`_. Once this is done, just enter::

  $ nix-shell

and get yourself a nice cup of coffee while Nix downloads and builds ALL dependencies (all the way down to the system level) of the ``ensemble_hic`` package and the package itself. Once this is done, you are dropped into a shell where you can, for example, start a Python interpreter and ``import ensemble_hic`` or run a simulation.  


Tests / documentation
---------------------
For most classes, there are unit tests to make sure things work as they're supposed to. You can run the tests by typing::

    $ cd tests
    $ python run_test.py
    
The API is fully documented. I tried to pay attention to some coding practices, so hopefully it is not too bad to read / use / adapt. There are also config files for the simulations we performed for our preprint. They can be found in ``config_files``. In them, paths to the input data file (in a sparse matrix format, three colums: bead i, bead j, count (or frequency) between beads i and j), the replica schedule and the simulation output folder have to be set. Then, you can launch a simulation with::

$ cd scripts/
$ mpirun -n 100 python run_simulation /path/to/config_file.cfg

The ``-n`` argument (here, 100) specifies the number of processes, which equals to ``# replicas + 1``. You can find the number of replicas in the config files. You most certainly want to run the simulations on a HPC system, as the simulations all require at least around 50 processes.

Contact
-------
If you have questions, don't hesitate to drop me a message.

Reference
---------
If you discuss / use / extend this software, please cite our paper:

``Carstens, S., Nilges, M., & Habeck, M. (2020). Bayesian inference of chromatin structure ensembles from population-averaged contact data. Proceedings of the National Academy of Sciences, 117(14), 7824-7830.``

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
