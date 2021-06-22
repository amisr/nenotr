nenotr
======

Overview
--------

The code that is used to estimate electron density from the scattered power received by AMISRs.

Current status of this repository: code is production ready and package is installable, but we are missing documentation, setup.cfg, and CI.

Quick Start
-----------

Then, installation of `flipchem` can be accomplished using ``pip``::

    pip install git+https://github.com/amisr/nenotr.git
    
which will install a command line utility called ``calculate_nenotr``.

Testing
-------

If you want to run tests, then you should instead:

    git clone https://github.com/amisr/nenotr.git
    
then install the package with ``pip``::

    cd nenotr
    pip install .
    
and finally you can run the test in the ``tests`` directory::

    cd tests
    python test_nenotr.py

