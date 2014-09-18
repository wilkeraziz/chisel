chisel
======

`chisel` is an SMT decoder based on sampling. It implements a number of sampling strategies and different decision rules for SMT.
It works with phrase-based and hierarchical phrase-based models. Lots of its features are not yet availabile in the repository, but that won't last long.


## Requirements

I have added a few features to some of the dependencies below, these changes might not yet have made their way upstream, so you might need to download them from my forks.

* kenlm
* cdec 
* jieba (for Chinese segmentation)

## Installing

I recommend you use virtualenv

    virtualenv chiselenv

* kenlm

        git clone https://github.com/wilkeraziz/kenlm.git 
        source chiselenv/bin/activate
        python setup.py install

* cdec

        git clone https://github.com/wilkeraziz/cdec.git 
        autoreconf -ifv
        ./configure
        make
        source chiselenv/bin/activate
        cd python
        python setup.py install

* jieba

        source chiselenv/bin/activate
        pip install jieba


## Citation

We are still preparing our paper, so please be kind and wait a bit ;)

## Development

`chisel` is developed by Wilker Aziz at the University of Sheffield.

## License

Copyright (C) 2014 Wilker Aziz

Licensed under the Apache License Version 2.0.
