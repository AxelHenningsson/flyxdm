# flyxdm [![ubuntu-linux conda-build](https://github.com/AxelHenningsson/flyxdm/actions/workflows/python-package-conda-linux-py39.yml/badge.svg)](https://github.com/AxelHenningsson/flyxdm/actions/workflows/python-package-conda-linux-py39.yml) [![macos conda-build](https://github.com/AxelHenningsson/flyxdm/actions/workflows/python-package-conda-macos-py39.yml/badge.svg)](https://github.com/AxelHenningsson/flyxdm/actions/workflows/python-package-conda-macos-py39.yml)


A demo implementation of fast, factorized, on the fly, tomographic far field X-ray diffraction microscopy imaging.

In the setting of scanning-3DXRD/HEDM we show how a special block-partioned system matrix factorization can be exploited to achieve a memory efficient and fast GPU-accelerated diffraction model implementation. This repository serves as a supplementary material for the publication:

Henningsson, A. & Hall, S. A. (2023).  
[An efficient system matrix factorization method for scanning diffraction based strain tensor tomography](https://doi.org/10.1107/S2053273323008136).  
*Acta Crystallographica Section A: Foundations and Advances*, **79**(6), 542–549.

# Demo
After installing the lib, please head to the demo folder and checkout the `simulate_diffraction.py` script which generates diffraction data into the `demo/data` folder. The data represents a single grain slice of alpha-quartz featuring both intragranular misorientations as well as strain, as can be seen below:

![git_simul](https://github.com/AxelHenningsson/flyxdm/assets/31615210/abe6d16d-64e3-461a-b44f-43212a660ca4)


The scripts `reconstruct_strain.py` and `reconstruct_ub.py` illustrate how intragranular strain and orientation reconstruction can take place using the simulated diffraction data.

# Install
Requirements: conda python environment, with 64 bit Python 3.7, 3.8 or 3.9.

Install anaconda and create a new conda environment in your terminal as
```
    conda create -n flyxdm python=3.9
    conda activate flyxdm
```
The ray-tracing primitives utilise the astra-toolbox for gpu acceleration. You may install these as:
```
    conda install -c astra-toolbox astra-toolbox
```
Next we install some more dependencies
```
    conda install -c conda-forge matplotlib pip xfab
```
You may now install flyxdm using pip as
```
    git clone https://github.com/AxelHenningsson/flyxdm.git
    cd flyxdm
    pip install -e .
```

# Credits
This library makes good use of the open source [astra-toolbox ray-tracing primitives.](https://github.com/astra-toolbox/astra-toolbox)
