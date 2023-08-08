# AutodiffCTWorkflows

# Installation
To create an environment, and install the required packages to install this package, run:
```
conda create -n autodiffCT
conda activate autodiffCT
./install.sh
```

## Extra installation instructions to run notebooks
If you want to additionally be able to run notebooks, run:

```
conda install -c anaconda ipykernel jupyter
```

To register the environment as a kernel you can select in a notebook:

```
python -m ipykernel install --user --name=autodiffCT
```

## Extra installation instructions for Walnut rotation axis alignment
Walnut rotation axis requires flexcalc (which requires flexdata and flextomo):

1. install flexdata ```pip install flexdata```
2. install flextomo ```pip install flextomo```
3. install flexcalc ```pip install flexcalc```


## Extra installation instructions for Fuelcell phase retrieval
To load the tomobank dataset of the PSI fuelcell, run:

```
conda install h5py dxchange tomopy -c anaconda -c conda-forge
```


# Notebooks

1. BH simulated -- DONE
2. BH playdoh -- DONE
3. phase simulated
4. phase fuelcell -- DONE
5. rot axis simulated -- DONE
6. rot axis walnut -- DONE
7. TV+CNN foam -- DONE


