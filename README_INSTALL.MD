conda create -n clustergen python=3.6 --yes
conda activate clustergen
conda install -c conda-forge tqdm --yes
conda install -c anaconda networkx --yes
conda install -c conda-forge openbabel=2.4.1 --yes
conda install -c conda-forge rdkit --yes
conda install scipy --yes
conda install -c conda-forge matplotlib --yes
conda install -c conda-forge svgutils --yes
conda install -c conda-forge cairosvg --yes
conda install -c anaconda scikit-image --yes
conda install -c anaconda joblib

conda create -n pymolenv --yes
conda activate pymolenv
conda install -c schrodinger pymol --yes
conda install -c conda-forge tqdm --yes

