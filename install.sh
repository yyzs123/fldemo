source "D:/miniconda3/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

echo "conda create -n fldemo python=3.9.7"
conda create -n fldemo python=3.9.7

echo "conda activate fldemo"
conda activate fldemo

# Install PyTorch (please visit pytorch.org to check your version according to your physical machines
conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install numpy
conda install matplotlib
conda install scikit-learn