
cd venv_testing_39
#se puede ejecutar directamente desde la terminal a partir de esta linea
mkdir docs
# New-Item es el equivalente a `touch` en mac 
# `;` es el equivalente a `&`
mkdir scripts ; New-Item scripts/__init__.py
mkdir notebooks
mkdir data ; cd data ; mkdir raw ; mkdir processed ; cd ..