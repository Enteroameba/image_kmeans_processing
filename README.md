# Image_kmeans_processing

repositorio para el proyecto de analisis de imagenes utilizando opencv2 y kmeans clustering

## Para generar ambientes virtuales en versiones diferentes a la 3.11

1. La version deseada de python no se descarga del repositorio de python, tienen que ser descargada de la tienda de windows.
2. Generar el ambiente virtual especifico utilizando el comando:

```bash
python3.9 -m venv venv_name
```

3. Activar el ambiente virtual ejecutando el archivo `Activate.ps1`.

4. Desactivar el ambiente virtual para proceder a instalar las librerias necesarias con el comando:

 ```bash
 deactivate
 ```

## Obtener de origen las librerias y transferiras la destino

1. En el codigo de origen ejecutar el comando: 

```python3
pip list --format=freeze > requirements.txt
```

esto generara un archivo .txt con un listado de todas las paqueterias que se estan utilizando en el codigo de origen.

2. Antes de instalar las librerias en el destino puede que sea necesario actualizar el pip installer

```bash
python -m pip install --upgrade pip
```

3. Al parecer las librerias propias de anaconda "anaconda" y "conda" pueden generar errores (cuando se migra de un proyecto de spyder a uno en VScode) por lo que puede ser recomendable borrar estas de la lista de requerimientos.

4. instalar las librerias necesarias en el destino utilizando el comando:

```bash
pip install -r requirements.txt
```
