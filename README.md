# TP1-MNA

<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="logo.png" alt="Logo" width="234" height="115">
  </a>

<h3 align="center">Reconocimiento facial - Grupo L</h3>

## Tabla de Contenidos

- [TP1-MNA](#tp1-mna)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Empezando](#empezando)
    - [Requisitos](#requisitos)
    - [Instalación](#instalación)
  - [Modo de uso](#modo-de-uso)
    - [Correr el programa](#correr-el-programa)
      - [Parametros de configuracion](#parametros-de-configuracion)
    - [Capturar imagenes](#capturar-imagenes)

## Empezando

Instrucciones para correr el programa de reconocimiento facial.
### Requisitos

1. Tener instalado python3. [Link para instalar python](https://www.python.org/downloads)
  
2. Tener instalado pip.

3. Tener instalado venv.
   - Este es un software de ambientes virtuales, sirve para mantener todas las librerias usadas en un solo lugar y además sirve para que toda persona que corra este software lo haga en las mismas condiciones.
   - [Link para instalar venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)


### Instalación

**Estas instrucciones son para sistemas Unix.**

1. Clonar la repo

    ```sh
    git clone https://github.com/pvedoya/TP1-MNA.git
    ```

2. Entrar a la carpeta recien clonada

    ```sh
    cd TP1-MNA
    ```

3. Crear un ambiente virtual para el proyecto. 

    ```sh
    python3 -m venv .venv
    ```

4. Entrar al ambiente virtual desde la shell

    ```sh
    source .venv/bin/activate
    ```

5. Instalar los requisitos al ambiente virtual

    ```sh
    python -m install -r requirements.txt
    # Es posible que al ejecutar este programa aparezcan algunos errores
    # pero por lo general pueden ser ignorados
    ```

En caso de tener un sistema operativo diferente, hay instrucciones de como instalar y correr venv en el [link de instalación.](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

De todos modos, se recomienda utilizar sistemas Unix.

## Modo de uso

Hay dos scripts principales para ejecutar.

- `main.py`
- `capture_dataset_faces.py`

`main.py` Es el programa que corre el reconocimiento facial.

`capture_dataset_faces.py` es un script para facilitar el sacado de las fotos para usar a la hora de entrenar al software.

### Correr el programa

Para correr el programa primero hay que configurarlo. Esto se hace a traves del archivo `configuration.ini`

#### Parametros de configuracion

1. **IS_KPCA = [TRUE | FALSE]**  determina si se va a usar PCA (FALSE) o KPCA (TRUE).
2. **KPCA_DEGREE = [INTEGER]** determina con que grado va a realizarse KPCA, de estar en TRUE IS_KPCA.
3. **PHOTO_SET = [STRING]** path al directorio que tiene las fotos de las personas.
4. **IS_VIDEO = [TRUE | FALSE]** determina si se usara el sistema de reconocimiento facial (TRUE), o si se usará una foto dada (punto 12) (FALSE).
5. **SVM_C = [INTEGER]** parámetro de regularidad del SVM.
6. **SVM_ITER = [INTEGER]** cantidad de iteraciones que realiza el SVM.
7. **HEIGHT = [INTEGER]** altura de las imágenes en el path (deben ser todas iguales).
8. **WIDTH = [INTEGER]** ancho de las imágenes en el path (deben ser todas iguales).
9. **PEOPLE_PER_SET = [INTEGER]** cuántas carpetas con personas distintas se encuentran en el path.
10. **IMG_PER_PERSON = [INTEGER]** cuántas imagenes van a ser tomadas por persona.
11. **EIGENVECTORS = [INTEGER]** cuántos autovectores se utilizan como base de representacion de las imágenes.
12. **PHOTO = [STRING]** si IS_VIDEO está en FALSE, el programa buscará la imagen en este path para asignar # a quien corresponde.

Si se tiene el argumento **IS_VIDEO** como **TRUE**, al inicializar el programa le aparecerá una pantalla de reconocimiento facial, donde le pedimos que se alinee con la cámara hasta que aparezca un recuadro verde, indicando que su cara fue reconocida, y presione la tecla "s" para sacar la foto, luego de ésto el programa procederá automáticamente a identificar la imagen.

Para ejecutar el programa:

```sh
python3 main.py
```

### Capturar imagenes

Este script existe para facilitar el tomado de las fotos que iran al dataset para luego compararse entre las demas. El script automaticamente toma los valores del `configuration.ini` y se ejecuta de la siguiente manera:

```sh
python3 capture_dataset_faces.py -n [NOMBRE]
```

Ejecutandolo de la siguiente manera se creara una carpeta dentro del path seleccionado en el archivo `configuration.ini` con el nombre [NOMBRE] y se abrira una pantalla de reconocimiento facial, donde le pedimos que se alinee con la cámara hasta que aparezca un recuadro verde, indicando que su cara fue reconocida, y presione la tecla "s" para sacar una foto. Por defecto se sacarán 10 fotos (hay que presionar s una vez para cada foto) y luego el programa se cerrará automaticamente.

Se pueden modificar los paremetros para cambiar la cantidad de fotos que se sacan y el lugar donde se guardan, entre otras cosas.

Para mas información acerca de como correr este script ejecutar:

```sh
python3 capture_dataset_faces.py -h
```
