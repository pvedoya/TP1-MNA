# TP1-MNA

**Instrucciones de configuración**
Previo a ejecutar el programa, debe realizar la configuración del mismo. Esto se hace editando el archivo
 'configuration.ini' dentro de la raíz del proyecto, donde usted puede modificar los siguientes valores:

1)IS_KPCA         = [TRUE | FALSE] # determina si se va a usar PCA (FALSE) o KPCA (TRUE).
2)KPCA_DEGREE     = [INTEGER] # determina con que grado va a realizarse KPCA, de estar en TRUE IS_KPCA.
3)PHOTO_SET       = [STRING] # path al directorio que tiene las fotos de las personas.
4)IS_VIDEO        = [TRUE | FALSE] # determina si se usara el sistema de reconocimiento facial (TRUE), 
                                  # o si se usará una foto dada (FALSE).
5)SVM_C           = [INTEGER] # parámetro de regularidad del SVM. 
6)SVM_ITER        = [INTEGER] # cantidad de iteraciones que realiza el SVM.
7)HEIGHT          = [INTEGER] # altura de las imágenes en el path (deben ser todas iguales).
8)WIDTH           = [INTEGER] # ancho de las imágenes en el path (deben ser todas iguales).
9)PEOPLE_PER_SET  = [INTEGER] # cuántas carpetas con personas distintas se encuentran en el path.
10)IMG_PER_PERSON = [INTEGER] # cuántas imagenes van a ser tomadas por persona.
11)EIGENVECTORS   = [INTEGER] # cuántos autovectores se utilizan como base de representacion de las imágenes.
12)PHOTO          = [STRING] # si IS_VIDEO está en FALSE, el programa buscará la imagen en este path para asignar 
                             # a quien corresponde.

Aclaración: Si usted asigna IS_KPCA como TRUE, al inicializar el programa le aparecerá una pantalla de reconocimiento
 facial, donde le pedimos que se alinee con la cámara hasta que aparezca un recuadro verde, 
 indicando que su cara fue reconocida, y presione la tecla "s" para sacar la foto, luego de ésto el programa procederá 
 automáticamente a identificar la imagen. 


**Instrucciones de ejecución**
Para el funcionamiento de este proyecto, es necesario tener instalado python3, pip y venv (si no los tiene puede ver 
como instalarlos aqui: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). 
Luego, parado en la raíz del proyecto, ejecute los siguientes comandos:

1)python3 -m venv .venv
2)source .venv/bin/activate
3)python -m pip install -r requirements.txt (este archivo fue configurado para python 3.8, si su versión es anterior,
 en vez de correr este comando, ejecute el archivo "requirements.sh")
4)python3 code/main.py (Esto iniciará el programa)

Aclaración: los pasos 1-3 sólo deben realizarse la primera vez, si desea volver a ejecutar el programa, simplemente
 vuelva a correr "python3 main.py". Si sale de la consola, para volver a posicionarse en el entorno virtual, vuelva
  a la raiz del proyecto y corra "source .venv/bin/activate", y luego "python3 main.py" para ejecutar el programa.


