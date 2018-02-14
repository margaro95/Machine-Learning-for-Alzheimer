# Machine Learning for Alzheimer

Este proyecto hará uso de los datos proporcionados por Ernesto Pereda.

Los datos se pueden encontrar en la carpeta /media/marcos/Seagate Expansion Drive/IFISC/alzheimer_ml

Antes de meternos a muerte con el proyecto, unas aclaraciones previas.

###
Explicación de abreviaciones

DCL == déficit cognitivo leve
QSM == queja subjetiva de memoria
No QSM == grupo de control
PLV == phase locking value
UMEC == nombre del paciente
MEGMAG == tipo de magnetómetro
NEMOS == no importa
rEC == resting eyes closed
(HO, BM) == otros atlas
AAL == automated anatomical labeling (hay un AAL2)
AVOI == anatomical volumes of interest
plv_pca_aal == principal component analysis
plv1 == no lo uses
plv2 == no lo uses
ps == podría ser "phase synchronization"

La principal carpeta es la que se llama Conectividad.
En conectividad hay tres carpetas, cada una de un grupo.
En ellas está el PLV de las bandas clásicas.

###
Explicación del PLV

La comunicación entre grandes partes del cerebro para la realización
de actos cognitivos se especula que se realice mediante el phase-lock
de las señales en esas zonas en el rango de gamma (30-80 Hz) durante
un limitado espacio de tiempo.

La PLS (phase-lock statistic) estudia estas relaciones. Se le añaden
surrogadas para permitir distinguir esta sincronía de las fluctuaciones
en el "background".

Hay un archivo en la raíz de este manual que te lo calcula en MATLAB.
También está el paper que lo explica. El link desde donde me lo descargué es este: https://es.mathworks.com/matlabcentral/fileexchange/31600-phase-locking-value

###
Cómo conseguir los inputs

En cada archivo hay seis campos de los cuales los interesantes son los de tipo estructura de datos. En info tienes una variable que se llama allpos que te da las 2459 posiciones de las fuentes. Hay otra variable que se llama model que contiene tres archivos llamados Atlas. Ernesto dice que ellos utilizan el atlas AAL porque es el que más resolución da y más próximo está al número efectivo de componentes reales (rango de la matriz de sensores == número de filas linealmente independientes de la matriz de sensores). Hay 45 AVOIs en cada hemisferio.
Al final con lo

Miguel se ha creado el vector de features de cada plv_pca de cada banda de cada paciente con
esto son los indices -> mask=triu(true(size(plv(1).plv_pca_aal)),1)
este es el vector -> plv(1).plv_pca_aal(mask)

###
Correo de Ernesto Pereda para Miguel

Hola, Miguel. Te acabo de habilitar el acceso a la carpeta en musk en la
que están los datos ...Es un directorio que se llama quejas y que como
verás tiene 4 subcarpetas. La principal por ahora es la que se llama
conectividad en la que verás otras tres, que son los 3 grupos (DCL es
Deterioro Cognitivo Leve, QSM es Quejas subjetivas de Memoria, y No QSM
es No quejas -control). Como te comenté la tarea es un poco más exigente
que en el caso de la epilepsia, ya que el grupo QSM, como tiene quejas
subjectivas, es en realidad cercano al control... En el excell que
tienes en la carpeta principal tienes para cada sujeto la edad y el
sexo, pero tenemos más datos (resultados de test neuropsicológicos, por
ejemplo) que se podrían usar para mejorar la clasificación ( o ver si lo
hacen en su caso).

   En las carpetas de conectividad veras que está ahora el PLV (voy a
añadir la información mutua, y luego vemos si merece la pena algo más)
en las bandas clásicas. Si abres un fichero en matlab verás que cada uno
contiene seis campos. 4 son tipo char, solo información (fecha en la que
se hizo el cálculo, tarea que en todos los casos es resting con ojos
cerrados rEC, el nombre del sujeto y el tipo de sensor -MAGNETÓMETROS-)

   Y luego hay dos estructuras de datos: info y plv. Como su nombre
indica, la info es con información: contiene la posición de las fuentes
(como ves 2459), y una subestructura que se llama model. Ahí lo que se
ponen son datos de los tres atlas diferentes que se usan normalmente
(cada poco sale uno nuevo, por influencia del fMRI), que son AAL (90
regiones), HO (Harvard-oxford, 64 regiones) y BM (clásico, Broadman).
Nosotros estamos usando ahora el AAL porque da buena resolución y es el
que más próximo está al número efectivo de componentes reales (rango de
la matriz de sensores) que quedan tras hacer el preprocesado (aprox. 88
de los 102 MAG que se registran). Con menos (HO, BM) se pierde
resolución, y con más se meten señales innecesarias.

    Si entras en info.model.aal veras diferentes variables, como el
nombre de las áreas (name) o su posición. Si quieres saber más sobre
atlas, esta página está muy bien

http://www.lead-dbs.org/?page_id=1004

   Resumiendo, que de las 2459 fuentes (correspondienes a un mesh de 1
cm de lado recubriendo la corteza y áreas subcorticales de materia gris)
se tienen que agrupar en regiones de interés (que es donde entran los
atlas, contestando a tu pregunta). Pero con el AAL nos quedamos al final
con matrices de 90x90 que son las ROIs consideradas con este atlas.

   Si vas ahora a plv verás que la primera columna es la banda, la
segunda los límites en Hz, la tercera la frecuencia de muestreo y luego
verás 3 matrices de PLV por cada atlas. El problema fundamental con los
atlas es que en cada ROI tienes un número diferente (por ejemplo, en AAL
aprox. 25) de fuentes por ROI. Pero necesitas pasar de 2459 fuentes a
90x90 PLVs. Lo más sencillo es definir primero una señal para cada ROI y
luego hacer el PLV de esa señal. Lo que hacemos nosotros (y mucha gente)
es coger la primera componente (pca) de las (en promedio) 25 fuentes por
cada ROI, que es lo que contiene la mayor potencia (en general, un
porcentaje muy grande). Así que habría que coger plv_pca_aal (el plv1 y
el plv2 son calculados de otra manera, que no tiene mayor interés, y el
all es todas las fuentes, sin hacer atlas, tremendamente redundante)
para cada banda y trabajar con eso

   Una forma interesante que estamos probando de hacer feature selection
a priori es escoger ad hoc un menor de la matriz completa, en el que se
tienen en cuenta solo 14 de las 90 áreas. Estas 14 son las que
corresponden a la Default Mode Network, que como sabes es un conjunto de
estructuras corticales y subcorticales que se activan en la fMRI en reposo

http://payload350.cargocollective.com/1/18/579772/9319089/F1.large_1000.jpg

incluyen el precuneus, la corteza entorrinal y el hipocampo, que son de
las más implicadas al principio del Alzheimer, y que se sabe que tienen
afectadas la conectividad. Los números de las áreas en el atlas son

2526313233343738616267688586

Con lo que la submatriz que te quedaría sería (14x14) cogiendo solo esas
filas/columnas. Es una reducción notable (de más de 4000 elementos en
90x90 a 91 en 14x14..). Otra forma de hacer FS sencilla es hacer la
estadística primero y quedarte con las ps más bajas. De todas maneras lo
hablamos

   Voy a subir en estos días también los datos de la MI normalizada, la
potencia está pero en fuentes, la tengo que pasar al atlas y eso me está
dando un poco de lata, pero no he querido esperar más

   Como ves un poco más rollo que la otra vez pero merece la pena
trabajar en fuentes, porque a efectos de publicación/interpretación es
mucho mejor.

   Cuando quieras y hayas ido echando un vistazo a los datos hacemos un
skype. Cuando ponga la MI te aviso.

   Una última cosa, ignora en el nombre de los ficheros lo de NEMOS y
UMEC; que indica solo la base de datos de procedencia a efectos internos
del lab.

   Un abrazo

#COMENTARIO DE RESULTADOS
Entrenar el modelo con todas las bandas de una vez y luego pedirle que te
clasifique el test mostrándole una sola banda no es buena idea. Se consiguen
scores del 0.59.
Es mejor idea entrenar el modelo solo con bandas de un tipo en particular y
luego pedirle que te clasifique el test que mostrándole una banda de ese tipo
en particular. Haciendo esto con la banda Gamma (6) se obtiene un score del
0.73.
