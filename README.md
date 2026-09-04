# Topic Modeling sobre reseñas de Letterboxd

**Análisis de PLN multilingüe sobre reseñas de películas estrenadas en 2024**
Alfredo Martín López · Marcos Fernández Arévalo · Pedro Pérez Blanco
Máster en Business Analytics (MUBA) — Universidad Pontificia Comillas ICADE · Datos No Estructurados

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![gensim](https://img.shields.io/badge/gensim-4.4-black)
![spaCy](https://img.shields.io/badge/spaCy-3.8-09A3D5?logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/code-MIT-blue)

---

## De qué va

Aplicamos LDA a reseñas de Letterboxd para descubrir de qué habla la audiencia de una película. El corpus es multilingüe, y eso acabó siendo el hallazgo principal: **3 de los 6 tópicos no son temas, son idiomas**, y se llevan el 32,3 % de las reseñas.

De ahí salió un experimento que no estaba en el trabajo original: repetir todo el modelado quedándonos solo con las reseñas en inglés, y comparar las dos versiones.

## Las preguntas

1. ¿De qué habla la gente cuando reseña una película?
2. ¿Qué pasa cuando se aplica LDA a un corpus con 48 idiomas mezclados?
3. Si se filtra a un solo idioma, ¿mejora el modelo?
4. ¿Mide la coherencia `c_v` lo que creemos que mide?

La tercera tiene una respuesta menos cómoda de lo que parece, y la cuarta explica por qué.

## Los datos

Dataset: [`pkchwy/letterboxd-all-movie-data`](https://huggingface.co/datasets/pkchwy/letterboxd-all-movie-data) en Hugging Face. 1,12 GB y 847.209 películas, cada una con hasta 10 reseñas.

**Licencia declarada: MIT.** Aun así, aquí no se redistribuye ningún dato, y por tres razones que conviene separar:

- Los `.parquet` intermedios pesan **32 MB** y los **regenera el notebook 1** en unos minutos. Versionar datos derivados que se reconstruyen solos es engordar el repositorio a cambio de nada.
- Contienen **texto escrito por personas reales**. Una cosa es analizarlo y otra republicarlo.
- La licencia MIT la declara quien subió el dataset, no Letterboxd ni quienes escribieron las reseñas. Sobre el texto de terceros esa declaración no alcanza.

Del dataset se usan solo `title`, `year`, `rating` y el texto de las reseñas. **El nombre de usuario no se extrae en ningún momento.**

## El pipeline

`01_preprocesamiento.ipynb` hace, en este orden:

1. Descarga el dataset de Hugging Face y filtra las películas de 2024.
2. Normaliza el texto y descarta reseñas vacías o demasiado cortas.
3. Lematiza con **spaCy** y etiqueta gramaticalmente; una parte con **flair**, que es contextual y bastante más lento, sobre una muestra.
4. Detecta el idioma con **langdetect**, con `DetectorFactory.seed` fijo para que sea determinista.
5. Escribe los `.parquet` que consume el notebook 2.

## El modelado

`02_topic_modeling.ipynb` entrena LDA con **gensim** sobre los dos corpus y compara. La métrica es la coherencia `c_v`, y hay un barrido de K de 2 a 15 en los dos.

**Se usa `LdaModel`, no `LdaMulticore`.** No es una preferencia estética. `LdaMulticore` reparte los documentos entre procesos y los incorpora según terminan, así que **con la misma semilla da resultados distintos** —medido: 0,5633 y 0,5404 en dos ejecuciones idénticas— y encima depende de cuántos núcleos tenga el ordenador. `LdaModel` dio 0,5890 las dos veces.

## Multilingüe contra solo inglés

| | A · multilingüe | B · solo inglés |
|---|---|---|
| Reseñas | 11.968 | 7.465 (62,4 %) |
| Idiomas | 48 | 1 |
| Coherencia `c_v` con K=6 | **0,6038** | **0,5107** |
| Coherencia `c_v` con K=11 | 0,6282 | 0,4531 |
| Mejor K del barrido | 12 (0,6747) | 2 (0,5675) |
| Tópicos de idioma con K=6 | 3 de 6 (32,3 % de las reseñas) | 0, por construcción |
| Tópicos de idioma con K=11 | 7 de 11 | 0, por construcción |

## Los resultados

**Filtrar por idioma no mejora la coherencia.** B queda por debajo de A en 12 de los 14 valores de K del barrido, y también por debajo de las 12 submuestras de A reducidas a su mismo tamaño —ese control existe para descartar que la caída sea solo por tener menos datos—. Así que no: filtrar no hace mejor al modelo según la métrica.

**Y aun así filtrar sirve.** Lo que se gana no es coherencia, es que el modelo deja de gastar la mitad de sus tópicos en separar idiomas. Con K=6 son 3 de 6; **con K=11 son 7 de 11**, o sea que subir K no resuelve el problema, lo agrava. Los tópicos que quedan en B hablan de cine.

**La lección de fondo es sobre la métrica.** `c_v` mide si las palabras de un tópico co-ocurren, y las palabras de un mismo idioma co-ocurren siempre. Un tópico que agrupa todo el español es coherentísimo y no dice nada de ninguna película.

> **Un tópico puede ser perfectamente coherente y perfectamente inútil.**

Esa es la conclusión, y no «filtrar inglés mejora el modelo», que es lo contrario de lo que miden los números.

## Reproducibilidad

- Semillas fijas en el muestreo, en `langdetect` y en LDA.
- `LdaModel` en lugar de `LdaMulticore`, por lo explicado arriba.
- Versiones exactas en `requirements.txt`.
- Los `.parquet` no se versionan, pero el notebook 1 los reconstruye desde la fuente pública.
- **Tiempos medidos** en un portátil sin GPU: unos 5 minutos el notebook 1 y unos 7 el 2.

## Limitaciones

**El análisis va sobre una muestra de 12.000 reseñas** (11.968 tras normalizar) de las ~112.000 de 2024, con semilla fija, para que el notebook entero se ejecute en minutos. Para el corpus completo basta poner `N_MUESTRA = None` en el notebook 1; no lo hemos ejecutado así, de modo que las cifras de este README son las de la muestra.

**Hay una discrepancia sin explicar en el tamaño del corpus.** El trabajo entregado decía 38.531 reseñas de 2024 y hoy el mismo filtro sobre la misma fuente da 119.096. Está documentada en el notebook 1 y **no la hemos resuelto**: puede ser que el dataset se haya ampliado desde entonces, pero no lo hemos comprobado y no vamos a afirmarlo.

**La coherencia de 0,678 que figura en la presentación entregada no la puede reproducir nadie**, porque salió de `LdaMulticore`. Es una cifra histórica, no un resultado del repositorio. Las cifras reproducibles son las de la tabla de arriba.

**«2024» es el año de estreno de la película**, no la fecha en que se escribió la reseña.

**`c_v` no es una medida de utilidad.** Todo el trabajo apunta a eso, pero conviene decirlo también como limitación: no tenemos una métrica que puntúe lo que de verdad nos importa.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Después, **en este orden**:

1. `01_preprocesamiento.ipynb` — descarga el dataset y genera los `.parquet`.
2. `02_topic_modeling.ipynb` — LDA sobre los dos corpus y la comparación.

El segundo **no funciona sin el primero**: los `.parquet` no están en el repositorio y hay que generarlos.

```
.
├── README.md
├── LICENSE
├── NOTICE
├── requirements.txt
├── .gitignore
├── 01_preprocesamiento.ipynb          <- descarga, limpieza, PLN, idioma
├── 02_topic_modeling.ipynb            <- LDA, experimento A/B, controles
├── Letterboxd_Presentacion_FINAL.pdf
└── *.parquet                          <- los genera el notebook 1
```

## Presentación

**[Letterboxd_Presentacion_FINAL.pdf](Letterboxd_Presentacion_FINAL.pdf)** — 26 diapositivas.

El `.pptx` editable se conserva en local y no forma parte del repositorio.

## Licencia

El **código y la documentación** se publican bajo [licencia MIT](LICENSE).

Los **datos no se redistribuyen**, así que la licencia del repositorio no se pronuncia sobre ellos. El detalle está en [NOTICE](NOTICE) y en la sección [Los datos](#los-datos).

## Autoría

Trabajo de equipo de la asignatura Datos No Estructurados:

- **Alfredo Martín López**
- **Marcos Fernández Arévalo**
- **Pedro Pérez Blanco**

El experimento A/B multilingüe frente a solo inglés, la corrección de la reproducibilidad y esta versión del repositorio son posteriores a la entrega. El trabajo original es de los tres.
