# Tesis CIENTIBECA — Clasificación de Habla Imaginada con EEG

Repositorio oficial del proyecto de investigación enfocado en la **clasificación de habla imaginada a partir de señales EEG**, utilizando modelos de aprendizaje automático y redes neuronales profundas.

Este proyecto forma parte de una beca de investigación (CIENTIBECA) y tiene como objetivo analizar la separabilidad de clases, la relevancia de bandas frecuenciales y la capacidad de distintos modelos para decodificar señales cerebrales asociadas al habla imaginada.

---

## Objetivos del Proyecto

- Procesamiento y normalización de señales EEG
- Evaluación de modelos clásicos (MLP)
- Implementación y análisis de EEGNet
- Estudio del impacto de ventanas temporales
- Evaluación de estrategias de data augmentation
- Análisis de importancia de bandas frecuenciales
- Generación de métricas y visualizaciones reproducibles

---

## Estructura del Repositorio
tesis_cientibeca/
│
├── src/ # Código fuente reproducible
│ ├── preprocessing/ # Limpieza y transformación de señales
│ ├── models/ # Definición de modelos (MLP, CNN, EEGNet)
│ ├── training/ # Scripts de entrenamiento
│ ├── evaluation/ # Métricas y validación
│ └── utils/ # Funciones auxiliares
│
├── experiments/ # Resultados finales y visualizaciones
│ ├── <nombre_experimento>/
│ │ ├── temporal_window_analysis/
│ │ └── visualization_results/
│ └── ...
│
├── notebooks/ # Exploración y análisis preliminar
│
├── configs/ # Archivos de configuración de experimentos
│
├── config.yml # Configuración general del proyecto
├── environment.yml # Entorno reproducible (Conda)
├── .gitignore
└── README.md


---

## Política de Datos y Resultados

Este repositorio:

### Incluye
- Código fuente reproducible
- Configuraciones de experimentos
- Resultados agregados (gráficos y métricas finales)
- Visualizaciones comparativas

### No incluye
- Datos crudos (`.mat`)
- Checkpoints de modelos (`.pt`, `.ckpt`)
- Runs individuales de entrenamiento
- Logs completos

Los datos deben almacenarse localmente y configurarse en `config.yml`.

---

## Instalación del Entorno

Clonar el repositorio:

```powershell
git clone https://github.com/enzocatorano/tesis_habla_imaginada.git
cd tesis_habla_imaginada

# crear un entorno Conda
conda env create -f environment.yml
conda activate eeg-speech
```

---

## Comentarios adicionales
Las formas en las que se fueron realizando los experimentos y pruebas fue cambiando sustancialmente a lo largo de los meses. Entre los
diferentes puntos se encuentran:
- Cambios del modelo. Desde MLPs, EEGNet, Deep/Shallow ConvNet, etc.
- Cambios en preprocesamiento. Desde el uso crudo de señales hasta el uso de ICA para la remocion de artefactos de parpadeo. Incluso el
uso de caracteristicas estadisticas de la señal.
- Data augmentation: Inicialmente no se estaba trabajando con ello, luego se comenzo a incorporar variando las técnicas de augmentacion
utilizadas, asi como la cantidad de augmentaciones dadas por cada trial original. Se utilizaron tecnicas de ventaneo, inyeccion de ruido
en bandas especificas y FTSurrogate.

Los resultados obtenidos hasta el momento llevan a concluir que se requeriran enfoques que puedan suplir el pequeño tamaño del dataset,
sea mediante el uso de modelos pre-entrenados en señales de EEG, o explorando tecnicas de aprendizaje auto-supervisado, incluso con el
mismo set de datos.

## Sobre .\experiments\EEGNet_full_baseline
Ese experimento expone un resultado significativamente por encima de la probabilidad base a la hora de predecir el comando/vocal de habla
imaginada. Sin embargo, revisiones posteriores llevaron a encontrar dataleakeage en el procedimiento del mismo. Aun asi, se decidio
guardarlo con fines de poner en evidencia el proceso de aprendizaje y mejora en una futura instancia de defensa de tesis.
