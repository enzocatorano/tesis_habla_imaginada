# Tesis — CIENTIBECA: Habla Imaginada (EEG)

**Resumen breve**  
Proyecto para estudiar señales EEG durante habla imaginada. El objetivo es procesar los datos crudos (archivos `.mat`) y preparar conjuntos reproducibles para análisis y modelado (features / modelos ML). Este repositorio contiene scripts de preprocesamiento, notebooks de exploración y la configuración básica para reproducir el entorno de desarrollo.

---

## Contenido principal del repositorio
- `src/` : scripts reproducibles (preprocesamiento, conversión `.mat` → `.npz`, utilidades).  
- `notebooks/` : notebooks de exploración y visualización (EDA).  
- `data/` : (vacío en el repo) `data/processed/` es la carpeta donde se guardan los `.npz` procesados localmente — **NO** se suben los datos al repo.  
- `config.yml` : archivo de configuración (rutas, `fs`, `processed_root`), leído por los scripts.  
- `environment.yml` (o `environment_clean.yml`) : archivo para crear el entorno Conda reproducible.  
- `README.md` : este archivo.

---

## Antes de empezar — política de datos
Los archivos `.mat` originales **no** están incluidos en este repo.  
- La base original (por ejemplo `Base_de_Datos_Habla_Imaginada/`) debe estar en tu máquina local fuera del repo.  
- Añadí en `config.yml` la ruta relativa o absoluta a esa carpeta para que los scripts encuentren los datos.
- No subas datos al repositorio a menos que sean pequeños y tengas permiso para compartirlos.

---

## Quickstart (Windows / PowerShell)

1. **Clonar el repo** (si aún no lo hiciste):
```powershell
git clone https://github.com/enzocatorano/tesis_habla_imaginada.git
