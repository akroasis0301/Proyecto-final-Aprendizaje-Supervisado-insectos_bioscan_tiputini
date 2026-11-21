# Clasificación del orden de insectos de trampas Malaise con transfer learning

**Proyecto final · Aprendizaje Supervisado · Maestría en Inteligencia Artificial, Universidad San Francisco de Quito (USFQ)**
Autor: Miguel Álvarez · noviembre de 2026

> Una base para el procesamiento asistido de muestras de biodiversidad neotropical (Estación de Biodiversidad Tiputini).

**Notebook en Google Colab (ejecutado):** https://colab.research.google.com/drive/1imKk9roH_bQaWWMEBgDfQpOYXfB3sQic?usp=sharing

## Problema
Separar por **orden** taxonómico los insectos capturados en trampas Malaise es el primer paso, manual y lento, de un inventario de biodiversidad. Este proyecto evalúa si una red preentrenada, sin reentrenar, más un clasificador clásico permiten automatizar ese paso, y compara con rigor estadístico un clasificador lineal y uno no lineal.

| | |
|---|---|
| **T (tarea)** | Predecir el orden de un insecto a partir de su foto de microscopio |
| **E (experiencia)** | Muestra de BIOSCAN-1M etiquetada por expertos (con respaldo de ADN) |
| **P (desempeño)** | Macro-F1 (el desbalance hace engañosa la accuracy) |

## Datos
- **BIOSCAN-1M** (Gharaee *et al.*, NeurIPS 2023), licencia CC-BY-3.0: 1.13 millones de insectos de trampas Malaise (Costa Rica, Canadá y Sudáfrica), fotografiados con microscopio.
- Se usan el paquete `cropped_256.hdf5` (7.41 GB) y `BIOSCAN_Insect_Dataset_metadata.tsv` (1.23 GB) de https://huggingface.co/datasets/bioscan-ml/BIOSCAN-1M
- **Muestra:** órdenes con al menos 500 imágenes y hasta 1 000 por orden, al azar con semilla 42. Partición 80 % desarrollo / 20 % test, estratificada.
- **Los datos no están en el repositorio.** El notebook los descarga solo si no los encuentra en `BIOSCAN_DIR` (por defecto `~/datasets/BIOSCAN-1M`).

## Método
1. **EDA** de los 1.1 M de metadatos y de la muestra.
2. **Feature extraction:** ResNet-50 preentrenada en ImageNet, **congelada** → 2 048 features por imagen.
3. **t-SNE** de los embeddings, coloreado por orden, más la pureza de vecinos.
4. **Pipelines** `StandardScaler → SelectKBest(f_classif) → clasificador`, con **optimización de hiperparámetros** por `GridSearchCV` + `RepeatedStratifiedKFold` (5×2): regresión logística (C, k) y SVM RBF (C, gamma). Curva de aprendizaje.
5. **Comparación estadística** regresión logística frente a SVM RBF con **repeated stratified k-fold 10×10** (mismos folds): Wilcoxon, t-test pareado y t-test corregido de Nadeau y Bengio.
6. **Evaluación final** en el test reservado (usado una sola vez): macro-F1, matriz de confusión y McNemar.

## Cómo ejecutar
### Opción recomendada: uv
```bash
# Instalar uv (una sola vez): brew install uv   o   curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                                    # crea el entorno con Python 3.12 y las versiones fijadas en uv.lock
uv run python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
export BIOSCAN_DIR=~/datasets/BIOSCAN-1M   # opcional: dónde guardar los datos (~8.6 GB)
uv run jupyter notebook insectos_bioscan_tiputini.ipynb      # Kernel → Restart & Run All
```
Para usar el entorno desde VS Code, registra el kernel: `uv run python -m ipykernel install --user --name bioscan --display-name "Python (bioscan)"`.

### Alternativa: pip
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook insectos_bioscan_tiputini.ipynb
```
- El notebook detecta el dispositivo: CUDA (Colab con GPU), MPS (Mac con Apple Silicon) o CPU.
- La primera ejecución descarga los datos y extrae los embeddings, que quedan guardados en `BIOSCAN_DIR` para no repetirlo.

## Registro de la ejecución y reutilización
- **Muestra guardada:** la primera ejecución copia las 10 500 imágenes de la muestra a `muestra_N1000_min500_seed42.h5` (~72 MB) junto con su CSV y un resumen de la población (`eda_poblacion_bioscan1m.json`). Con esos tres archivos, el notebook corre en **modo ligero**, sin los 8.6 GB (útil en Colab).
- **Embeddings:** `embeddings_resnet50_10500_seed42.npz`; la ResNet-50 no se vuelve a ejecutar.
- **Checkpoints** (`BIOSCAN_DIR/registro/v1/*.joblib`): búsquedas de hiperparámetros, t-SNE, curva de aprendizaje, validación cruzada 10×10 y los dos modelos finales entrenados. Cada nombre lleva una huella de su configuración: si cambia un parámetro, se recalcula. `RECALCULAR=1` fuerza todo.
- **Bitácora y resultados:** `registro.log` (con fecha y hora), `metricas.json`, `entorno.json`, `cv_results_*.csv`, `comparacion_10x10_por_fold.csv`, `curva_aprendizaje.csv`, `predicciones_test.csv`, `composicion_familias_muestra.csv`, `acierto_test_por_familia.csv` y `particion.npz`.

## Resultados
Muestra de 10 500 imágenes (11 órdenes); desarrollo 8 400, test 2 100.

| | Regresión logística | SVM RBF |
|---|---|---|
| Hiperparámetros (CV 5×2) | C = 0.01, k = 1 024 | C = 10, gamma = 0.1/k, k = 1 024 |
| macro-F1 · CV 10×10 | 0.8506 ± 0.0099 | 0.8548 ± 0.0102 |
| macro-F1 · test | 0.851 | 0.861 |
| segundos por entrenamiento | 1.75 | 26.5 |

- **Comparación 10×10** (100 folds pareados): diferencia media +0.0042 a favor del SVM. Wilcoxon y t-test ingenuo dan p < 10⁻⁷, pero el **t-test corregido de Nadeau y Bengio da p = 0.0997**: con α = 0.05 no hay evidencia de diferencia. McNemar en test: p = 0.085.
- **Recomendación:** regresión logística, por el criterio del modelo más simple dentro del empate (15 veces más rápida, 170 veces más liviana).
- **Errores:** se concentran en Hemiptera y Diptera (los órdenes más heterogéneos) y en la confusión Lepidoptera ↔ Trichoptera (órdenes hermanos).
- **Desbalance:** la población está muy concentrada (Pielou *J* = 0.28, Diptera 79 %); la muestra es casi equitativa entre órdenes (*J* = 0.997), pero no dentro de ellos. Las familias raras de cada orden se aciertan unos 7 puntos menos que las 3 dominantes (Fisher p < 0.001), aunque por familia la relación tamaño–acierto no es significativa (Spearman p = 0.29).
- **Overfitting:** las curvas de validación (pérdida de entrenamiento y de CV frente a C) tienen forma de U, y el C elegido está en el mínimo de la pérdida de validación en ambos modelos. Hay brecha entre entrenamiento y validación (0.13 la regresión logística y 0.12 el SVM), pero el test reproduce la CV.
- **Curvas de aprendizaje** (ambos modelos): alta varianza. Más datos por orden, o una representación específica del dominio, son el siguiente paso.

## Uso de IA generativa
Se usó Claude (Anthropic) como asistente. Los prompts están como comentarios al inicio de cada celda de código y en el anexo del notebook.

## Referencias
Las referencias, en formato IEEE, están en la sección 9 del notebook.
