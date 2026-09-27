## Visión General del Pipeline

Tres etapas: limpiar y dividir los datos, definir una política de aumentación, y luego ajustar (fine-tune)
y seleccionar un detector sobre un backbone preentrenado.

```mermaid
flowchart LR
    raw[("Dataset crudo<br/>Export de Roboflow, 6 clases")]
    dp["Procesamiento de Datos<br/>limpieza de etiquetas, nueva división 70/20/10"]
    processed[("Dataset procesado<br/>train / val / test")]
    aug["Política de Aumentación<br/>configuración basada en evidencia"]
    policy[("Política de aumentación")]
    tr["Entrenamiento<br/>ajuste fino, evaluación,<br/>comparación, selección"]
    model[("Modelo final")]

    raw --> dp
    dp --> processed
    processed -- "imágenes de train" --> aug
    aug --> policy
    processed -- "train + val" --> tr
    processed -. "test, reservado" .-> tr
    policy -- "online, solo train" --> tr
    tr --> model

    classDef stage fill:#14141c,stroke:#6e6e7a,color:#f2f2f5;
    classDef artifact fill:#101015,stroke:#8a8a96,color:#f2f2f5,stroke-dasharray:4 3;
    class dp,aug,tr stage
    class raw,processed,policy,model artifact
```

**Estado:** Procesamiento de Datos terminado. Política de aumentación definida, aún no evaluada en un
entrenamiento. Entrenamiento no iniciado.

### 1. Procesamiento de Datos

- Se limpió el export crudo antes que nada: se descartaron imágenes corruptas y se corrigieron o
  eliminaron anotaciones incorrectas, ya que un export de Roboflow puede traer errores de etiquetado que
  de otro modo perjudicarían el entrenamiento de forma silenciosa.
- Se re-dividió el dataset por nuestra cuenta (70/20/10) en lugar de mantener el split de Roboflow: el de
  ellos resultó desparejo entre clases, lo que habría hecho inútil la evaluación por clase.
- El conjunto de test queda reservado y no se toca hasta la evaluación final.

### 2. Política de Aumentación

- Se eligieron las técnicas de aumentación midiendo primero los datos procesados (brillo, desenfoque,
  color, balance de clases).
- Hallazgos principales: al dataset le faltan imágenes oscuras o borrosas como las que tendría un video
  real de la cinta transportadora; algunas clases se distinguen principalmente por color; el
  "desbalance" de clases se debe a cuántos objetos aparecen por foto, no a cuántas fotos hay por clase.
- Se aplica **online** durante el entrenamiento (una configuración compartida, no una copia reescrita del
  dataset): esto mantiene a todo el equipo sincronizado y garantiza que el conjunto de test no pueda
  aumentarse por accidente.
- Se definió una pequeña ablación (con vs. sin nuestros cambios) para medir el beneficio.

### 3. Entrenamiento — no iniciado

- Fine-tuning de YOLO preentrenado (YOLOv8n, YOLO11n) en lugar de entrenar desde cero: necesita mucho
  menos datos y más eficiente.
- Evaluación con mAP, curva PR y matriz de confusión: la matriz de confusión es la más relevante para un
  caso de uso de clasificación de residuos, ya que muestra qué materiales se confunden entre sí.
- Comparación de ambas variantes de YOLO en precisión y velocidad de inferencia, y selección del mejor
  compromiso.
