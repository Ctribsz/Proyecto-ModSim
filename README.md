# 🚨 Simulación de Evacuación - Proyecto de Modelación y Simulación

## 📌 Visión General
Este proyecto implementa un **modelo basado en agentes (MBA)** para simular la evacuación de un centro comercial en emergencia. El objetivo principal es **minimizar el tiempo total de evacuación** y **identificar cuellos de botella** en diferentes escenarios de emergencia.

**¿Para qué sirve?**  
- Ayuda a diseñar espacios públicos más seguros
- Evalúa estrategias de evacuación ante emergencias
- Proporciona información crítica para la toma de decisiones en seguridad pública
- Analiza cómo factores como el bloqueo de salidas o la presencia de personas con movilidad reducida afectan el proceso de evacuación

## 📂 Estructura del Proyecto

### 🌐 `app.py` - Interfaz de Usuario
- **Propósito**: Frontend interactivo basado en Streamlit
- **Funcionalidades**:
  - Permite ejecutar tres escenarios de simulación: *Baseline*, *Bloqueo* y *Anchos*
  - Visualiza curvas de evacuación en tiempo real
  - Genera métricas clave y permite descargar resultados
  - Muestra tiempos individuales de evacuación (como en la imagen que compartiste)

### 🧠 `src/` - Componentes Nucleares

#### `agents.py` - Lógica de Agentes
- **PersonAgent**:
  - Define el comportamiento de cada persona (niños, adultos, adultos mayores, personas con discapacidad)
  - Implementa decisiones realistas: elección de salida, velocidad variable, pánico, familiaridad con el lugar
  - Maneja estados: `CALMO` → `EVACUANDO` → `EVACUADO`
- **ExitAgent**:
  - Simula salidas con capacidad realista (personas/segundo)
  - Gestiona colas y throughput de evacuación

#### `model.py` - Modelo de Simulación
- **EvacuationModel**:
  - Inicializa el entorno (mapa, salidas, población heterogénea)
  - Genera campo de distancias (BFS) hacia salidas
  - Controla el flujo principal de la simulación
  - Almacena datos demográficos de la población

#### `scenarios.py` - Escenarios Experimentales
- **Baseline**: Escenario estándar (todas las salidas abiertas)
- **Bloqueo**: Simula bloqueo de una salida en un tiempo específico
- **Anchos**: Analiza cómo el ancho de las salidas afecta el tiempo de evacuación

#### `metrics.py` - Análisis y Visualización
- **run_model()**: Ejecuta simulaciones y recopila métricas
- **save_times()** y **save_metrics()**: Almacena resultados
- **plot_curva()** y **plot_curvas_comparadas()**: Genera visualizaciones profesionales

#### `space.py` - Geometría y Navegación
- **bfs_distance_field()**: Calcula distancia óptima a salidas
- **neighbors_moore()**: Define vecindad de Moore para movimiento

### 🧪 `experiments/` - Ejecución por Línea de Comandos
- Scripts para ejecutar escenarios desde terminal
- Generan resultados en carpetas `results/`
- Útiles para análisis profundos y corridas masivas

## 📊 Interpretando los Resultados

### Tiempos Individuales (como en tu captura)
- **`id`**: Identificador único de cada agente (persona)
- **`t_exit`**: Tiempo en segundos cuando la persona logra evacuar
- **Interpretación**: 
  - Permite identificar quiénes tardan más (personas con movilidad reducida, niños)
  - Muestra la distribución de tiempos de evacuación
  - Es clave para calcular percentiles (P50, P90) y tiempo total

### Métricas Clave
| Métrica | Significado | Importancia |
|---------|-------------|-------------|
| **makespan** | Tiempo total de evacuación | Tiempo máximo para que todos evacúen |
| **p50** | Tiempo en que el 50% ha evacuado | Indica eficiencia media |
| **p90** | Tiempo en que el 90% ha evacuado | Mide si hay grupos vulnerables |
| **evacuados** | Número total de personas que evacuaron | Verifica si hubo víctimas |
| **reelecciones_promedio** | Decisiones de cambio de salida por persona | Indica confusión durante evacuación |
| **throughput_exit_X** | Personas por segundo por salida | Identifica cuellos de botella |

## 🚀 Cómo Ejecutar el Proyecto

1. **Requisitos**:
   ```bash
   pip install mesa streamlit numpy pandas matplotlib
   ```

2. **Ejecutar la interfaz**:
   ```bash
   streamlit run app.py
   ```

3. **Ejecutar experimentos desde terminal**:
   ```bash
   python experiments/run_baseline.py --agents 300
   python experiments/run_bloqueo.py --t_bloqueo 60 --exit_index 1
   ```