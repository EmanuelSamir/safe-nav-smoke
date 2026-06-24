# Safe Navigation in Smoke Environments via Multi-Agent Drones

Este repositorio contiene la infraestructura y algoritmos para un proyecto de investigación enfocado en la **navegación segura de múltiples drones (2D) en entornos simulados de humo**. 

El objetivo principal es permitir que múltiples agentes naveguen de forma segura (evitando colisiones entre ellos y minimizando el riesgo al atravesar zonas peligrosas) utilizando predicción del comportamiento del entorno y algoritmos de control predictivo.

## 🚀 Características Principales

1. **Smoke Forecasting (Predicción de Humo):**
   - Entrenamos modelos para predecir la propagación del humo en el tiempo.
   - **Propuesta Principal:** Fourier Neural Operator (FNO).
   - **Baseline:** ConvLSTM.
   
2. **Navegación y Control Predictivo:**
   - Utilizamos múltiples variantes de **MPPI** (Model Predictive Path Integral).
   - Implementamos un controlador **Dual Guard**, que hereda de MPPI pero incorpora una función de seguridad (Safety Shield / Control Barrier Functions) durante el rollout para garantizar seguridad teórica.

3. **Simulación Controlada (Playback):**
   - Dado que la simulación de fluidos de humo es muy costosa computacionalmente, el entorno usa un sistema de **Playback**. 
   - El Playback lee simulaciones previamente guardadas para garantizar comparaciones 1 a 1 justas entre diferentes controladores y modelos, todo bajo las mismas condiciones.

---

## 📂 Arquitectura Actual

El código base se divide principalmente en los módulos core (`src/`) y los scripts de ejecución o experimentos.

* `src/env/`: Entornos de simulación, incluyendo la lógica de `playback` para cargar episodios pre-calculados de humo.
* `src/models/` y `src/training/`: Arquitecturas de redes neuronales (FNO, ConvLSTM) en PyTorch Lightning y sus lógicas de entrenamiento.
* `src/controllers/`: Controladores de navegación (MPPI, Dual Guard, CBF, HJ).
* `src/wrappers/`: Adaptadores para acoplar las predicciones de humo al entorno.
* `scripts/`: Scripts bash/Slurm para ejecución en HPC (High Performance Computing).

## 🛠️ Flujo de Trabajo Actual (Legacy Pipeline)

Actualmente, el flujo de recolección de datos funcional consta de dos scripts en la raíz de `src/`:

1. `src/0_data_playback_physics_collection.py`: Recolecta la física/dinámica base.
2. `src/1_data_playback_env_collection.py`: Genera los episodios del entorno.

*(Nota: Los scripts de la versión 2 en adelante están deprecados y serán reemplazados por la nueva arquitectura descrita a continuación).*

---

## 🏗️ Filosofía de Configuración y Futura Estructura (Próximamente)

El proyecto se encuentra en plena migración para abandonar Hydra en favor de **Pydantic Estricto**. La nueva arquitectura busca resolver dolores de cabeza con configuraciones ocultas, promoviendo las siguientes reglas de diseño:

* **Separación de Responsabilidades:** El código core (`src/`) debe ser completamente agnóstico a la ejecución. 
* **Cero Valores por Defecto (No Defaults):** Los archivos YAML serán explícitos y la única fuente de verdad.
* **Cero Parámetros Fantasma:** Uso estricto de `model_config = ConfigDict(extra="forbid")` para que cualquier parámetro obsoleto en el YAML detenga la ejecución.
* **Manejo Explícito de Directorios:** En lugar de directorios dinámicos inyectados mágicamente, el orquestador creará su propia carpeta con un timestamp y guardará una copia del YAML para reproducibilidad.

### Nueva Estructura de Proyectos (`projects/`)

Para aplicar esta filosofía, los experimentos y el pipeline dejarán de estar directamente en `src/` y se moverán a una nueva carpeta `projects/`. Cada experimento será autocontenido y agrupará su propia lógica de orquestación, sus esquemas compuestos (Pydantic) y sus archivos YAML:

```text
├── src/                             # CÓDIGO CORE (agnóstico a la ejecución)
│
└── projects/                        # EXPERIMENTOS ACUMULATIVOS Y PIPELINE
    │
    ├── 01_data_collection/          # PROYECTO 1: Recolectar datos
    │   ├── schema.py                # Esquema compuesto SOLO para colectar
    │   ├── run.py                   # El script ejecutable de colecta
    │   ├── config_omni_fast.yaml    # Config para el robot Omni
    │   └── config_quad_slow.yaml    # Config para el Quadruped
    │
    └── 02_sac_benchmark/            # PROYECTO 2: Comparativa de controladores
        ├── schema.py                # Esquema compuesto SOLO para el benchmark
        ├── run.py                   # El script que itera los controladores
        └── benchmark_v1.yaml        # Config con la lista de controladores
```

Con esta estructura, es mucho más sencillo iterar experimentos aislados (ej: testear modelos, comparar controladores o realizar una integración total) sin romper pipelines pasados o arrastrar configuraciones heredadas.
