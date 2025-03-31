# Modelado-Matemtico-para-Optimizacin-de Procesos
# OptimizaCNN: Sistema de Optimización de Procesos Industriales con Redes Neuronales Convolucionales

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🚀 Introducción

OptimizaCNN es un sistema avanzado para la optimización de procesos industriales mediante modelado matemático y aprendizaje profundo. Utilizando redes neuronales convolucionales (CNN), el sistema aprende las complejas relaciones no lineales entre múltiples variables de un proceso y permite identificar los parámetros óptimos que maximizan el rendimiento, minimizando al mismo tiempo los costos operativos.

<p align="center">
  <img src="docs/images/system_overview.png" alt="Vista general del sistema" width="700">
</p>

## 🔑 Características Principales

- **Modelado predictivo con CNN**: Aprovecha el poder de las redes neuronales convolucionales para capturar relaciones complejas y no lineales entre variables de proceso.
- **Optimización avanzada**: Implementa algoritmos de optimización matemática como SLSQP, L-BFGS-B y Nelder-Mead para encontrar los parámetros óptimos.
- **Análisis de sensibilidad**: Evalúa el impacto de cada variable en el rendimiento del proceso a través de análisis de sensibilidad detallados.
- **Interfaz gráfica intuitiva**: Interactúa con el sistema a través de una interfaz gráfica completa, sin necesidad de programación.
- **Soporte para GPU**: Acelera el entrenamiento de modelos utilizando GPU cuando está disponible.
- **Visualización de datos**: Presenta gráficos detallados del proceso de entrenamiento y resultados de optimización.

## 💼 Aplicaciones

Este sistema es ideal para:

- **Procesos químicos**: Optimización de reacciones químicas, mezclado y composición.
- **Manufactura**: Mejora de parámetros de producción para aumentar rendimiento y calidad.
- **Energía**: Optimización de sistemas de generación y distribución de energía.
- **Biotecnología**: Refinamiento de procesos de fermentación y biorreactores.
- **Investigación académica**: Herramienta de enseñanza y exploración de modelado matemático.
- **Análisis de datos industriales**: Exploración y modelado predictivo de datos de planta.

## 🛠️ Tecnologías Utilizadas

- **[Python](https://www.python.org/)**: Lenguaje base de programación
- **[PyTorch](https://pytorch.org/)**: Framework de aprendizaje profundo para modelado con CNN
- **[SciPy](https://scipy.org/)**: Algoritmos de optimización matemática
- **[NumPy](https://numpy.org/)**: Manipulación eficiente de arrays y operaciones matemáticas
- **[Pandas](https://pandas.pydata.org/)**: Análisis y manipulación de datos tabulares
- **[Matplotlib](https://matplotlib.org/)** y **[Seaborn](https://seaborn.pydata.org/)**: Visualización de datos
- **[Tkinter](https://docs.python.org/3/library/tkinter.html)**: Interfaz gráfica de usuario
- **[OpenCV (cv2)](https://opencv.org/)**: Procesamiento de imágenes (utilizado para transformaciones de datos)

## 📊 Ejemplo de Aplicación

El sistema incluye un conjunto de datos de muestra (`proceso_industrial_eafit.csv`) que simula un proceso químico industrial con las siguientes variables:

- **Variables de entrada**: temperatura, presión, concentración, tiempo de residencia, velocidad de agitación y flujo de entrada
- **Variables de salida**: eficiencia, consumo energético y costo por lote

La optimización de este proceso permite identificar los parámetros que maximizan la eficiencia mientras se minimiza el consumo energético y los costos.

## 📂 Estructura del Proyecto

```
OptimizaCNN/
├── optimizacion_cnn.py       # Implementación principal del sistema
├── requirements.txt          # Dependencias del proyecto
├── data/
│   └── proceso_industrial_eafit.csv  # Conjunto de datos de muestra
├── docs/
│   ├── manual_usuario.md     # Manual de usuario detallado
│   └── images/               # Imágenes para documentación
└── examples/
    └── ejemplo_basico.py     # Ejemplo de uso programático
```

## 🔧 Requisitos e Instalación

### Requisitos del Sistema

- Python 3.8 o superior
- Pip (gestor de paquetes de Python)
- GPU compatible con CUDA (opcional, para aceleración de entrenamiento)

### Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/OptimizaCNN.git
   cd OptimizaCNN
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta la aplicación:
   ```bash
   python optimizacion_cnn.py
   ```

## 🎮 Uso Básico

1. **Carga de datos**: Importa tus datos desde archivos CSV o Excel.
2. **Selección de variables**: Elige las variables de entrada y salida para el modelado.
3. **Construcción del modelo**: Configura y entrena la red neuronal CNN.
4. **Optimización**: Define los límites para cada parámetro y ejecuta la optimización.
5. **Análisis**: Visualiza los resultados y realiza análisis de sensibilidad.

Consulta el [manual de usuario](docs/manual_usuario.md) para instrucciones detalladas.

## 🔬 Fundamentos Teóricos

El sistema implementa un enfoque híbrido de modelado y optimización:

1. **Modelado con CNN**: Las redes neuronales convolucionales transforman datos tabulares en una estructura matricial, permitiendo capturar interacciones complejas entre variables.

2. **Normalización y preprocesamiento**: Los datos se normalizan para mejorar la convergencia y estabilidad del entrenamiento.

3. **Optimización matemática**: Se utilizan algoritmos de optimización no lineal para encontrar los parámetros que maximizan o minimizan una función objetivo.

4. **Análisis de sensibilidad**: Evaluación sistemática del impacto de cada variable en el resultado del proceso.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio
2. Crea una rama (`git checkout -b feature/amazing-feature`)
3. Realiza tus cambios y haz commit (`git commit -m 'Add amazing feature'`)
4. Sube tus cambios (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.

## 📚 Citación

Si utilizas este software en tu investigación, por favor cítalo como:

```
Autor, A. (2025). OptimizaCNN: Sistema de Optimización de Procesos Industriales con Redes Neuronales Convolucionales. GitHub. https://github.com/tu-usuario/OptimizaCNN
```

---

<p align="center">
  <i>Desarrollado con 💙 por el equipo de Modelado Matemático EAFIT</i>
</p>
