import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.optimize import minimize
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import pickle
import os
import sys

class OptimizacionProcesosCNN:
    def __init__(self):
        """
        Inicializa el sistema de optimización de procesos con CNN
        """
        self.model = None
        self.data = None
        self.proceso_optimizado = None
        self.parametros_optimos = None
        self.rendimiento_optimo = None
        
    def cargar_datos(self, archivo):
        """
        Carga los datos desde un archivo CSV o Excel
        
        Args:
            archivo (str): Ruta al archivo de datos
        
        Returns:
            bool: True si la carga fue exitosa, False en caso contrario
        """
        try:
            if archivo.endswith('.csv'):
                self.data = pd.read_csv(archivo)
            elif archivo.endswith('.xlsx') or archivo.endswith('.xls'):
                self.data = pd.read_excel(archivo)
            else:
                print("Formato de archivo no soportado. Use CSV o Excel.")
                return False
            
            print(f"Datos cargados exitosamente. Dimensiones: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return False
    
    def preprocesar_datos(self, variables_entrada, variable_salida, test_size=0.2):
        """
        Preprocesa los datos para entrenar la red neuronal
        
        Args:
            variables_entrada (list): Lista de nombres de columnas para las variables de entrada
            variable_salida (str): Nombre de la columna para la variable de salida
            test_size (float): Proporción de datos para validación
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        try:
            # Verificar que las columnas existan
            for col in variables_entrada + [variable_salida]:
                if col not in self.data.columns:
                    print(f"La columna {col} no existe en los datos")
                    return None
            
            # Extraer variables de entrada y salida
            X = self.data[variables_entrada].values
            y = self.data[variable_salida].values
            
            # Normalizar los datos
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            
            X_norm = (X - self.X_mean) / self.X_std
            y_norm = (y - self.y_mean) / self.y_std
            
            # Reformatear X para CNN si es necesario
            # Aquí asumimos que las variables de entrada se pueden organizar en una matriz 2D
            # Podemos ajustar esto según la aplicación específica
            n_samples = X_norm.shape[0]
            grid_size = int(np.ceil(np.sqrt(len(variables_entrada))))
            
            # Rellenar con ceros si es necesario
            pad_size = grid_size * grid_size - len(variables_entrada)
            X_padded = np.pad(X_norm, ((0, 0), (0, pad_size)), 'constant')
            
            # Reformatear a formato de imagen para CNN
            X_reshaped = X_padded.reshape(n_samples, grid_size, grid_size, 1)
            
            # Dividir en conjuntos de entrenamiento y prueba
            n_train = int(n_samples * (1 - test_size))
            X_train, X_test = X_reshaped[:n_train], X_reshaped[n_train:]
            y_train, y_test = y_norm[:n_train], y_norm[n_train:]
            
            # Guardar las variables originales para la optimización
            self.variables_entrada = variables_entrada
            self.variable_salida = variable_salida
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            print(f"Error al preprocesar los datos: {e}")
            return None
    
    def construir_modelo_cnn(self, input_shape, learning_rate=0.001):
        """
        Construye la arquitectura de la red neuronal CNN
        
        Args:
            input_shape (tuple): Dimensiones de los datos de entrada (height, width, channels)
            learning_rate (float): Tasa de aprendizaje para el optimizador
            
        Returns:
            keras.Model: Modelo de red neuronal CNN
        """
        try:
            # Definir arquitectura de la CNN
            inputs = Input(shape=input_shape)
            
            # Primera capa convolucional
            x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            
            # Segunda capa convolucional
            x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            
            # Aplanar y pasar a capas densas
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(64, activation='relu')(x)
            
            # Capa de salida (regresión)
            outputs = Dense(1, activation='linear')(x)
            
            # Crear modelo
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compilar modelo
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            self.model = model
            print("Modelo CNN construido:")
            model.summary()
            
            return model
        
        except Exception as e:
            print(f"Error al construir el modelo CNN: {e}")
            return None
    
    def entrenar_modelo(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, patience=10):
        """
        Entrena el modelo CNN con los datos proporcionados
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de validación
            epochs (int): Número máximo de épocas de entrenamiento
            batch_size (int): Tamaño del lote para entrenamiento
            patience (int): Número de épocas de paciencia para early stopping
            
        Returns:
            keras.History: Historial de entrenamiento
        """
        try:
            # Callback para detención temprana
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            
            # Entrenar modelo
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluar modelo
            evaluation = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"Evaluación del modelo - MSE: {evaluation[0]}, MAE: {evaluation[1]}")
            
            return history
        
        except Exception as e:
            print(f"Error al entrenar el modelo: {e}")
            return None
    
    def objetivo_optimizacion(self, x):
        """
        Función objetivo para optimización.
        Predice el rendimiento del proceso dados los parámetros de entrada.
        
        Args:
            x (array): Valores de los parámetros del proceso a optimizar
            
        Returns:
            float: Valor negativo del rendimiento (para minimizar)
        """
        try:
            # Normalizar los parámetros
            x_norm = (x - self.X_mean) / self.X_std
            
            # Reformatear para CNN
            grid_size = int(np.ceil(np.sqrt(len(self.variables_entrada))))
            pad_size = grid_size * grid_size - len(self.variables_entrada)
            x_padded = np.pad(x_norm, (0, pad_size), 'constant')
            x_reshaped = x_padded.reshape(1, grid_size, grid_size, 1)
            
            # Predecir rendimiento
            y_pred_norm = self.model.predict(x_reshaped, verbose=0)[0][0]
            
            # Desnormalizar
            y_pred = y_pred_norm * self.y_std + self.y_mean
            
            # Devolver el negativo para maximizar el rendimiento
            return -y_pred
        
        except Exception as e:
            print(f"Error en la función objetivo: {e}")
            return np.inf
    
    def optimizar_proceso(self, limites, metodo='SLSQP', max_iter=100):
        """
        Optimiza los parámetros del proceso utilizando el modelo entrenado
        
        Args:
            limites (list): Lista de tuplas (min, max) para cada parámetro
            metodo (str): Método de optimización
            max_iter (int): Número máximo de iteraciones
            
        Returns:
            dict: Resultados de la optimización
        """
        try:
            if self.model is None:
                print("Error: Primero debe entrenar un modelo")
                return None
            
            # Punto inicial (promedios de los datos)
            x0 = self.X_mean
            
            # Ejecutar optimización
            resultado = minimize(
                self.objetivo_optimizacion,
                x0,
                method=metodo,
                bounds=limites,
                options={'maxiter': max_iter, 'disp': True}
            )
            
            if resultado.success:
                # Parámetros optimizados
                parametros_optimos = resultado.x
                
                # Rendimiento predicho con estos parámetros
                rendimiento_optimo = -self.objetivo_optimizacion(parametros_optimos)
                
                # Guardar resultados
                self.parametros_optimos = parametros_optimos
                self.rendimiento_optimo = rendimiento_optimo
                
                # Crear diccionario de resultados
                self.proceso_optimizado = {
                    'parametros': {var: valor for var, valor in zip(self.variables_entrada, parametros_optimos)},
                    'rendimiento': rendimiento_optimo,
                    'convergencia': resultado.success,
                    'iteraciones': resultado.nit,
                    'mensaje': resultado.message
                }
                
                print("Optimización completada con éxito:")
                for var, valor in zip(self.variables_entrada, parametros_optimos):
                    print(f"  {var}: {valor:.4f}")
                print(f"Rendimiento predicho: {rendimiento_optimo:.4f}")
                
                return self.proceso_optimizado
            else:
                print(f"La optimización no convergió: {resultado.message}")
                return None
        
        except Exception as e:
            print(f"Error durante la optimización: {e}")
            return None
    
    def guardar_modelo(self, ruta_modelo, ruta_metadata=None):
        """
        Guarda el modelo entrenado y los metadatos asociados
        
        Args:
            ruta_modelo (str): Ruta para guardar el modelo
            ruta_metadata (str, optional): Ruta para guardar metadatos
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        try:
            # Guardar modelo
            self.model.save(ruta_modelo)
            
            # Guardar metadatos
            if ruta_metadata:
                metadata = {
                    'X_mean': self.X_mean,
                    'X_std': self.X_std,
                    'y_mean': self.y_mean,
                    'y_std': self.y_std,
                    'variables_entrada': self.variables_entrada,
                    'variable_salida': self.variable_salida,
                    'proceso_optimizado': self.proceso_optimizado
                }
                with open(ruta_metadata, 'wb') as f:
                    pickle.dump(metadata, f)
            
            print(f"Modelo guardado en {ruta_modelo}")
            return True
        
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            return False
    
    def cargar_modelo(self, ruta_modelo, ruta_metadata=None):
        """
        Carga un modelo previamente guardado y sus metadatos
        
        Args:
            ruta_modelo (str): Ruta del modelo guardado
            ruta_metadata (str, optional): Ruta de los metadatos
            
        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        try:
            # Cargar modelo
            self.model = tf.keras.models.load_model(ruta_modelo)
            
            # Cargar metadatos
            if ruta_metadata:
                with open(ruta_metadata, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.X_mean = metadata['X_mean']
                self.X_std = metadata['X_std']
                self.y_mean = metadata['y_mean']
                self.y_std = metadata['y_std']
                self.variables_entrada = metadata['variables_entrada']
                self.variable_salida = metadata['variable_salida']
                self.proceso_optimizado = metadata['proceso_optimizado']
            
            print(f"Modelo cargado desde {ruta_modelo}")
            return True
        
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False
    
    def analizar_sensibilidad(self, variable, valores, parametros_base=None):
        """
        Realiza un análisis de sensibilidad para una variable
        
        Args:
            variable (str): Nombre de la variable a analizar
            valores (array): Valores para evaluar la variable
            parametros_base (dict, optional): Valores base para las demás variables
            
        Returns:
            dict: Resultados del análisis de sensibilidad
        """
        try:
            if self.model is None:
                print("Error: Primero debe entrenar un modelo")
                return None
            
            # Si no se proporcionan parámetros base, usar los óptimos o promedios
            if parametros_base is None:
                if self.parametros_optimos is not None:
                    parametros_base = {var: valor for var, valor in 
                                      zip(self.variables_entrada, self.parametros_optimos)}
                else:
                    parametros_base = {var: valor for var, valor in 
                                      zip(self.variables_entrada, self.X_mean)}
            
            # Verificar que la variable esté en las variables de entrada
            if variable not in self.variables_entrada:
                print(f"Error: La variable {variable} no está en las variables de entrada")
                return None
            
            # Calcular rendimiento para cada valor
            rendimientos = []
            for valor in valores:
                # Crear vector de parámetros
                params = parametros_base.copy()
                params[variable] = valor
                
                # Convertir a array en el orden correcto
                x = np.array([params[var] for var in self.variables_entrada])
                
                # Predecir rendimiento
                rendimiento = -self.objetivo_optimizacion(x)
                rendimientos.append(rendimiento)
            
            resultados = {
                'variable': variable,
                'valores': valores,
                'rendimientos': rendimientos
            }
            
            return resultados
        
        except Exception as e:
            print(f"Error en el análisis de sensibilidad: {e}")
            return None

# Interfaz gráfica
class InterfazOptimizacion(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Sistema de Optimización de Procesos con CNN")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        self.optimizer = OptimizacionProcesosCNN()
        self.variables_entrada = []
        self.variable_salida = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica principal"""
        # Marco principal con pestañas
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña: Datos
        self.tab_datos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_datos, text="Datos")
        self.crear_tab_datos()
        
        # Pestaña: Modelo
        self.tab_modelo = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_modelo, text="Modelo CNN")
        self.crear_tab_modelo()
        
        # Pestaña: Optimización
        self.tab_optimizacion = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_optimizacion, text="Optimización")
        self.crear_tab_optimizacion()
        
        # Pestaña: Análisis
        self.tab_analisis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analisis, text="Análisis")
        self.crear_tab_analisis()
        
        # Barra de estado
        self.barra_estado = ttk.Label(self, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.barra_estado.pack(side=tk.BOTTOM, fill=tk.X)
    
    def crear_tab_datos(self):
        """Crea la pestaña de carga y visualización de datos"""
        # Marco para carga de datos
        frame_carga = ttk.LabelFrame(self.tab_datos, text="Carga de Datos")
        frame_carga.pack(fill=tk.X, padx=10, pady=10)
        
        # Botón para cargar datos
        ttk.Button(frame_carga, text="Cargar archivo CSV/Excel", 
                  command=self.cargar_archivo).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.lbl_archivo = ttk.Label(frame_carga, text="Ningún archivo seleccionado")
        self.lbl_archivo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        
        # Marco para selección de variables
        frame_variables = ttk.LabelFrame(self.tab_datos, text="Selección de Variables")
        frame_variables.pack(fill=tk.X, padx=10, pady=10)
        
        # Lista de variables disponibles
        frame_disponibles = ttk.Frame(frame_variables)
        frame_disponibles.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame_disponibles, text="Variables Disponibles:").pack(anchor=tk.W)
        self.lista_disponibles = tk.Listbox(frame_disponibles, selectmode=tk.EXTENDED, height=10)
        self.lista_disponibles.pack(fill=tk.BOTH, expand=True)
        scroll_disp = ttk.Scrollbar(self.lista_disponibles, orient=tk.VERTICAL, 
                                   command=self.lista_disponibles.yview)
        self.lista_disponibles.configure(yscrollcommand=scroll_disp.set)
        scroll_disp.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botones para mover variables
        frame_botones = ttk.Frame(frame_variables)
        frame_botones.pack(side=tk.LEFT, padx=10, pady=30)
        
        ttk.Button(frame_botones, text="→", 
                  command=lambda: self.mover_variables(self.lista_disponibles, self.lista_entrada)
                 ).pack(pady=5)
        ttk.Button(frame_botones, text="←", 
                  command=lambda: self.mover_variables(self.lista_entrada, self.lista_disponibles)
                 ).pack(pady=5)
        ttk.Button(frame_botones, text="→", 
                  command=lambda: self.seleccionar_salida()
                 ).pack(pady=20)
        ttk.Button(frame_botones, text="←", 
                  command=lambda: self.quitar_salida()
                 ).pack(pady=5)
        
        # Lista de variables de entrada
        frame_entrada = ttk.Frame(frame_variables)
        frame_entrada.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame_entrada, text="Variables de Entrada:").pack(anchor=tk.W)
        self.lista_entrada = tk.Listbox(frame_entrada, selectmode=tk.EXTENDED, height=10)
        self.lista_entrada.pack(fill=tk.BOTH, expand=True)
        scroll_ent = ttk.Scrollbar(self.lista_entrada, orient=tk.VERTICAL, 
                                  command=self.lista_entrada.yview)
        self.lista_entrada.configure(yscrollcommand=scroll_ent.set)
        scroll_ent.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Variable de salida
        frame_salida = ttk.Frame(frame_variables)
        frame_salida.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame_salida, text="Variable de Salida:").pack(anchor=tk.W)
        self.lista_salida = tk.Listbox(frame_salida, selectmode=tk.SINGLE, height=10)
        self.lista_salida.pack(fill=tk.BOTH, expand=True)
        
        # Marco para visualización de datos
        frame_vista = ttk.LabelFrame(self.tab_datos, text="Vista previa de datos")
        frame_vista.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tabla para mostrar los datos
        self.tabla_datos = ttk.Treeview(frame_vista)
        self.tabla_datos.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scroll_y = ttk.Scrollbar(frame_vista, orient=tk.VERTICAL, 
                                command=self.tabla_datos.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tabla_datos.configure(yscrollcommand=scroll_y.set)
        
        scroll_x = ttk.Scrollbar(frame_vista, orient=tk.HORIZONTAL, 
                                command=self.tabla_datos.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tabla_datos.configure(xscrollcommand=scroll_x.set)
        
        # Botones de acción
        frame_acciones = ttk.Frame(self.tab_datos)
        frame_acciones.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(frame_acciones, text="Preprocesar Datos", 
                  command=self.preprocesar_datos).pack(side=tk.RIGHT, padx=5)
    
    def crear_tab_modelo(self):
        """Crea la pestaña para configuración y entrenamiento del modelo CNN"""
        # Marco para parámetros del modelo
        frame_params = ttk.LabelFrame(self.tab_modelo, text="Parámetros del Modelo")
        frame_params.pack(fill=tk.X, padx=10, pady=10)
        
        # Grid para los parámetros
        row = 0
        ttk.Label(frame_params, text="Tasa de aprendizaje:").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_lr = ttk.Entry(frame_params)
        self.entry_lr.insert(0, "0.001")
        self.entry_lr.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        row += 1
        ttk.Label(frame_params, text="Épocas:").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_epochs = ttk.Entry(frame_params)
        self.entry_epochs.insert(0, "100")
        self.entry_epochs.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        row += 1
        ttk.Label(frame_params, text="Tamaño del lote:").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_batch_size = ttk.Entry(frame_params)
        self.entry_batch_size.insert(0, "32")
        self.entry_batch_size.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        row += 1
        ttk.Label(frame_params, text="Paciencia para early stopping:").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_patience = ttk.Entry(frame_params)
        self.entry_patience.insert(0, "10")
        self.entry_patience.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        row += 1
        ttk.Label(frame_params, text="Proporción de validación:").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_test_size = ttk.Entry(frame_params)
        self.entry_test_size.insert(0, "0.2")
        self.entry_test_size.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Botones de acción para el modelo
        frame_acciones = ttk.Frame(self.tab_modelo)
        frame_acciones.pack(fill=tk.X, padx=10, pady=10)
        
        self.btn_construir = ttk.Button(frame_acciones, text="Construir Modelo", 
                                       command=self.construir_modelo)
        self.btn_construir.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_entrenar = ttk.Button(frame_acciones, text="Entrenar Modelo", 
                                      command=self.entrenar_modelo, state=tk.DISABLED)
        self.btn_entrenar.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_guardar = ttk.Button(frame_acciones, text="Guardar Modelo", 
                                     command=self.guardar_modelo, state=tk.DISABLED)
        self.btn_guardar.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_cargar = ttk.Button(frame_acciones, text="Cargar Modelo", 
                                    command=self.cargar_modelo)
        self.btn_cargar.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Marco para visualización del entrenamiento
        frame_grafico = ttk.LabelFrame(self.tab_modelo, text="Progreso de Entrenamiento")
        frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas para el gráfico de entrenamiento
        self.fig_train, self.ax_train = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, frame_grafico)
        self.canvas_train.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Inicializar gráficos vacíos
        self.ax_train[0].set_title('Pérdida de Entrenamiento')
        self.ax_train[0].set_xlabel('Época')
        self.ax_train[0].set_ylabel('Pérdida')
        self.ax_train[1].set_title('Error Absoluto Medio')
        self.ax_train[1].set_xlabel('Época')
        self.ax_train[1].set_ylabel('MAE')
        self.fig_train.tight_layout()
        self.canvas_train.draw()
        
        # Etiqueta para información del modelo
        self.lbl_info_modelo = ttk.Label(self.tab_modelo, text="Modelo no construido", 
                                        background='lightgray', padding=10)
        self.lbl_info_modelo.pack(fill=tk.X, padx=10, pady=10)
    
    def crear_tab_optimizacion(self):
        """Crea la pestaña para optimización de procesos"""
        # Marco para parámetros de optimización
        frame_opt = ttk.LabelFrame(self.tab_optimizacion, text="Parámetros de Optimización")
        frame_opt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Marco para límites de variables
        self.frame_limites = ttk.Frame(frame_opt)
        self.frame_limites.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(self.frame_limites, text="Variable").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(self.frame_limites, text="Mínimo").grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.frame_limites, text="Máximo").grid(row=0, column=2, padx=5, pady=5)
        
        # Marco para método de optimización
        frame_metodo = ttk.Frame(frame_opt)
        frame_metodo.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(frame_metodo, text="Método de optimización:").pack(side=tk.LEFT, padx=5)
        self.combo_metodo = ttk.Combobox(frame_metodo, values=["SLSQP", "L-BFGS-B", "TNC", "Nelder-Mead"])
        self.combo_metodo.set("SLSQP")
        self.combo_metodo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame_metodo, text="Máximo de iteraciones:").pack(side=tk.LEFT, padx=5)
        self.entry_max_iter = ttk.Entry(frame_metodo, width=10)
        self.entry_max_iter.insert(0, "100")
        self.entry_max_iter.pack(side=tk.LEFT, padx=5)
        
        # Botones de acción
        frame_acciones = ttk.Frame(self.tab_optimizacion)
        frame_acciones.pack(fill=tk.X, padx=10, pady=10)
        
        self.btn_optimizar = ttk.Button(frame_acciones, text="Optimizar Proceso", 
                                       command=self.optimizar_proceso, state=tk.DISABLED)
        self.btn_optimizar.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Marco para resultados de optimización
        frame_resultados = ttk.LabelFrame(self.tab_optimizacion, text="Resultados de la Optimización")
        frame_resultados.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget para mostrar resultados
        self.txt_resultados = tk.Text(frame_resultados, height=10, width=80)
        self.txt_resultados.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scroll_res = ttk.Scrollbar(frame_resultados, orient=tk.VERTICAL, 
                                  command=self.txt_resultados.yview)
        scroll_res.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_resultados.configure(yscrollcommand=scroll_res.set)
        
        # Insertar mensaje inicial
        self.txt_resultados.insert(tk.END, "Optimice el proceso para ver los resultados aquí.")
        self.txt_resultados.config(state=tk.DISABLED)
    
    def crear_tab_analisis(self):
        """Crea la pestaña para análisis de sensibilidad y visualización"""
        # Marco para análisis de sensibilidad
        frame_sensibilidad = ttk.LabelFrame(self.tab_analisis, text="Análisis de Sensibilidad")
        frame_sensibilidad.pack(fill=tk.X, padx=10, pady=10)
        
        # Selección de variable para análisis
        frame_var = ttk.Frame(frame_sensibilidad)
        frame_var.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(frame_var, text="Variable a analizar:").pack(side=tk.LEFT, padx=5)
        self.combo_var_analisis = ttk.Combobox(frame_var)
        self.combo_var_analisis.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame_var, text="Valor mínimo:").pack(side=tk.LEFT, padx=5)
        self.entry_min_analisis = ttk.Entry(frame_var, width=10)
        self.entry_min_analisis.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame_var, text="Valor máximo:").pack(side=tk.LEFT, padx=5)
        self.entry_max_analisis = ttk.Entry(frame_var, width=10)
        self.entry_max_analisis.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame_var, text="Número de puntos:").pack(side=tk.LEFT, padx=5)
        self.entry_puntos_analisis = ttk.Entry(frame_var, width=10)
        self.entry_puntos_analisis.insert(0, "20")
        self.entry_puntos_analisis.pack(side=tk.LEFT, padx=5)
        
        # Botón para realizar análisis
        self.btn_analizar = ttk.Button(frame_sensibilidad, text="Realizar Análisis", 
                                      command=self.analizar_sensibilidad, state=tk.DISABLED)
        self.btn_analizar.pack(pady=10)
        
        # Marco para gráfico de análisis
        frame_grafico = ttk.LabelFrame(self.tab_analisis, text="Resultados del Análisis")
        frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas para el gráfico de análisis
        self.fig_analisis, self.ax_analisis = plt.subplots(figsize=(8, 5))
        self.canvas_analisis = FigureCanvasTkAgg(self.fig_analisis, frame_grafico)
        self.canvas_analisis.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Inicializar gráfico vacío
        self.ax_analisis.set_title('Análisis de Sensibilidad')
        self.ax_analisis.set_xlabel('Valor de la Variable')
        self.ax_analisis.set_ylabel('Rendimiento Predicho')
        self.fig_analisis.tight_layout()
        self.canvas_analisis.draw()
    
    # Funciones de la pestaña de datos
    def cargar_archivo(self):
        """Permite al usuario seleccionar y cargar un archivo CSV o Excel"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("Archivos CSV", "*.csv"), 
                      ("Archivos Excel", "*.xlsx *.xls"),
                      ("Todos los archivos", "*.*")]
        )
        
        if archivo:
            self.actualizar_estado(f"Cargando archivo {archivo}...")
            
            # Cargar datos
            if self.optimizer.cargar_datos(archivo):
                self.lbl_archivo.config(text=os.path.basename(archivo))
                self.actualizar_tabla_datos()
                self.actualizar_listas_variables()
                self.actualizar_estado(f"Archivo cargado: {os.path.basename(archivo)}")
            else:
                messagebox.showerror("Error", "No se pudo cargar el archivo.")
                self.actualizar_estado("Error al cargar el archivo.")
    
    def actualizar_tabla_datos(self):
        """Actualiza la tabla con los datos cargados"""
        # Limpiar tabla actual
        for item in self.tabla_datos.get_children():
            self.tabla_datos.delete(item)
        
        # Configurar columnas
        self.tabla_datos["columns"] = list(self.optimizer.data.columns)
        self.tabla_datos["show"] = "headings"
        
        for col in self.optimizer.data.columns:
            self.tabla_datos.heading(col, text=col)
            self.tabla_datos.column(col, width=100)
        
        # Mostrar primeras 50 filas
        for i, row in self.optimizer.data.head(50).iterrows():
            self.tabla_datos.insert("", tk.END, values=list(row))
    
    def actualizar_listas_variables(self):
        """Actualiza las listas de selección de variables"""
        # Limpiar listas
        self.lista_disponibles.delete(0, tk.END)
        self.lista_entrada.delete(0, tk.END)
        self.lista_salida.delete(0, tk.END)
        
        # Agregar columnas a lista de disponibles
        for col in self.optimizer.data.columns:
            self.lista_disponibles.insert(tk.END, col)
    
    def mover_variables(self, origen, destino):
        """Mueve las variables seleccionadas entre listas"""
        seleccionados = origen.curselection()
        if not seleccionados:
            return
        
        # Obtener textos seleccionados
        items = [origen.get(idx) for idx in seleccionados]
        
        # Agregar a destino
        for item in items:
            destino.insert(tk.END, item)
        
        # Eliminar de origen (en reversa para mantener índices)
        for idx in reversed(seleccionados):
            origen.delete(idx)
    
    def seleccionar_salida(self):
        """Selecciona la variable de salida"""
        seleccionado = self.lista_disponibles.curselection()
        if not seleccionado:
            return
        
        # Solo permitir una variable de salida
        self.lista_salida.delete(0, tk.END)
        
        # Mover la variable seleccionada
        item = self.lista_disponibles.get(seleccionado[0])
        self.lista_salida.insert(tk.END, item)
        self.lista_disponibles.delete(seleccionado[0])
    
    def quitar_salida(self):
        """Quita la variable de salida"""
        if self.lista_salida.size() == 0:
            return
        
        # Mover de vuelta a disponibles
        item = self.lista_salida.get(0)
        self.lista_disponibles.insert(tk.END, item)
        self.lista_salida.delete(0)
    
    def preprocesar_datos(self):
        """Preprocesa los datos según las variables seleccionadas"""
        # Obtener variables seleccionadas
        variables_entrada = list(self.lista_entrada.get(0, tk.END))
        
        if not variables_entrada:
            messagebox.showerror("Error", "Debe seleccionar al menos una variable de entrada.")
            return
        
        if self.lista_salida.size() == 0:
            messagebox.showerror("Error", "Debe seleccionar una variable de salida.")
            return
        
        variable_salida = self.lista_salida.get(0)
        
        try:
            # Guardar variables para uso posterior
            self.variables_entrada = variables_entrada
            self.variable_salida = variable_salida
            
            # Actualizar combo de variables para análisis
            self.combo_var_analisis["values"] = variables_entrada
            if variables_entrada:
                self.combo_var_analisis.current(0)
            
            # Crear filas para límites de variables
            for widget in self.frame_limites.winfo_children():
                if int(widget.grid_info()["row"]) > 0:
                    widget.destroy()
            
            for i, var in enumerate(variables_entrada):
                row = i + 1
                ttk.Label(self.frame_limites, text=var).grid(row=row, column=0, padx=5, pady=5)
                
                # Calcular valores min/max de los datos
                min_val = self.optimizer.data[var].min()
                max_val = self.optimizer.data[var].max()
                
                min_entry = ttk.Entry(self.frame_limites, width=10)
                min_entry.insert(0, str(min_val))
                min_entry.grid(row=row, column=1, padx=5, pady=5)
                
                max_entry = ttk.Entry(self.frame_limites, width=10)
                max_entry.insert(0, str(max_val))
                max_entry.grid(row=row, column=2, padx=5, pady=5)
            
            test_size = float(self.entry_test_size.get())
            
            # Preprocesar datos
            resultado = self.optimizer.preprocesar_datos(
                variables_entrada, variable_salida, test_size)
            
            if resultado:
                self.X_train, self.X_test, self.y_train, self.y_test = resultado
                messagebox.showinfo("Preprocesamiento", "Datos preprocesados correctamente.")
                self.actualizar_estado("Datos preprocesados correctamente.")
                
                # Habilitar botón de construcción de modelo
                self.btn_construir.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "Error al preprocesar los datos.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al preprocesar los datos: {e}")
            self.actualizar_estado(f"Error: {e}")
    
    # Funciones de la pestaña de modelo
    def construir_modelo(self):
        """Construye el modelo CNN"""
        try:
            if not hasattr(self, 'X_train'):
                messagebox.showerror("Error", "Primero debe preprocesar los datos.")
                return
            
            # Obtener parámetros
            learning_rate = float(self.entry_lr.get())
            
            # Construir modelo
            input_shape = self.X_train.shape[1:]
            modelo = self.optimizer.construir_modelo_cnn(input_shape, learning_rate)
            
            if modelo:
                # Mostrar resumen del modelo
                info_modelo = f"Modelo CNN construido\n"
                info_modelo += f"Forma de entrada: {input_shape}\n"
                info_modelo += f"Total de parámetros: {modelo.count_params()}\n"
                
                self.lbl_info_modelo.config(text=info_modelo)
                self.actualizar_estado("Modelo CNN construido correctamente.")
                
                # Habilitar botón de entrenamiento
                self.btn_entrenar.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "Error al construir el modelo CNN.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al construir el modelo: {e}")
            self.actualizar_estado(f"Error: {e}")
    
    def entrenar_modelo(self):
        """Entrena el modelo CNN"""
        try:
            if not hasattr(self, 'X_train') or self.optimizer.model is None:
                messagebox.showerror("Error", "Primero debe construir el modelo.")
                return
            
            # Obtener parámetros
            epochs = int(self.entry_epochs.get())
            batch_size = int(self.entry_batch_size.get())
            patience = int(self.entry_patience.get())
            
            # Entrenar en un hilo separado para no bloquear la interfaz
            self.actualizar_estado("Entrenando modelo CNN... Esto puede tomar tiempo.")
            self.btn_entrenar.config(state=tk.DISABLED)
            
            threading.Thread(target=self._ejecutar_entrenamiento, 
                           args=(epochs, batch_size, patience)).start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {e}")
            self.actualizar_estado(f"Error: {e}")
            self.btn_entrenar.config(state=tk.NORMAL)
    
    def _ejecutar_entrenamiento(self, epochs, batch_size, patience):
        """Ejecuta el entrenamiento en un hilo separado"""
        try:
            # Entrenar modelo
            history = self.optimizer.entrenar_modelo(
                self.X_train, self.y_train, self.X_test, self.y_test,
                epochs, batch_size, patience
            )
            
            if history:
                # Actualizar gráficos de entrenamiento
                self.actualizar_graficos_entrenamiento(history)
                
                # Habilitar botones
                self.btn_entrenar.config(state=tk.NORMAL)
                self.btn_guardar.config(state=tk.NORMAL)
                self.btn_optimizar.config(state=tk.NORMAL)
                self.btn_analizar.config(state=tk.NORMAL)
                
                self.actualizar_estado("Modelo entrenado correctamente.")
            else:
                messagebox.showerror("Error", "Error al entrenar el modelo.")
                self.btn_entrenar.config(state=tk.NORMAL)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {e}")
            self.actualizar_estado(f"Error: {e}")
            self.btn_entrenar.config(state=tk.NORMAL)
    
    def actualizar_graficos_entrenamiento(self, history):
        """Actualiza los gráficos de entrenamiento"""
        # Limpiar gráficos
        self.ax_train[0].clear()
        self.ax_train[1].clear()
        
        # Graficar pérdida
        self.ax_train[0].plot(history.history['loss'], label='train')
        self.ax_train[0].plot(history.history['val_loss'], label='validation')
        self.ax_train[0].set_title('Pérdida de Entrenamiento')
        self.ax_train[0].set_xlabel('Época')
        self.ax_train[0].set_ylabel('Pérdida (MSE)')
        self.ax_train[0].legend()
        
        # Graficar MAE
        self.ax_train[1].plot(history.history['mae'], label='train')
        self.ax_train[1].plot(history.history['val_mae'], label='validation')
        self.ax_train[1].set_title('Error Absoluto Medio (MAE)')
        self.ax_train[1].set_xlabel('Época')
        self.ax_train[1].set_ylabel('MAE')
        self.ax_train[1].legend()
        
        self.fig_train.tight_layout()
        self.canvas_train.draw()
    
    def guardar_modelo(self):
        """Guarda el modelo entrenado"""
        if self.optimizer.model is None:
            messagebox.showerror("Error", "No hay un modelo para guardar.")
            return
        
        try:
            # Pedir ubicación para guardar
            ruta_modelo = filedialog.asksaveasfilename(
                title="Guardar modelo",
                defaultextension=".h5",
                filetypes=[("Modelo H5", "*.h5"), ("Todos los archivos", "*.*")]
            )
            
            if not ruta_modelo:
                return
            
            # Crear nombre para metadatos
            ruta_metadata = ruta_modelo.replace(".h5", "_metadata.pkl")
            
            # Guardar modelo
            if self.optimizer.guardar_modelo(ruta_modelo, ruta_metadata):
                messagebox.showinfo("Guardar Modelo", "Modelo guardado correctamente.")
                self.actualizar_estado(f"Modelo guardado en {ruta_modelo}")
            else:
                messagebox.showerror("Error", "Error al guardar el modelo.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar el modelo: {e}")
            self.actualizar_estado(f"Error: {e}")
    
    def cargar_modelo(self):
        """Carga un modelo previamente guardado"""
        try:
            # Pedir ubicación del modelo
            ruta_modelo = filedialog.askopenfilename(
                title="Cargar modelo",
                filetypes=[("Modelo H5", "*.h5"), ("Todos los archivos", "*.*")]
            )
            
            if not ruta_modelo:
                return
            
            # Buscar metadatos
            ruta_metadata = ruta_modelo.replace(".h5", "_metadata.pkl")
            if not os.path.exists(ruta_metadata):
                usar_metadata = messagebox.askyesno(
                    "Metadatos no encontrados",
                    "No se encontraron los metadatos del modelo. "
                    "¿Desea especificar una ubicación diferente?"
                )
                
                if usar_metadata:
                    ruta_metadata = filedialog.askopenfilename(
                        title="Cargar metadatos",
                        filetypes=[("Archivo pickle", "*.pkl"), ("Todos los archivos", "*.*")]
                    )
                    if not ruta_metadata:
                        ruta_metadata = None
                else:
                    ruta_metadata = None
            
            # Cargar modelo
            if self.optimizer.cargar_modelo(ruta_modelo, ruta_metadata):
                messagebox.showinfo("Cargar Modelo", "Modelo cargado correctamente.")
                self.actualizar_estado(f"Modelo cargado desde {ruta_modelo}")
                
                # Actualizar interfaz
                if ruta_metadata and hasattr(self.optimizer, 'variables_entrada'):
                    self.variables_entrada = self.optimizer.variables_entrada
                    self.variable_salida = self.optimizer.variable_salida
                    
                    # Actualizar combo de variables para análisis
                    self.combo_var_analisis["values"] = self.variables_entrada
                    if self.variables_entrada:
                        self.combo_var_analisis.current(0)
                
                # Habilitar botones
                self.btn_guardar.config(state=tk.NORMAL)
                self.btn_optimizar.config(state=tk.NORMAL)
                self.btn_analizar.config(state=tk.NORMAL)
                
                # Actualizar info del modelo
                info_modelo = f"Modelo CNN cargado\n"
                if hasattr(self.optimizer, 'variables_entrada'):
                    info_modelo += f"Variables de entrada: {len(self.optimizer.variables_entrada)}\n"
                info_modelo += f"Total de parámetros: {self.optimizer.model.count_params()}\n"
                
                self.lbl_info_modelo.config(text=info_modelo)
            else:
                messagebox.showerror("Error", "Error al cargar el modelo.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el modelo: {e}")
            self.actualizar_estado(f"Error: {e}")
    
    # Funciones de la pestaña de optimización
    def optimizar_proceso(self):
        """Optimiza el proceso utilizando el modelo entrenado"""
        if self.optimizer.model is None:
            messagebox.showerror("Error", "Primero debe entrenar o cargar un modelo.")
            return
        
        try:
            # Obtener límites de variables
            limites = []
            for i, var in enumerate(self.variables_entrada):
                row = i + 1
                min_widget = self.frame_limites.grid_slaves(row=row, column=1)[0]
                max_widget = self.frame_limites.grid_slaves(row=row, column=2)[0]
                
                min_val = float(min_widget.get())
                max_val = float(max_widget.get())
                
                limites.append((min_val, max_val))
            
            # Obtener parámetros de optimización
            metodo = self.combo_metodo.get()
            max_iter = int(self.entry_max_iter.get())
            
            # Optimizar
            self.actualizar_estado("Optimizando proceso... Esto puede tomar tiempo.")
            threading.Thread(target=self._ejecutar_optimizacion, 
                           args=(limites, metodo, max_iter)).start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al configurar la optimización: {e}")
            self.actualizar_estado(f"Error: {e}")
    
    def _ejecutar_optimizacion(self, limites, metodo, max_iter):
        """Ejecuta la optimización en un hilo separado"""
        try:
            # Optimizar
            resultados = self.optimizer.optimizar_proceso(limites, metodo, max_iter)
            
            if resultados:
                # Mostrar resultados
                self.actualizar_resultados_optimizacion(resultados)
                self.actualizar_estado("Proceso optimizado correctamente.")
            else:
                messagebox.showerror("Error", "La optimización no convergió.")
                self.actualizar_estado("Error: La optimización no convergió.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la optimización: {e}")
            self.actualizar_estado(f"Error: {e}")
    
    def actualizar_resultados_optimizacion(self, resultados):
        """Actualiza el widget de texto con los resultados de la optimización"""
        # Habilitar edición
        self.txt_resultados.config(state=tk.NORMAL)
        
        # Limpiar texto actual
        self.txt_resultados.delete(1.0, tk.END)
        
        # Mostrar resultados
        self.txt_resultados.insert(tk.END, "RESULTADOS DE LA OPTIMIZACIÓN\n")
        self.txt_resultados.insert(tk.END, "===========================\n\n")
        
        self.txt_resultados.insert(tk.END, "Parámetros Óptimos:\n")
        for var, valor in resultados['parametros'].items():
            self.txt_resultados.insert(tk.END, f"  {var}: {valor:.4f}\n")
        
        self.txt_resultados.insert(tk.END, f"\nRendimiento Predicho: {resultados['rendimiento']:.4f}\n")
        self.txt_resultados.insert(tk.END, f"Convergencia: {resultados['convergencia']}\n")
        self.txt_resultados.insert(tk.END, f"Iteraciones: {resultados['iteraciones']}\n")
        self.txt_resultados.insert(tk.END, f"Mensaje: {resultados['mensaje']}\n")
        
        # Deshabilitar edición
        self.txt_resultados.config(state=tk.DISABLED)
    
    # Funciones de la pestaña de análisis
    def analizar_sensibilidad(self):
        """Realiza un análisis de sensibilidad para una variable"""
        if self.optimizer.model is None:
            messagebox.showerror("Error", "Primero debe entrenar o cargar un modelo.")
            return
        
        try:
            # Obtener parámetros
            variable = self.combo_var_analisis.get()
            
            if not variable:
                messagebox.showerror("Error", "Debe seleccionar una variable para analizar.")
                return
            
            min_val = float(self.entry_min_analisis.get())
            max_val = float(self.entry_max_analisis.get())
            num_puntos = int(self.entry_puntos_analisis.get())
            
            # Crear valores para evaluar
            valores = np.linspace(min_val, max_val, num_puntos)
            
            # Usar parámetros óptimos como base si están disponibles
            parametros_base = None
            if hasattr(self.optimizer, 'proceso_optimizado') and self.optimizer.proceso_optimizado:
                parametros_base = self.optimizer.proceso_optimizado['parametros']
            
            # Realizar análisis
            self.actualizar_estado(f"Analizando sensibilidad para {variable}...")
            threading.Thread(target=self._ejecutar_analisis, 
                           args=(variable, valores, parametros_base)).start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al configurar el análisis: {e}")
            self.actualizar_estado(f"Error: {e}")
    
    def _ejecutar_analisis(self, variable, valores, parametros_base):
        """Ejecuta el análisis de sensibilidad en un hilo separado"""
        try:
            # Realizar análisis
            resultados = self.optimizer.analizar_sensibilidad(variable, valores, parametros_base)
            
            if resultados:
                # Mostrar resultados
                self.actualizar_grafico_sensibilidad(resultados)
                self.actualizar_estado(f"Análisis de sensibilidad completado para {variable}.")
            else:
                messagebox.showerror("Error", "Error al realizar el análisis de sensibilidad.")
                self.actualizar_estado("Error en el análisis de sensibilidad.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el análisis: {e}")
            self.actualizar_estado(f"Error: {e}")
    
    def actualizar_grafico_sensibilidad(self, resultados):
        """Actualiza el gráfico con los resultados del análisis de sensibilidad"""
        # Limpiar gráfico
        self.ax_analisis.clear()
        
        # Graficar resultados
        variable = resultados['variable']
        valores = resultados['valores']
        rendimientos = resultados['rendimientos']
        
        self.ax_analisis.plot(valores, rendimientos, 'o-', color='blue')
        
        # Marcar el punto óptimo si está disponible
        if hasattr(self.optimizer, 'proceso_optimizado') and self.optimizer.proceso_optimizado:
            params_opt = self.optimizer.proceso_optimizado['parametros']
            rend_opt = self.optimizer.proceso_optimizado['rendimiento']
            
            if variable in params_opt:
                valor_opt = params_opt[variable]
                
                # Verificar si el valor óptimo está dentro del rango analizado
                if min(valores) <= valor_opt <= max(valores):
                    self.ax_analisis.plot(valor_opt, rend_opt, 'o', color='red', markersize=8)
                    self.ax_analisis.axvline(x=valor_opt, color='red', linestyle='--', alpha=0.5)
                    self.ax_analisis.text(valor_opt, min(rendimientos), 
                                         f"Óptimo: {valor_opt:.4f}", 
                                         ha='center', va='bottom', color='red')
        
        # Configurar gráfico
        self.ax_analisis.set_title(f'Análisis de Sensibilidad: {variable}')
        self.ax_analisis.set_xlabel(f'Valor de {variable}')
        self.ax_analisis.set_ylabel('Rendimiento Predicho')
        self.ax_analisis.grid(True, linestyle='--', alpha=0.7)
        
        # Redimensionar y mostrar
        self.fig_analisis.tight_layout()
        self.canvas_analisis.draw()
    
    # Funciones generales
    def actualizar_estado(self, mensaje):
        """Actualiza la barra de estado"""
        self.barra_estado.config(text=mensaje)
        self.update_idletasks()

# Función principal
def main():
    app = InterfazOptimizacion()
    app.mainloop()

if __name__ == "__main__":
    main()
