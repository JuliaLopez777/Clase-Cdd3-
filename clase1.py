import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, stats
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random

class MiModulo:
    """
    Módulo base que contiene herramientas estadísticas:
    - Evaluación de histogramas como estimación de densidad.
    - Estimación de densidades por métodos de núcleo.
    - Modelos de regresión (lineal, logística).
    - Análisis de variables cualitativas (test de chi-cuadrado).
    """

    def genera_histograma(self, h, datos):
        self.datos = np.array(datos)
        bins = np.arange(np.min(self.datos) - h, np.max(self.datos) + h, h)  # Ajuste en el margen inferior
        fr_abs = np.zeros(len(bins) - 1)

        for ind in range(len(self.datos)):
            for ind_bin in range(len(bins) - 1):
                if self.datos[ind] < bins[ind_bin + 1] and self.datos[ind] >= bins[ind_bin]:
                    fr_abs[ind_bin] += 1
                    break

        estim_hist = fr_abs / (len(self.datos) * h)
        return bins, estim_hist

    def evalua_histograma(self, x, h):
        bins, estim_hist = self.genera_histograma(h)
        estimaciones_x = np.zeros(len(x))

        for i in range(len(x)):
            for ind_bin in range(len(bins) - 1):
                if x[i] >= bins[ind_bin] and x[i] < bins[ind_bin + 1]:
                    estimaciones_x[i] = estim_hist[ind_bin]
                    break

        return estimaciones_x
    def kernel_gaussiano(self, x, xi, h):
        return norm.pdf((x - xi) / h) / h

    def kernel_uniforme(self, x, xi, h):
        u = abs((x - xi) / h)
        return 0.5 / h if u <= 1 else 0

    def kernel_cuadratico(self, x, xi, h):
        u = abs((x - xi) / h)
        return (3/4) * (1 - u**2) / h if u <= 1 else 0

    def kernel_triangular(self, x, xi, h):
        u = abs((x - xi) / h)
        return (1 - u) / h if u <= 1 else 0

    def densidad_nucleo(self, datos, h, x_eval, kernel='gaussiano'):
        """
        Estima la densidad usando el método de núcleos.
        """
        n = len(datos)
        estimacion = []
        for x in x_eval:
            suma = 0
            for xi in datos:
                if kernel == 'gaussiano':
                    suma += self.kernel_gaussiano(x, xi, h)
                elif kernel == 'uniforme':
                    suma += self.kernel_uniforme(x, xi, h)
                elif kernel == 'cuadratico':
                    suma += self.kernel_cuadratico(x, xi, h)
                elif kernel == 'triangular':
                    suma += self.kernel_triangular(x, xi, h)
            estimacion.append(suma / n)
        return np.array(estimacion)

class Regresion(MiModulo):

    def ajustar_modelo(self):
        pass

    def predecir(self, x_nuevo):
        pass

    def intervalo_confianza(self):
        pass

class RegresionLineal(Regresion):
    def __init__(self, x, y):
            self.x = np.array(x)
            self.y = np.array(y)
            self.X = sm.add_constant(self.x)  # Añadir la constante para el intercepto
            self.modelo = None
            self.resultados = None

    def ajustar_modelo(self):
        """Ajusta el modelo de regresión lineal."""
        self.modelo = sm.OLS(self.y, self.X).fit()
        self.resultados = self.modelo
        return self.resultados

    def coeficientes(self):
        """Devuelve los coeficientes b0 (intercepto) y b1 (pendiente)."""
        return self.resultados.params

    def residuos(self):
        """Devuelve los residuos del modelo ajustado."""
        return self.resultados.resid

    def predecir(self, x_nuevo, alpha=0.05):
        """
        Realiza una predicción para un nuevo valor de x y devuelve un resumen con intervalos de confianza.
        """
        x_nuevo_array = np.array([[1, x_nuevo]])
        pred = self.resultados.get_prediction(x_nuevo_array)
        resumen = pred.summary_frame(alpha=alpha)

        fila = resumen.iloc[0]
        texto = (
            f"Predicción para x = {x_nuevo}:\n"
            f"- media (mean): {fila['mean']:.4f} : valor medio estimado de y\n"
            f"- mean_ci_lower: {fila['mean_ci_lower']:.4f} : límite inferior del intervalo de confianza para la media\n"
            f"- mean_ci_upper: {fila['mean_ci_upper']:.4f} : límite superior del intervalo de confianza para la media\n"
            f"- obs_ci_lower: {fila['obs_ci_lower']:.4f} : límite inferior del intervalo de predicción\n"
            f"- obs_ci_upper: {fila['obs_ci_upper']:.4f} : límite superior del intervalo de predicción"
        )

        return texto

    def intervalo_confianza(self, alpha=0.05):
        """Calcula los intervalos de confianza para los coeficientes."""
        return self.resultados.conf_int(alpha=alpha)

    def graficar(self, xlabel='X', ylabel='Y'):
        """Grafica los datos y la recta de regresión con etiquetas personalizables en los ejes."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.x, self.y, label='Datos')
        plt.plot(self.x, self.resultados.fittedvalues, color='red', label='Recta de Regresión')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Regresión Lineal')
        plt.legend()
        plt.show()

    def qq_plot(self):
        """Genera un qq plot de los residuos."""
        plt.figure(figsize=(8, 6))
        stats.probplot(self.residuos(), dist="norm", plot=plt)
        plt.title('Q-Q Plot de los Residuos')
        plt.show()

    def analisis_residuos(self, xlabel='X', ylabel='Y'):
        """Genera un gráfico de los residuos vs los valores predichos con etiquetas personalizables en los ejes."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.x, self.residuos(), color='blue')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Diagnóstico de Residuos')
        plt.show()

    def analisis_p_valor(self, alpha=0.05):
        """
        Analiza el p-valor de la pendiente para evaluar si hay evidencia suficiente
        para rechazar la hipótesis nula de que la pendiente es cero.
        """
        p_valor = self.resultados.pvalues[1]  # El p-valor del coeficiente de x
        texto = f"P-valor para la pendiente: {p_valor:.4f}\n"

        if p_valor < alpha:
            texto += (
                f"Como el p-valor es menor que {alpha}, hay evidencia suficiente "
                "para rechazar la hipótesis nula.\n"
                " La pendiente es estadísticamente significativa.")
        else:
            texto += (
                f"Como el p-valor es mayor que {alpha}, no hay evidencia suficiente "
                "para rechazar la hipótesis nula.\n"
                " No se puede afirmar que la pendiente sea significativamente distinta de cero.")

        return texto

    def resumen(self):
        """Devuelve un resumen del modelo ajustado."""
        return self.resultados.summary()


    def estimar_sigma_cuadrado(self):
        """Estima la varianza del error (σ²)."""
        residuos = self.residuos()
        n = len(self.x)
        sigma_cuadrado = np.sum(residuos**2) / (n - 2)  # Fórmula de varianza del error
        return sigma_cuadrado

    def indice_error_maximo(self):
        residuos = self.residuos()  # Puede ser np.array o pd.Series
        residuos_abs = np.abs(residuos)

        if isinstance(residuos, pd.Series):
            indice_max = residuos_abs.idxmax()
        else:
            indice_max = np.argmax(residuos_abs)

        residuo_max = residuos_abs[indice_max]
        return residuo_max, indice_max

class DivisorDatos:
    """
    Clase para dividir datos en conjuntos de entrenamiento y prueba (train/test).
    """

    def __init__(self, test_size=0.3, random_state=None):
        self.test_size = test_size
        if random_state is None:
            random_state = random.randint(0, 10000)
        self.random_state = random_state

    def dividir(self, X, y):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Parámetros:
        - X: variables independientes (puede ser array o DataFrame)
        - y: variable dependiente (array o Serie)

        Devuelve:
        - X_train, X_test, y_train, y_test
        """
        X = np.array(X)
        y = np.array(y)
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

#---------------------------REGRESION LOGISTICA -------------------------------

class RegresionLogistica(Regresion):
    """
    Clase que implementa una regresión logística utilizando statsmodels.

    Hereda de la clase Regresion y ajusta un modelo logístico binario
    (valores 0 o 1) sobre los datos proporcionados.
    """

    def __init__(self, x, y):
        """
        Inicializa la clase con los datos x e y y ajusta el modelo logístico.

        Parámetros:
        x : array
            Variable(s) independiente(s).
        y : array
            Variable dependiente (debe ser binaria: 0 o 1).
        """
        x = np.array(x)
        y = np.array(y)

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("La variable dependiente y debe ser binaria (0 o 1).")

        self.x = x
        self.y = y

        x_const = sm.add_constant(x)
        self.modelo = sm.Logit(y, x_const)
        self.resultado = self.modelo.fit()

    def ajustar_modelo(self):
        """
        Ajusta el modelo de regresión logística utilizando los datos proporcionados
        durante la inicialización de la clase.

        Retorna:
        -------
        resultado del modelo ajustado (statsmodels)
        """
        ...
        self.modelo = sm.Logit(self.y, self.X)
        self.resultados = self.modelo.fit()
        return self.resultados

    def coeficientes(self):
        """
        Devuelve los coeficientes del modelo ajustado.
        """
        self.beta0 = self.resultados.params[0]
        self.beta1 = self.resultados.params[1]
        return self.beta0, self.beta1

    def predecir(self, x_nuevo, umbral=0.5):
        """
        Realiza predicciones de probabilidad con nuevos valores de x.

        Parámetros:
        ----------
        nuevo_x : array-like
            Nuevos valores de la variable independiente.

        Retorna:
        -------
        array
            Probabilidades predichas.
        """
        x_nuevo_array = sm.add_constant(np.array(x_nuevo))
        probabilidades = self.resultados.predict(x_nuevo_array)
        predicciones = (probabilidades >= umbral).astype(int)
        return predicciones, probabilidades

    def matriz_confusion(self, x_test, y_test, umbral=0.5):
        """
        Calcula la matriz de confusión comparando las predicciones del modelo con valores reales.

        Retorna:
        - matriz: np.array de 2x2 con la forma:
            [[Verdaderos Negativos, Falsos Positivos],
             [Falsos Negativos, Verdaderos Positivos]]
        """
        y_pred, _ = self.predecir(x_test, umbral)
        y_test = np.array(y_test)

        vp = np.sum((y_pred == 1) & (y_test == 1))
        vn = np.sum((y_pred == 0) & (y_test == 0))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))

        matriz = np.array([[vn, fp], [fn, vp]])
        return matriz

    def sensibilidad_especificidad(self, x_test, y_test, umbral=0.5):
        """
        Calcula la sensibilidad y especificidad del modelo.

        Retorna:
        - sensibilidad: FLOTante
        - especificidad: flotante
        """
        matriz = self.matriz_confusion(x_test, y_test, umbral)
        vn, fp = matriz[0]
        fn, vp = matriz[1]

        sensibilidad = vp / (vp + fn) if (vp + fn) != 0 else 0
        especificidad = vn / (vn + fp) if (vn + fp) != 0 else 0

        return sensibilidad, especificidad

    def calcular_auc(self, x_test, y_test):
        """
        Calcula el AUC (Área bajo la curva ROC) y grafica la curva ROC.

        Retorna:
        - auc_valor: float
        """
        from sklearn.metrics import roc_curve, auc

        x_test_const = sm.add_constant(np.array(x_test))
        probabilidades = self.resultados.predict(x_test_const)
        fpr, tpr, _ = roc_curve(y_test, probabilidades)
        auc_valor = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc_valor:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Falsos Positivos')
        plt.ylabel('Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend()
        plt.grid(True)
        plt.show()

        return auc_valor

    def resumen(self):
        """
        Muestra el resumen estadístico del modelo ajustado.
        retorna tambien t observado, p valor y el error estandar
        """
        self.t_observado = self.resultados.tvalues[1]
        self.p_valor = self.resultados.pvalues[1]
        self.error_estandar = self.resultados.bse[1]
        return self.resultados.summary(), self.t_observado, self.p_valor, self.error_estandar

#--------------------------------GENERADOR DE DATOS--------------------------------
class GeneradorDatos:
    """
    Clase para generar datos con distintas distribuciones y estimar sus densidades.
    Soporta:
    - Distribución Normal
    - Distribución Bart Simpson (BS)
    """

    def __init__(self, N=1000):
        """
        atributos = N (tamanio de la muestra)
        Inicializa la clase con el número de datos a generar.
        """
        self.N = N
        self.datos_normal = None
        self.datos_bart_simpson = None

    def generar_normal(self, media=0, desviacion=1):
        """Genera datos con distribución normal."""
        self.datos_normal = np.random.normal(media, desviacion, size=self.N)
        return self.datos_normal

    def generar_bart_simpson(self):
        """
        Genera datos con distribución Bart Simpson (BS).
        """
        u = np.random.uniform(size=self.N)
        datos_bs = u.copy()

        ind = np.where(u > 0.5)[0]
        datos_bs[ind] = np.random.normal(0, 1, size=len(ind))

        for j in range(5):
            ind = np.where((u > j * 0.1) & (u <= (j + 1) * 0.1))[0]
            datos_bs[ind] = np.random.normal(j / 2 - 1, 0.1, size=len(ind))

        self.datos_bart_simpson = datos_bs
        return datos_bs

    def densidad_normal(self, graficar=True):
        """Estima y grafica la densidad de los datos normales generados."""
        if self.datos_normal is None:
            raise ValueError("Primero se deben generar los datos.")
        kde = gaussian_kde(self.datos_normal)
        x = np.linspace(min(self.datos_normal), max(self.datos_normal), 1000)
        y = kde(x)

        if graficar:
            plt.plot(x, y, label="Densidad Normal")
            plt.title("Estimación de Densidad - Normal")
            plt.xlabel("Valor")
            plt.ylabel("Densidad")
            plt.grid(True)
            plt.legend()
            plt.show()

        return x, y

    def densidad_bart_simpson(self, graficar=True):
        """Estima y grafica la densidad de los datos Bart Simpson (BS)generados."""
        if self.datos_bart_simpson is None:
            raise ValueError("Primero debe generar los datos BS.")
        kde = gaussian_kde(self.datos_bart_simpson)
        x = np.linspace(min(self.datos_bart_simpson), max(self.datos_bart_simpson), 1000)
        y = kde(x)

        if graficar:
            plt.plot(x, y, label="Densidad Bart Simpson", color='orange')
            plt.title("Estimación de Densidad - Bart Simpson")
            plt.xlabel("Valor")
            plt.ylabel("Densidad")
            plt.grid(True)
            plt.legend()
            plt.show()

        return x, y
# -------------------------------PARTE CUALITATIVAS (CHI CUADRADO)-------------------------------------------
class Cualitativas:
    """
    Clase para análisis de variables cualitativas. Permite realizar un test de bondad de ajuste
    para comparar frecuencias observadas y esperadas usando chi cuadrado.
    """
    def __init__(self, observadas, probabilidades_esperadas):
        """
        observadas: lista o array con frecuencias observadas.
        probabilidades_esperadas: lista con probabilidades esperadas bajo H0 (suma debe ser 1).
        """
        self.observadas = np.array(observadas)
        self.n = np.sum(self.observadas)
        self.prob_esp = np.array(probabilidades_esperadas)

        if not np.isclose(np.sum(self.prob_esp), 1):
            raise ValueError("Las probabilidades esperadas deben sumar 1.")

        self.esperadas = self.n * self.prob_esp
        self.k = len(self.observadas)
        self.grados_libertad = self.k - 1

    def calcular_chi2(self):
        """
        Calcula el estadístico chi-cuadrado observado.
        """
        chi2_obs = np.sum((self.observadas - self.esperadas) ** 2 / self.esperadas)
        return chi2_obs

    def valor_critico(self, alpha=0.05):
        """
        Calcula el valor crítico teórico para un nivel de significancia alpha.
        """
        return chi2.ppf(1 - alpha, df=self.grados_libertad)

    def p_valor(self):
        """
        Calcula el p-valor asociado al estadístico chi-cuadrado observado.
        """
        chi2_obs = self.calcular_chi2()
        return 1 - chi2.cdf(chi2_obs, df=self.grados_libertad)

    def resumen_conclusiones(self, alpha=0.05):
        """
        Devuelve un resumen con:
        - chi2 observado
        - chi2 teórico
        - p-valor
        - conclusiones usando ambos criterios
        """
        chi2_obs = self.calcular_chi2()
        chi2_teorico = self.valor_critico(alpha)
        p = self.p_valor()

        conclusion_critico = (
            "Se rechaza H0: hay diferencias significativas entre lo observado y lo esperado."
            if chi2_obs > chi2_teorico else
            "No se rechaza H0: no hay evidencia suficiente para decir que hay diferencias significativas."
        )

        conclusion_pvalor = (
            f"Se rechaza H0: p-valor = {p:.4f} < alpha = {alpha}."
            if p < alpha else
            f"No se rechaza H0: p-valor = {p:.4f} ≥ alpha = {alpha}."
        )

        resumen = (
            f"Chi-cuadrado observado: {chi2_obs:.4f}\n"
            f"Chi-cuadrado teórico (alpha = {alpha}): {chi2_teorico:.4f}\n"
            f"P-valor: {p:.4f}\n\n"
            f"Conclusión según valor crítico: {conclusion_critico}\n"
            f"Conclusión según p-valor: {conclusion_pvalor}"
        )

