# Módulo de Herramientas Estadísticas

Este módulo contiene una colección de clases en Python para realizar análisis estadísticos como:

- Estimación de densidades (histogramas y núcleos)
- Regresiones lineales y logísticas
- Análisis de variables cualitativas (pruebas de bondad de ajuste)(chi cuadrado)

### MiModulo
Contiene herramientas para:

- Generar histogramas como estimadores de densidad
- Estimar densidades por el método de núcleos (kernel Gaussiano, uniforme, triangular, cuadrático)

### DivisorDatos
Divide datps en conjuntos de entrenamiento y prueba usando sklearn.

### RegresionLineal
Implementa regresión lineal simple usando statsmodels, con herramientascomo:

- Análisis de residuos
- QQ-plot
- Gráficos y resumen del modelo
- Intervalos de confianza y predicción

### RegresionLogistica
Implementa regresión logística binaria(o y 1). Permite:

- Ajustar modelos logísticos
- Predecir clases y probabilidades
- Generar matriz de confusión, sensibilidad, especificidad y curva ROC con AUC
- Permite obtener un resumen donde resultan los coeficientes, errores estandar, t obs y pvalor

### Cualitativas
Clase diseñada para realizar pruebas de bondad de ajuste usando el test chi cuadrado.

Permite:

- Calcular el estadístico chi cuadrado observado
- Obtener el valor crítico y p-valor
- Realizar conclusiones con ambos criterios

---

## Ejemplo de uso: Análisis de Variables Cualitativas

Supongamos que tienes una distribución observada de preferencias de colores entre un grupo de personas, y se desea probar si las personas eligen los colores con igual probabilidad.

```
# Frecuencias observadas (por ejemplo: rojo, azul, verde)
observadas = [30, 50, 20]

# h0: todas las categorías son igualmente probables
probabilidades_esperadas = [1/3, 1/3, 1/3]

# test
test = Cualitativas(observadas, probabilidades_esperadas)

# resumen completo
print(test.resumen_conclusiones(alpha=0.05))
