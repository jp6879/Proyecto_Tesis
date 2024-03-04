# En este momento decidimos cuales son los parámetros que dejamos fijos al momento de generar los datos
# Se dejan fijos el numero de compartimientos N, el rango de tamaños de compartimientos l0 y lf, el tiempo final de simulación lf
# Además se muestrea el tiempo en dos partes, una corta y una larga para tener una mejor resolución en la parte inicial de la señal
# Siendo esta imporatnte para diferenciar las señales de diferentes distribuciones de tamaños de compartimientos en el caso de que estas sean muy similares (lcm y σ grandes)

# Parámetros fijos
N = 5000
time_sample_lenght_long = 1000
time_sample_lenght_short = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 45

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N) # Esto nos da un muestreo de 0,008998 μm en lc
t_short = collect(range(0, 0.1, length = time_sample_lenght_short)) # Muestreo corto de 0.1 ms
t_long = collect(range(0.1, 1, length = time_sample_lenght_long)) # Muestreo largo de 10 ms

# Concatenamos los tiempos para tener un muestreo completo 
t = vcat(t_short, t_long)

# Parametros que se varian, estos se corresponden a la mediana y la desviación estándar de la distribución de tamaños de compartimientos lcms en μm y σs adimensionales

lcms = 0.5:0.01:6
σs = 0.01:0.01:1

