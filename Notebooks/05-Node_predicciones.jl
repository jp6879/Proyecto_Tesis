### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 81f922dc-76d0-4a98-b691-b180aa911b26
using PlutoUI

# ╔═╡ 41be4360-5c24-4b23-b25f-48a2b187cbc1
md"# 7 - Neural Ordinary Differential Equations para la predicción de una serie temporal

* Construimos una Neural ODE con el objetivo de predecir señales de Hahn a partir de una cantidad reducida de datos. Como las señales de Hahn se cruzan entre sí tanto en el mismo punto inicial como en diferentes puntos, utilizamos un embedding para representar las señales de Hahn en un espacio de características de mayor dimensión. Este embedding está compuesto por la señal de Hahn original y la derivada de la señal de Hahn. Luego, utilizamos una Neural ODE para predecir la señal de Hahn en el futuro.

* Para tener un conjunto de señales sencillos para entrenar y chequear que funciona, utilizo señales de Hahn simuladas con un mismo $\sigma = 1$ y ver si la red puede predecir 50 señales de estas. Para un proceso mas complejo se buscaria entrenar con multiples señales de Hahn con diferentes $\sigma$.

* De manera mas general seleccionamos un subconjunto reducido de señales que sea representativo aproximadamente de todas las señales que tenemos simuladas en el dataset. El cual nos daría una red que cumpliría con las predicciones de los datos de manera general."

# ╔═╡ d2d4ea40-a472-4d24-a66b-8be285fca558
md"### Importamos las librerias necesarias"

# ╔═╡ 75189838-98d6-495a-8421-1f0ebbea6899
begin
#	using Flux
#	using Flux: train!
#	using DataFrames
#	using CSV
#	using DifferentialEquations
#	using SciMLSensitivitya
#   using ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux
#	using Interpolations
#	using OrdinaryDiffEq
#	using IterTools: ncycle
end

# ╔═╡ f27f194d-c3c1-42d5-b80b-4309914fe3a3
md"### Cargamos los parámetros fijos"

# ╔═╡ 6b605ab7-f8e2-49aa-8b85-2a2989d677d1
begin
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
	σs = 0.01:0.01:1;
end

# ╔═╡ 068dcddc-49cf-4696-a0f6-26a42d318a76
md"### Utilidades para cargar las señales a partir de un rango de índices"

# ╔═╡ 4a057240-4179-457c-ae63-9972a23d81b0
md"### Función para las derivadas"

# ╔═╡ bbe34a66-12da-46a0-b62f-a9b110584afd
# Vamos a hacer una función que nos permita calcular las derivadas de las señales
# Para esto vamos a usar diferencias finitas centradas
# La función recibe como argumentos el arreglo de señales y el arreglo de tiempos
# La función regresa un arreglo de derivadas de las señales

function derivate_signals(t,signal)
    # Calcula el tamaño de la ventana
    w = 1
    # Calcula el tamaño de la señal
    n = length(signal)
    # Inicializa el arreglo de derivadas
    derivadas = zeros(n)
    for i in 1:n
        # Encuentra los índices de la ventana
        inicio = max(1, i-w)
        final = min(n, i+w)
        # Utiliza diferencias finitas centradas si es posible
        if inicio != i && final != i
            derivadas[i] = (signal[final] - signal[inicio]) / (t[final] - t[inicio])
        elseif inicio == i
            # Diferencia hacia adelante si estamos en el comienzo del arreglo
            derivadas[i] = (signal[i+1] - signal[i]) / (t[i+1] - t[i])
        else
            # Diferencia hacia atrás si estamos al final del arreglo
            derivadas[i] = (signal[i] - signal[i-1]) / (t[i] - t[i-1])
        end
    end
    return derivadas
end;

# ╔═╡ d5bbf401-92ec-4475-a987-ccebc37eeee7
begin
	# Función para leer las señales
	function GetSignals(path_read)
	    dataSignals = CSV.read(path_read * "/dataSignals.csv", DataFrame)
	    dataSignals = Matrix(dataSignals)
	    return dataSignals
	end
	
	# Funcion que tomas algunas señales para utilizarlas en el entrenamiento de la NODE, recibe el numero de señales que quermos tomar y de donde las queremos tomar
	function Get_Signals_Test(rango, path_read, muestreo_corto, muestreo_largo)
	    # Leemos las señales desde el archivo
	    dataSignals = Float32.(GetSignals(path_read))
	    # Tomamos únicamente el rango de señales que vamos a tomar para esta red
	    Signals_test = Float32.(Matrix(dataSignals[:,rango]'))
	    # Tomamos un subconjunto de tiempos para agilizar el entrenamiento, tiempos cortos es decir el muestreo hasta 0.1 s
	    Signals_test_short = Signals_test[:,1:muestreo_corto:1000]
	    # Tiempos largos es decir el muestreo después de 0.1 s
	    Signals_test_long = Signals_test[:,1001:muestreo_largo:end]
	    # Concatenamos las señales con el nuevo muestreo para agilizar el entrenamiento
	    Signals_test = hcat(Signals_test_short, Signals_test_long)
	    return Signals_test
	end
	
	###################################################################################
	# Función que idientifca las señales utilizadas, es decir los parámetros lcm y σ
	function Get_Signals_Test_Parameters(numSignals,lcms,σs)
	    dim1 = dimlcm = length(lcms)
	    dim2 = dimσ = length(σs)
	
	    # Inicializamos los arreglos que vamos a utilizar
	    column_lcm = zeros(dim1*dim2)
	    column_σs = zeros(dim1*dim2)
	    # Convertimos los iterables de lcm y sigma con los cuales generamos las señales en arreglos
	    aux_lcm = collect(lcms)
	    aux_σs = collect(σs)
	
	    # Asignamos los valores de lcm y sigma ordenados tal cual están las señales, es decir los datos comienzan en σ = 0.1 y lcm = 0.5 y va en aumento de lcm hasta 6 y luego pasa al siguiente σ
	    for i in 1:dim1
	        for j in 1:dim2
	            column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
	            column_σs[(i - 1)*dim2 + j] = aux_σs[j]
	        end
	    end
	    # Con esto ya tenemos los arrelgos de los parámetros que se utilizaron para generar las señales
	    column_lcm_test = column_lcm[1:Int(length(column_lcm)/numSignals):end]
	    column_σs_test = column_σs[1:Int(length(column_σs)/numSignals):end]
	    
	    return column_lcm_test, column_σs_test
	end
	
	# Función que devuelve señales de prueba, sus derivadas y los parámetros con los que se generaron
	function Get_Signals_Data_Training(path_read, rango, lcms, sigmas, muestreo_corto, muestreo_largo)
	    # Obtenemos las señales que vamos a utilizar
	    Signals_test = Get_Signals_Test(rango, path_read, muestreo_corto, muestreo_largo)
	    # Obtenemos los parámetros con los que se generaron las señales
	    column_lcm_test, column_sigmass_test = Get_Signals_Test_Parameters(rango,lcms,sigmas)
	    # Calculamos las derivadas de las señales
	    Signals_test_derivadas = zeros(size(Signals_test))
	    for i in 1:size(Signals_test)[1]
	        Signals_test_derivadas[i,:] = derivate_signals(t,Signals_test[i,:])
	    end
	    Signals_test_derivadas = Float32.(Matrix(Signals_test_derivadas'))
	    # Normalizamos las derivadas
	    for i in 1:size(Signals_test)[1]
	        Signals_test_derivadas[:,i] = Signals_test_derivadas[:,i] ./ maximum(abs.(Signals_test_derivadas[:,i]))
	    end
	    return Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmass_test
	end;
end

# ╔═╡ bb42d2b8-9a84-419a-ad7f-f92fc9c8bb58
md"### Entrenamiento de las NODE"

# ╔═╡ 9c694f8f-19ee-4554-a8a4-910c7470ad53
md"### Función main donde se exploran distintas redes"

# ╔═╡ 7441f18e-f9c0-4ffb-af0f-6467eff7d489
md"Los resultados obtenidos que se presentaron con la mejor red en este entrenamiento no fueron buenos"

# ╔═╡ 6f2ab7ee-149e-4f9d-a5bc-1fe1d737ed9f
md"# 8 -Exploración para un conjunto de señales mas complejo pero representativo del dataset que tenemos

* Creamos nuevas funciones para cargar las señales de Hahn para poder tener los $\sigma$ que queremos y las $l_{cm}$ que queremos, con esto creamos un conjuno de señales representativo de las señales que tenemos en el dataset."

# ╔═╡ 8f4d677d-248c-4187-be29-033b85fc7474
PlutoUI.Resource("https://imgur.com/eXTmj8l.png")

# ╔═╡ a2e08efa-93f5-4022-a335-74b111c8fb00
md"* En este caso tenemos 60 señales con $\sigma = \{0.01,~0.2,~0.4,~0.6,~0.8,~1\}$ y $l_{cm} = \{0.5,~0.75,~1.0,~1.25,~1.5,~1.75,~2.0,~2.25,~2.5,~2.75\}$"

# ╔═╡ 2d472dc9-7e40-4488-b13a-a736b09f96de
md"#### Nueva función de main donde se exploran las nuevas redes entrenadas"

# ╔═╡ 1a3eb77e-740d-447e-9af2-d972ab2bd208
md"### El programa no difiere mucho del anterior, solo que ahora se seleccionan las señales de Hahn de manera mas representativa y se exploraron únicamente 12 señales

* Se añade la implementación de una función costo con penalización pensando en que las funciones deben ser estrictamente decrecientes añadiendo el término:

$$\begin{equation}

\lambda \sum_{j=1}^{\mu} \sum_{i=1}^{N-1} max\left(S^{(j)}(t_{i+1}) - S^{(j)}(t_i),\: 0\right),

\end{equation}$$

donde $\lambda$ es un nuevo hiperparámetro del modelo."

# ╔═╡ 6b934743-ce99-4f66-a1c7-cb71b321c3b5
md"#### Función de entrenamiento nueva donde se añade el parámetro $\lambda$"

# ╔═╡ 49f5148a-e552-4896-9e94-a58092e27826
md"#### Nueva función de main donde se exploran las nuevas redes entrenadas"

# ╔═╡ 8a3d2bc4-2301-4d8b-b43f-fe6fe0435408
md"Se resumen los resultados en la siguiente tabla"

# ╔═╡ ed21550e-cb50-43ba-903d-061b5f69975a
md"$$\begin{aligned}
& \begin{array}{cccccc}
\hline \hline
\textbf{ID} & \textbf{Arq} & \textbf{Ac} & \textbf{Opt} & \textbf{BS} & \textbf{L Tr} & \textbf{Loss Pred} \\ \hline
1  & 2, 16, 16, 1            & relu                 & AdamW                 & 50                   & 0.15966              & 0.05333               \\ \hline
2  & 2, 16, 16, 1            & tanh                 & AdamW                 & 50                   & 0.19056              & 0.07639               \\ \hline
4  & 2, 32, 64, 16, 1        & relu                 & AdamW                 & 50                   & 0.07914              & 0.02104               \\ \hline
5  & 2, 32, 64, 16, 1        & tanh                 & AdamW                 & 50                   & 0.13468              & 0.01940               \\ \hline
6  & 2, 32, 64, 16, 1        & swish                & AdamW                 & 50                   & 0.07149              & 0.02450               \\ \hline
7  & 2, 128, 64, 16, 1       & relu                 & AdamW                 & 50                   & 0.06233              & 0.01870               \\ \hline
8  & 2, 128, 64, 16, 1       & tanh                 & AdamW                 & 50                   & 0.12185              & 0.01868               \\ \hline
9  & 2, 128, 64, 16, 1       & swish                & AdamW                 & 50                   & 0.06562              & 0.01956               \\ \hline
10 & 2, 64, 128, 64, 32, 16, 1 & relu                & AdamW                 & 50                   & 0.04986              & 0.01680               \\ \hline
12 & 2, 64, 128, 64, 32, 16, 1 & swish               & AdamW                 & 50                   & 0.04708              & 0.01640               \\ \hline
\hline
\end{array}
\end{aligned}$$"

# ╔═╡ 69376adb-64f8-4204-ad32-10039ac55594
md"De estos resultados obtenemos que la arquitectura 12 es aquella que obtiene un menor error final de loss tanto de entrenamiento como de predicción, sin embargo continuan siendo valores altos para el problema que se está tratando de resolver, por lo observado anteriormente en las otras exploraciones de redes.
Aún así se muestra una visualización de los resultados obtenidos con esta arquitectura."

# ╔═╡ ff6806fb-edf0-4b96-b8b6-e6d1d978ca7f
PlutoUI.Resource("https://imgur.com/Q9QDJN6.png")

# ╔═╡ 7c088c54-afd5-483f-ad52-830baf67ea85
md"Debido a que la mayoria de las señales en la predicción tienen valores cercanos a 0 debido a que ocurren para valores de $t$ cercanos al decaimiento es que tenemos un error de predicción menor al de entrenamiento, entonces en principio no se puede decir que la red haya aprendido a generalizar correctamente.

Veamos las predicciones de la red neuronal para ciertas señales
"

# ╔═╡ ab90dfea-d78f-4d94-a0dc-a12e0388667c
PlutoUI.Resource("https://imgur.com/qmKl741.png")

# ╔═╡ e6d90d95-d580-4214-ba7d-755cb27da810
PlutoUI.Resource("https://imgur.com/N3JaflY.png")

# ╔═╡ 2e20900c-696d-44fb-b638-fc53aac18d4e
PlutoUI.Resource("https://imgur.com/ZeowymY.png")

# ╔═╡ d35cbba2-e247-4469-b07a-125323f05076
PlutoUI.Resource("https://imgur.com/v7FHaqw.png")

# ╔═╡ e3108149-acd3-4128-989d-e484882a3db6
md"Sigue ocurriendo que las señales se amontonan quizas en la zona donde mas densidad de datos hay. Por lo que la red no aprende a generalizar correctamente.

Traté de sacar tantas señales similares en esos puntos dando el siguiente conjunto de señales menos representativo que el anteriro pero aún así cubriendo el espacio de señales que tenemos en el dataset.

En este caso no se utilizó la penalización en la función de costo."

# ╔═╡ b982949f-3307-4225-aac2-96a93f386012
md"* En este caso tenemos 30 señales mezclando $\sigma = \{0.01,~0.2,~0.4,~0.6,~0.8,~1\}$ y $l_{cm} = \{0.5,~0.75,~1.0,~1.25,~1.5\}$"

# ╔═╡ d267aaa5-368b-4a7d-a683-9afdeeaa5ba0
PlutoUI.Resource("https://imgur.com/DhF7L73.png")

# ╔═╡ d9eeddf2-518c-47e5-b96a-4a3326237e2b
md"Los resultados de la exploración de las redes entrenadas con este conjunto de señales son los siguientes"

# ╔═╡ 68a7a366-65fb-4c04-b507-8a5084364859
md"$$\begin{aligned}
& \begin{array}{cccccc}
\hline \hline
\textbf{ID} & \textbf{Arq} & \textbf{Ac} & \textbf{Opt} & \textbf{BS} & \textbf{L Tr} & \textbf{Loss Pred} \\ \hline
1  & 2, 8, 1            & relu                 & AdamW                 & 30                   & 0.18138              & 0.06550               \\ \hline
2  & 2, 8, 1            & relu                 & AdamW                 & 60                   & 0.29828              & 0.91709               \\ \hline
3  & 2, 8, 1            & relu                 & RMSProp               & 30                   & 0.21141              & 0.08631               \\ \hline
4  & 2, 8, 1            & relu                 & RMSProp               & 60                   & 0.32081              & 0.72909               \\ \hline
5  & 2, 8, 1            & tanh                 & AdamW                 & 30                   & 0.16616              & 0.12923               \\ \hline
6  & 2, 8, 1            & tanh                 & AdamW                 & 60                   & 0.20412              & 0.06995               \\ \hline
7  & 2, 8, 1            & tanh                 & RMSProp               & 30                   & 0.13303              & 0.31857               \\ \hline
8  & 2, 8, 1            & tanh                 & RMSProp               & 60                   & 0.23196              & 0.09752               \\ \hline
9  & 2, 8, 1            & swish                & AdamW                 & 30                   & 0.20328              & 0.04683               \\ \hline
10 & 2, 8, 1            & swish                & AdamW                 & 60                   & 0.22994              & 0.13114               \\ \hline
11 & 2, 8, 1            & swish                & RMSProp               & 30                   & 0.27957              & 0.39572               \\ \hline
12 & 2, 8, 1            & swish                & RMSProp               & 60                   & 0.29670              & 0.51079               \\ \hline
13 & 2, 16, 16, 1       & relu                 & AdamW                 & 30                   & 0.12436              & 0.23578               \\ \hline
14 & 2, 16, 16, 1       & relu                 & AdamW                 & 60                   & 0.20449              & 0.06106               \\ \hline
15 & 2, 16, 16, 1       & relu                 & RMSProp               & 30                   & 0.15192              & 0.12456               \\ \hline
16 & 2, 16, 16, 1       & relu                 & RMSProp               & 60                   & 0.14615              & 0.12199               \\ \hline
17 & 2, 16, 16, 1       & tanh                 & AdamW                 & 30                   & 0.10095              & 0.23465               \\ \hline
18 & 2, 16, 16, 1       & tanh                 & AdamW                 & 60                   & 0.12239              & 0.35243               \\ \hline
19 & 2, 16, 16, 1       & tanh                 & RMSProp               & 30                   & 0.10738              & 0.30425               \\ \hline
20 & 2, 16, 16, 1       & tanh                 & RMSProp               & 60                   & 0.10979              & 0.30945               \\ \hline
21 & 2, 16, 16, 1       & swish                & AdamW                 & 30                   & 0.12717              & 0.44012               \\ \hline
22 & 2, 16, 16, 1       & swish                & AdamW                 & 60                   & 0.17742              & 0.08315               \\ \hline
23 & 2, 16, 16, 1       & swish                & RMSProp               & 30                   & 0.09651              & 0.36198               \\ \hline
24 & 2, 16, 16, 1       & swish                & RMSProp               & 60                   & 0.16837              & 0.07821               \\ \hline
25 & 2, 32, 64, 16, 1   & relu                 & AdamW                 & 30                   & 0.06255              & 0.05192               \\ \hline
26 & 2, 32, 64, 16, 1   & relu                 & AdamW                 & 60                   & 0.08133              & 0.08197               \\ \hline
27 & 2, 32, 64, 16, 1   & relu                 & RMSProp               & 30                   & 0.05814              & 0.04315               \\ \hline
28 & 2, 32, 64, 16, 1   & relu                 & RMSProp               & 60                   & 0.09630              & 0.38492               \\ \hline
29 & 2, 32, 64, 16, 1   & tanh                 & AdamW                 & 30                   & 0.06744              & 0.03438               \\ \hline
30 & 2, 32, 64, 16, 1   & tanh                 & AdamW                 & 60                   & 0.07898              & 0.07613               \\ \hline
32 & 2, 32, 64, 16, 1 & tanh & RMSProp & 60 & 0.07112 & 0.03230 \\ \hline
33 & 2, 32, 64, 16, 1 & swish & AdamW & 30 & 0.06724 & 0.11372 \\ \hline
34 & 2, 32, 64, 16, 1 & swish & AdamW & 60 & 0.08945 & 0.30299 \\\hline
35 & 2, 32, 64, 16, 1 & swish & RMSProp & 30 & 0.06251 & 0.08129 \\\hline
36 & 2, 32, 64, 16, 1 & swish & RMSProp & 60 & 0.08936 & 0.22087 \\\hline
37 & 2, 128, 64, 16, 1 & relu & AdamW & 30 & 0.05204 & 0.02930 \\\hline
38 & 2, 128, 64, 16, 1 & relu & AdamW & 60 & 0.05856 & 0.03259 \\\hline
39 & 2, 128, 64, 16, 1 & relu & RMSProp & 30 & 0.04697 & 0.02880 \\\hline
40 & 2, 128, 64, 16, 1 & relu & RMSProp & 60 & 0.05734 & 0.03662 \\\hline
41 & 2, 128, 64, 16, 1 & tanh & AdamW & 30 & 0.06668 & 0.02979 \\\hline
42 & 2, 128, 64, 16, 1 & tanh & AdamW & 60 & 0.08812 & 0.04844 \\\hline
43 & 2, 128, 64, 16, 1 & tanh & RMSProp & 30 & 0.05864 & 0.02979 \\\hline
44 & 2, 128, 64, 16, 1 & tanh & RMSProp & 60 & 0.08439 & 0.02979 \\\hline
45 & 2, 128, 64, 16, 1 & swish & AdamW & 30 & 0.05757 & 0.04805 \\\hline
46 & 2, 128, 64, 16, 1 & swish & AdamW & 60 & 0.07007 & 0.11682 \\\hline
47 & 2, 128, 64, 16, 1 & swish & RMSProp & 30 & 0.04901 & 0.03452 \\\hline
48 & 2, 128, 64, 16, 1 & swish & RMSProp & 60 & 0.05360 & 0.03311 \\\hline
49 & 2, 64, 128, 64, 32, 16, 1 & relu & AdamW & 30 & 0.04022 & 0.02865 \\\hline
50 & 2, 64, 128, 64, 32, 16, 1 & relu & AdamW & 60 & 0.04647 & 0.02873 \\\hline
51 & 2, 64, 128, 64, 32, 16, 1 & relu & RMSProp & 30 & 0.04108 & 0.03032 \\\hline
52 & 2, 64, 128, 64, 32, 16, 1 & relu & RMSProp & 60 & 0.03754 & 0.03280 \\\hline
53 & 2, 64, 128, 64, 32, 16, 1 & tanh & AdamW & 30 & 0.06183 & 0.02966 \\\hline
54 & 2, 64, 128, 64, 32, 16, 1 & tanh & AdamW & 60 & 0.06758 & 0.02967 \\\hline
55 & 2, 64, 128, 64, 32, 16, 1 & tanh & RMSProp & 30 & 0.05917 & 0.02989 \\\hline
56 & 2, 64, 128, 64, 32, 16, 1 & tanh & RMSProp & 60 & 0.06266 & 0.02970 \\\hline
57 & 2, 64, 128, 64, 32, 16, 1 & swish & AdamW & 30 & 0.04732 & 0.03786 \\\hline
58 & 2, 64, 128, 64, 32, 16, 1 & swish & AdamW & 60 & 0.04928 & 0.03862 \\\hline
59 & 2, 64, 128, 64, 32, 16, 1 & swish & RMSProp & 30 & 0.04627 & 0.03786 \\\hline
60 & 2, 64, 128, 64, 32, 16, 1 & swish & RMSProp & 60 & 0.04451 & 0.04630 \\\hline
61 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & relu & AdamW & 30 & 0.03343 & 0.04498 \\\hline
62 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & relu & AdamW & 60 & 0.03981 & 0.02833 \\ \hline
63 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & relu & RMSProp & 30 & 0.03658 & 0.03185 \\ \hline
64 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & relu & RMSProp & 60 & 0.03583 & 0.03020 \\ \hline
65 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & tanh & AdamW & 30 & 0.08163 & 0.02985 \\ \hline
66 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & tanh & AdamW & 60 & 0.10231 & 0.02944 \\ \hline
67 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & tanh & RMSProp & 30 & 0.08671 & 0.02683 \\ \hline
68 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & tanh & RMSProp & 60 & 0.09466 & 0.02790 \\ \hline
69 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & swish & AdamW & 30 & 0.04235 & 0.03622 \\ \hline
70 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & swish & AdamW & 60 & 0.04735 & 0.04101 \\ \hline
71 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & swish & RMSProp & 30 & 0.03905 & 0.30271 \\ \hline
72 & 2, 128, 256, 32, 64, 32, 16, 8, 1 & swish & RMSProp & 60 & 0.03591 & 0.08891 \\
\hline
\end{array}
\end{aligned}$$"

# ╔═╡ b05737c5-1fbc-4373-9bdc-0f12771ad675
md"De estos es la red 67 la que tiene menor error de predicción, tengamos en cuenta que esta exploración fue durante 1000 épocas únicamente para hacerlo más rápido.
Veamos los resultados con esta red."

# ╔═╡ adf4105c-c396-4cfd-831a-aeb00943b837
md"* Veamos el loss"

# ╔═╡ b6e7a9be-c319-4771-a4a8-d02514dc3f8a
PlutoUI.Resource("https://imgur.com/qQUekQq.png")

# ╔═╡ 46c38871-9c51-4b9b-8b74-67a2de56f174
md"Mismo caso que lo anteriror, sin embargo esta vez hay mucha mas diferencia entre el loss de entrenamiento y el de predicción."

# ╔═╡ 90e36bb3-5e07-43dd-ab17-d007dda37453
md"* Veamos las señales utilizadas"

# ╔═╡ d8060b67-fb20-4331-becf-81ff5e954dee
PlutoUI.Resource("https://imgur.com/aGK9KeG.png")

# ╔═╡ 4e947fec-7da7-4089-9c13-d63796f4d2a7
md"* Veamos las predicciones"

# ╔═╡ a61a2ca2-6a0c-4564-837b-dfe01a676878
PlutoUI.Resource("https://imgur.com/qmKl741.png")

# ╔═╡ ba91305c-d7a3-4669-95f2-086d3e443eb5
PlutoUI.Resource("https://imgur.com/8XQgD6q.png")

# ╔═╡ f11c45bb-323c-46b9-8b65-50de913bb888
PlutoUI.Resource("https://imgur.com/0PXPYnD.png")

# ╔═╡ 6be239a0-9c7b-4b70-93a0-d607679d004e
PlutoUI.Resource("https://imgur.com/4hCMq7V.png")

# ╔═╡ e6f7468f-7d3f-42ec-99c9-faa5c95b8ded
md"Mismo que antes, las señales se amontonan, no logran predecir con generalidad e incluso en el entrenamiento es malo"

# ╔═╡ 328f18a0-6703-4220-bdab-a992c9c5e8e9
begin
	# Función que crea el modelo de la red neuronal que va a estar dentro de la ODE
	function create_model(layers::Vector{Int}, activation)
	    """Función que crea el modelo de la red neuronal que va a estar dentro de la ODE
	
	    Parameters:
	        layers::Vector{Int} : Vector con las capas de la red neuronal
	        activation : Función de activación de la red neuronal para todas las capas
	
	    Returns:
	        function Chain : Red neuronal creada
	    """
	    # Creamos la red neuronal con las capas y la activación que nos pasan
	    activations = [activation for i in 1:length(layers) - 2]
	    # Regresamos la red neuronal
	    return Chain([Dense(layers[i], layers[i+1], activations[i]) for i in 1:length(layers) - 2]..., Dense(layers[end-1], layers[end]))
	end
	
	###################################################################################
	
	# Función que entrena la NODE con mini-batchs
	function Train_Neural_ODE(nn, U0, extra_parameters ,num_epochs, train_loader, opt, eta, Signals, Signals_forecast, t, tforecast)
	    """Función que entrena la NODE con mini-batchs
	    Parameters:
	        nn : Red neuronal que va a estar dentro de la ODE
	        U0 : Condiciones iniciales de la ODE
	        extra_parameters : Parámetros extra que se le pasan a la red neuronal
	        num_epochs : Número de epochs que se va a entrenar la red
	        train_loader : DataLoader con los datos de entrenamiento
	        opt : Optimizador que se va a utilizar
	        eta : Learning rate
	        Signals : datos de las señales que se van a utilizar para entrenar la red
	        Signals_forecast : datos de las señales que se buscan predecir
	        t : Tiempos de las señales recortado que se usa para entrenar la red
	        tforecast : Tiempos de las señales que se van a predecir
	    Returns:
	        loss : Arreglo con el loss de la red neuronal
	        p : Parámetros de la red neuronal
	        loss_forecast : Arreglo con el loss de la red neuronal para los datos de las señales que se predicen
	    """
	    # Tiempo sobre el cual resolver
	    tspan = (0f0, 1f0)
	
	    # Parametrizar indices para poder acceder a los parametros extras de la red neuronal durante el entrenamiento y predicción
	    f(x,p) = round(Int, x * (length(p) - 1)) + 1
	
	    # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial
	    p, re = Flux.destructure(nn) 
	
	    # Optimizardor que se va a utilizar
	    opt = opt(eta)
	
	    # Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
	    function predict_NeuralODE(u0, parametros, time_batch)
	        # dSdt = NN(S, parametros_extra) 
	        function dSdt(u, p, t; parametros_extra = parametros)
	            # Selecciona el indice de los parametros extra en el tiempo t
	            indx = f(t, parametros)
	            # Selecciona los parametros extra en el tiempo t
	            parametros_actuales = parametros[indx]
	            # Concatena los el valor de S(t) con los parametros extra en el tiempo t
	            entrada_red = vcat(u, parametros_actuales)
	            # Regresa la salida de la red neuronal reconstruida con los parámetros p y esta vez con los parametros extra en el tiempo t
	            return re(p)(entrada_red) 
	        end
	
	        # Definimos el problema de la ODE
	        prob = ODEProblem(dSdt, u0, tspan)
	
	        # Resolvemos la ODE y la devolvemos
	        return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-7, abstol = 1e-7))
	    end
	
	    # Función que predice las señales para un conjunto de condiciones iniciales
	    function Predict_Singals(U0, parametros_extra, time_batch)
	        Predicted_Signals = zeros(size(time_batch))
	        # Iteramos sobre las condiciones iniciales y las derivadas de las señales
	        for i in 1:length(U0)
	            u0 = Float32[U0[i]]
	            predicted_signal = predict_NeuralODE(u0, parametros_extra[:, i], time_batch)[1, :]
	            Predicted_Signals = hcat(Predicted_Signals, predicted_signal)
	        end
	        Predicted_Signals[:,2:end]
	    end
	
	    # Función de loss que vamos a minimizar, recibe un batch de señales y un batch de tiempos
	    function loss_node(batch, time_batch)
	        y = Predict_Singals(U0, extra_parameters, time_batch)
	        return Flux.mse(y, batch')
	    end
	    
	    # Función de callback para guardar el loss cada epoch
	    global iter = 0
	    loss = []
	    loss_forecast = []
	    callback = function ()
	        global iter += 1
	        if iter % (length(train_loader)) == 0
	            epoch = Int(iter / length(train_loader))
	            actual_loss = loss_node(Signals, t)
	            forecast_loss = loss_node(Signals_forecast, tforecast)
	            # println("Epoch = $epoch || Loss: $actual_loss || Loss Forecast: $forecast_loss")
	            push!(loss, actual_loss)
	            push!(loss_forecast, forecast_loss)
	        end
	        return false
	    end
	
	    # Entrenamos la red neuronal con mini-batchs
	    Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, num_epochs), opt, cb = callback)
	    
	    # Devolvemos el loss final y los parametros de la red neuronal
	    return loss, p, loss_forecast
	
	end;
end

# ╔═╡ c28428eb-0543-4ffb-b63a-9e4d376a1090
# Función que entrena la NODE con mini-batchs
function Train_Neural_ODE(nn, U0, extra_parameters ,num_epochs, train_loader, opt, eta, Signals, Signals_forecast, t, tforecast, lamb = 1.0)
    # Tiempo sobre el cual resolver
    tspan = (0f0, 1f0)

    # Parametrizar indices para poder acceder a los parametros extras de la red neuronal
    f(x,p) = round(Int, x * (length(p) - 1)) + 1
    
    p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

    # Optimizardor
    opt = opt(eta)

    # Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
    function predict_NeuralODE(u0, parametros, time_batch)
        # dSdt = NN(S, parametros_extra) 
        function dSdt(u, p, t; parametros_extra = parametros)
            indx = f(t, parametros) 
            parametros_actuales = parametros[indx] # Selecciona los parametros extra en el tiempo t
            entrada_red = vcat(u, parametros_actuales) # Concatena los el valor de S(t) con los parametros extra en el tiempo t
            return re(p)(entrada_red) # Regresa la salida de la red neuronal re creada con los parámetros p
        end

        prob = ODEProblem(dSdt, u0, tspan)

        return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-7, abstol = 1e-7)) # Regresa la solución de la ODE
    end

    # Función que predice las señales para un conjunto de condiciones iniciales
    function Predict_Singals(U0, parametros_extra, time_batch)
        Predicted_Signals = zeros(size(time_batch))
        for i in 1:length(U0)
            u0 = Float32[U0[i]]
            predicted_signal = predict_NeuralODE(u0, parametros_extra[:, i], time_batch)[1, :]
            Predicted_Signals = hcat(Predicted_Signals, predicted_signal)
        end    
        Predicted_Signals[:,2:end]
    end

    # Función de penalización para tratar de mantener la señal monotonamente decrecente
    function penalization_term(time_batch,y)
        pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
        return pen
    end

    # Función de pérdida que vamos a minimizar, recibe un batch de señales y un batch de tiempos
    function loss_node(batch, time_batch)
        y = Predict_Singals(U0, extra_parameters, time_batch)
        y_forecasted = Predict_Singals(U0, extra_parameters, tforecast)
        return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y) + penalization_term(tforecast, y_forecasted))
    end
    
    # Función de callback para guardar el loss cada epoch
    global iter = 0
    loss = []
    loss_forecast = []
    callback = function ()
        global iter += 1
        if iter % (length(train_loader)) == 0
            epoch = Int(iter / length(train_loader))
            actual_loss = loss_node(Signals, t)
            forecast_loss = loss_node(Signals_forecast, tforecast)
            println("Epoch = $epoch || Loss: $actual_loss || Loss Forecast: $forecast_loss")
            push!(loss, actual_loss)
            push!(loss_forecast, forecast_loss)
        end
        return false
    end

    # Entrenamos la red neuronal con mini-batchs
    Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, num_epochs), opt, cb = callback)
    
    # Devolvemos el loss final y los parametros de la red neuronal
    return loss, p, loss_forecast

end

# ╔═╡ ff4826e3-9667-4db6-9a6a-70a67bc7bd90
function main()
    # Arquitecturas que vamos a utilizar
    architectures = [
        [[2, 16, 16, 1], relu], # Dos capas ocultas
        [[2, 16, 16, 1], tanh], # Misma con activación tanh
        [[2, 16, 16, 1], swish], # Misma con activación swish
        
        [[2, 32, 64, 16, 1], relu], # Tres capas ocultas
        [[2, 32, 64, 16, 1], tanh], # Misma con activación tanh
        [[2, 32, 64, 16, 1], swish], # Misma con activación swish

        [[2, 128, 64, 16, 1], relu], # Tres capas ocultas con mas neuronas
        [[2, 128, 64, 16, 1], tanh], # Misma con activación tanh
        [[2, 128, 64, 16, 1], swish], # Misma con activación swish
        
        [[2, 64, 128, 64, 32, 16, 1], relu], # Cinco capas ocultas
        [[2, 64, 128, 64, 32, 16, 1], tanh], # Misma con activación tanh
        [[2, 64, 128, 64, 32, 16, 1], swish], # Misma con activación swish
        ]

    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [AdamW]]

    # Numero de mini-batchs que vamos a utilizar 
    batchs_size = 60 # En este caso cada bath es una señal completa de entrenamiento

    # Vector de configuraciones que vamos a utilizar

    configuraciones = []

    for arch in architectures
        for opt in optimizers
            for batch_size in batchs_size
                push!(configuraciones, (arch, opt, batch_size))
            end
        end
    end

    path_read = "/home/juan.morales/datos_PCA"
    # path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

    t_short = collect(range(0, 0.1, length = 1000))
    t_long = collect(range(0.1, 1, length = 100))
        
    # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
    muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
    muestreo_largo = 4

    # Esto da 75 tiempos 50 puntos desde 0 a 0.1 y 25 puntos desde 0.1 a 1
    t_short = t_short[1:muestreo_corto:end]
    t_long = t_long[1:muestreo_largo:end]
    
    t = vcat(t_short, t_long)

    # Tomamos 6 sigmas y 10 tamaños de compartimientos para cada sigma o sea 60 señales
    sampled_sigmas = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    lcm_range = 1:25:250
    
    # Obtenemos las señales representativas para un conjunto de sigmas y lcms
    Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

    # A partir de que indice de tiempo vamos a tomar predicciones
    # Los anteriroes tiempos se usan para entrenar la NODE
    idx_forecast = 61

    # Tiempos de entrenamiento y de predicción
    tforecast = t[idx_forecast:end]
    ttrain = t[1:idx_forecast-1]

    # Señaes de entrenamiento y de predicción
    Signals_train = Signals_rep[:,1:idx_forecast-1]
    Signals_valid = Signals_rep[:,idx_forecast:end]

    # Derivadas de las señales de entrenamiento y de predicción
    Signals_derivadas_train = Signals_rep_derivadas[1:idx_forecast-1,:]
    Signals_derivadas_valid = Signals_rep_derivadas[idx_forecast:end,:]

    # Tomamos un learning rate de 0.001
    eta = 1e-4

    # Vamos a tomar 2000 épocas para entrenar todas las arquitecturas
    epochs = 2000

    # Todas las señales tienen la misma condición inicial U0 = 1
    U0 = ones32(size(Signals_rep)[1])

    # Para el entrenamiento en el cluster vamos iterando sobre las configuraciones y guardamos los resultados en archivos csv
    # architecture, opt, batch_size = configuraciones[1]
    architecture, opt, batch_size = configuraciones[parse(Int128,ARGS[1])]

    layers = architecture[1]
    activation = architecture[2]

    if activation == tanh
        activation_string = "tanh"
    elseif activation == relu
        activation_string = "relu"
    elseif activation == swish
        activation_string = "swish"
    end

    if opt == AdamW
        opt_string = "AdamW"
    elseif opt == RMSProp
        opt_string = "RMSProp"
    end

    # Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
    train_loader = Flux.Data.DataLoader((Signals_train, ttrain), batchsize = batch_size)

    # Vamos a crear el modelo de la red neuronal que va a estar dentro de la ODE
    nn = create_model(layers, activation)
    
    # Parámetro de penalización
    lambd = 0.1

    # Entrenamos una NODE con mini-batchs para cada arquitectura, optimizador y tamaño de mini-batch y guardamos el loss y los parametros de la red neuronal
    architecture_loss, theta, loss_forecast = Train_Neural_ODE(nn, U0, Signals_derivadas_train, epochs, train_loader, opt, eta, Signals_train, Signals_valid, ttrain, tforecast, lambd)

    # println("Arquitectura: $architecture", " || Optimizador: $opt", " || Tamaño de mini-batch: $batch_size", " || Loss: $(architecture_loss[end])", " || Loss Forecast: $(loss_forecast[end])")

    # actual_id = 1
    actual_id = parse(Int128,ARGS[1])
    actual_layer = string(layers)
    actual_activation = activation_string
    actual_optimizer = opt_string
    actual_loss_final_train = architecture_loss[end]
    actual_loss_final_forecast = loss_forecast[end]
    actual_batch_size = batch_size

    # Guardamos los resultados en un archivo csv

    df_results_total = DataFrame(ID = actual_id, Arquitectura = actual_layer, Activación = actual_activation, Optimizador = actual_optimizer, Batch_Size = actual_batch_size, Loss_Final_Entrenamiento = actual_loss_final_train, Loss_Final_Predicción = actual_loss_final_forecast)

    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    CSV.write("/home/juan.morales/Representative_mini_trainingNODE/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    

    # Guardamos los loss y los parametros de la red neuronal en archivos csv

    Loss_Matrix = zeros((length(architecture_loss), 2))

    for i in 1:length(architecture_loss)
        Loss_Matrix[i,1] = architecture_loss[i]
        Loss_Matrix[i,2] = loss_forecast[i]
    end

    df_losses = DataFrame(Loss_Matrix, :auto)

    rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
    rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Losses/$(actual_id)_losses.csv", df_losses)
    # CSV.write("/home/juan.morales/Representative_mini_trainingNODE/Losses/$(actual_id)_losses.csv", df_losses)
    CSV.write("/home/juan.morales/Representative_mini_trainingNODE/Losses/11_losses.csv", df_losses)

    df_theta = DataFrame(reshape(theta, length(theta), 1), :auto)
    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Parameters/$(actual_id)_Parameters.csv", df_losses)
    CSV.write("/home/juan.morales/Representative_mini_trainingNODE/Parameters/$(actual_id)_Parameters.csv", df_theta)

end;

# ╔═╡ 292a1193-e6e6-49fa-9785-2ffabadb87b5
# Función que entrena la NODE con mini-batchs
function Train_Neural_ODE(nn, U0, extra_parameters ,num_epochs, train_loader, opt, eta, Signals, Signals_forecast, t, tforecast, lamb = 1.0)
    # Tiempo sobre el cual resolver
    tspan = (0f0, 1f0)

    # Parametrizar indices para poder acceder a los parametros extras de la red neuronal
    f(x,p) = round(Int, x * (length(p) - 1)) + 1
    
    p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

    # Optimizardor
    opt = opt(eta)

    # Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
    function predict_NeuralODE(u0, parametros, time_batch)
        # dSdt = NN(S, parametros_extra) 
        function dSdt(u, p, t; parametros_extra = parametros)
            indx = f(t, parametros) 
            parametros_actuales = parametros[indx] # Selecciona los parametros extra en el tiempo t
            entrada_red = vcat(u, parametros_actuales) # Concatena los el valor de S(t) con los parametros extra en el tiempo t
            return re(p)(entrada_red) # Regresa la salida de la red neuronal re creada con los parámetros p
        end

        prob = ODEProblem(dSdt, u0, tspan)

        return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-7, abstol = 1e-7)) # Regresa la solución de la ODE
    end

    # Función que predice las señales para un conjunto de condiciones iniciales
    function Predict_Singals(U0, parametros_extra, time_batch)
        Predicted_Signals = zeros(size(time_batch))
        for i in 1:length(U0)
            u0 = Float32[U0[i]]
            predicted_signal = predict_NeuralODE(u0, parametros_extra[:, i], time_batch)[1, :]
            Predicted_Signals = hcat(Predicted_Signals, predicted_signal)
        end    
        Predicted_Signals[:,2:end]
    end

    # Función de penalización para tratar de mantener la señal monotonamente decrecente
    function penalization_term(time_batch,y)
        pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
        return pen
    end

    # Función de pérdida que vamos a minimizar, recibe un batch de señales y un batch de tiempos
    function loss_node(batch, time_batch)
        y = Predict_Singals(U0, extra_parameters, time_batch)
        y_forecasted = Predict_Singals(U0, extra_parameters, tforecast)
        return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y) + penalization_term(tforecast, y_forecasted))
    end
    
    # Función de callback para guardar el loss cada epoch
    global iter = 0
    loss = []
    loss_forecast = []
    callback = function ()
        global iter += 1
        if iter % (length(train_loader)) == 0
            epoch = Int(iter / length(train_loader))
            actual_loss = loss_node(Signals, t)
            forecast_loss = loss_node(Signals_forecast, tforecast)
            println("Epoch = $epoch || Loss: $actual_loss || Loss Forecast: $forecast_loss")
            push!(loss, actual_loss)
            push!(loss_forecast, forecast_loss)
        end
        return false
    end

    # Entrenamos la red neuronal con mini-batchs
    Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, num_epochs), opt, cb = callback)
    
    # Devolvemos el loss final y los parametros de la red neuronal
    return loss, p, loss_forecast

end

# ╔═╡ 37e2363d-ea23-4b21-a8aa-bebbb87d9600
function main()
    # Arquitecturas que vamos a utilizar
    architectures = [
        [[2, 16, 16, 1], relu], # Dos capas ocultas
        [[2, 16, 16, 1], tanh], # Misma con activación tanh
        [[2, 16, 16, 1], swish], # Misma con activación swish
        
        [[2, 32, 64, 16, 1], relu], # Tres capas ocultas
        [[2, 32, 64, 16, 1], tanh], # Misma con activación tanh
        [[2, 32, 64, 16, 1], swish], # Misma con activación swish

        [[2, 128, 64, 16, 1], relu], # Tres capas ocultas con mas neuronas
        [[2, 128, 64, 16, 1], tanh], # Misma con activación tanh
        [[2, 128, 64, 16, 1], swish], # Misma con activación swish
        
        [[2, 64, 128, 64, 32, 16, 1], relu], # Cinco capas ocultas
        [[2, 64, 128, 64, 32, 16, 1], tanh], # Misma con activación tanh
        [[2, 64, 128, 64, 32, 16, 1], swish], # Misma con activación swish
        ]

    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [AdamW]]

    # Numero de mini-batchs que vamos a utilizar 
    batchs_size = 60 # En este caso cada bath es una señal completa de entrenamiento

    # Vector de configuraciones que vamos a utilizar

    configuraciones = []

    for arch in architectures
        for opt in optimizers
            for batch_size in batchs_size
                push!(configuraciones, (arch, opt, batch_size))
            end
        end
    end

    path_read = "/home/juan.morales/datos_PCA"
    # path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

    t_short = collect(range(0, 0.1, length = 1000))
    t_long = collect(range(0.1, 1, length = 100))
        
    # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
    muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
    muestreo_largo = 4

    # Esto da 75 tiempos 50 puntos desde 0 a 0.1 y 25 puntos desde 0.1 a 1
    t_short = t_short[1:muestreo_corto:end]
    t_long = t_long[1:muestreo_largo:end]
    
    t = vcat(t_short, t_long)

    # Tomamos 6 sigmas y 10 tamaños de compartimientos para cada sigma o sea 60 señales
    sampled_sigmas = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    lcm_range = 1:25:250
    
    # Obtenemos las señales representativas para un conjunto de sigmas y lcms
    Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

    # A partir de que indice de tiempo vamos a tomar predicciones
    # Los anteriroes tiempos se usan para entrenar la NODE
    idx_forecast = 61

    # Tiempos de entrenamiento y de predicción
    tforecast = t[idx_forecast:end]
    ttrain = t[1:idx_forecast-1]

    # Señaes de entrenamiento y de predicción
    Signals_train = Signals_rep[:,1:idx_forecast-1]
    Signals_valid = Signals_rep[:,idx_forecast:end]

    # Derivadas de las señales de entrenamiento y de predicción
    Signals_derivadas_train = Signals_rep_derivadas[1:idx_forecast-1,:]
    Signals_derivadas_valid = Signals_rep_derivadas[idx_forecast:end,:]

    # Tomamos un learning rate de 0.001
    eta = 1e-4

    # Vamos a tomar 2000 épocas para entrenar todas las arquitecturas
    epochs = 2000

    # Todas las señales tienen la misma condición inicial U0 = 1
    U0 = ones32(size(Signals_rep)[1])

    # Para el entrenamiento en el cluster vamos iterando sobre las configuraciones y guardamos los resultados en archivos csv
    # architecture, opt, batch_size = configuraciones[1]
    architecture, opt, batch_size = configuraciones[parse(Int128,ARGS[1])]

    layers = architecture[1]
    activation = architecture[2]

    if activation == tanh
        activation_string = "tanh"
    elseif activation == relu
        activation_string = "relu"
    elseif activation == swish
        activation_string = "swish"
    end

    if opt == AdamW
        opt_string = "AdamW"
    elseif opt == RMSProp
        opt_string = "RMSProp"
    end

    # Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
    train_loader = Flux.Data.DataLoader((Signals_train, ttrain), batchsize = batch_size)

    # Vamos a crear el modelo de la red neuronal que va a estar dentro de la ODE
    nn = create_model(layers, activation)
    
    # Parámetro de penalización
    lambd = 0.1

    # Entrenamos una NODE con mini-batchs para cada arquitectura, optimizador y tamaño de mini-batch y guardamos el loss y los parametros de la red neuronal
    architecture_loss, theta, loss_forecast = Train_Neural_ODE(nn, U0, Signals_derivadas_train, epochs, train_loader, opt, eta, Signals_train, Signals_valid, ttrain, tforecast, lambd)

    # println("Arquitectura: $architecture", " || Optimizador: $opt", " || Tamaño de mini-batch: $batch_size", " || Loss: $(architecture_loss[end])", " || Loss Forecast: $(loss_forecast[end])")

    # actual_id = 1
    actual_id = parse(Int128,ARGS[1])
    actual_layer = string(layers)
    actual_activation = activation_string
    actual_optimizer = opt_string
    actual_loss_final_train = architecture_loss[end]
    actual_loss_final_forecast = loss_forecast[end]
    actual_batch_size = batch_size

    # Guardamos los resultados en un archivo csv

    df_results_total = DataFrame(ID = actual_id, Arquitectura = actual_layer, Activación = actual_activation, Optimizador = actual_optimizer, Batch_Size = actual_batch_size, Loss_Final_Entrenamiento = actual_loss_final_train, Loss_Final_Predicción = actual_loss_final_forecast)

    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    CSV.write("/home/juan.morales/Representative_mini_trainingNODE/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    

    # Guardamos los loss y los parametros de la red neuronal en archivos csv

    Loss_Matrix = zeros((length(architecture_loss), 2))

    for i in 1:length(architecture_loss)
        Loss_Matrix[i,1] = architecture_loss[i]
        Loss_Matrix[i,2] = loss_forecast[i]
    end

    df_losses = DataFrame(Loss_Matrix, :auto)

    rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
    rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Losses/$(actual_id)_losses.csv", df_losses)
    # CSV.write("/home/juan.morales/Representative_mini_trainingNODE/Losses/$(actual_id)_losses.csv", df_losses)
    CSV.write("/home/juan.morales/Representative_mini_trainingNODE/Losses/11_losses.csv", df_losses)

    df_theta = DataFrame(reshape(theta, length(theta), 1), :auto)
    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Parameters/$(actual_id)_Parameters.csv", df_losses)
    CSV.write("/home/juan.morales/Representative_mini_trainingNODE/Parameters/$(actual_id)_Parameters.csv", df_theta)

end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "acf8a7d70217bc877a8d448fd5475fd91c6dd480"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─41be4360-5c24-4b23-b25f-48a2b187cbc1
# ╟─d2d4ea40-a472-4d24-a66b-8be285fca558
# ╠═81f922dc-76d0-4a98-b691-b180aa911b26
# ╠═75189838-98d6-495a-8421-1f0ebbea6899
# ╟─f27f194d-c3c1-42d5-b80b-4309914fe3a3
# ╠═6b605ab7-f8e2-49aa-8b85-2a2989d677d1
# ╟─068dcddc-49cf-4696-a0f6-26a42d318a76
# ╠═d5bbf401-92ec-4475-a987-ccebc37eeee7
# ╟─4a057240-4179-457c-ae63-9972a23d81b0
# ╠═bbe34a66-12da-46a0-b62f-a9b110584afd
# ╟─bb42d2b8-9a84-419a-ad7f-f92fc9c8bb58
# ╠═328f18a0-6703-4220-bdab-a992c9c5e8e9
# ╟─9c694f8f-19ee-4554-a8a4-910c7470ad53
# ╠═c28428eb-0543-4ffb-b63a-9e4d376a1090
# ╟─7441f18e-f9c0-4ffb-af0f-6467eff7d489
# ╟─6f2ab7ee-149e-4f9d-a5bc-1fe1d737ed9f
# ╠═8f4d677d-248c-4187-be29-033b85fc7474
# ╟─a2e08efa-93f5-4022-a335-74b111c8fb00
# ╟─2d472dc9-7e40-4488-b13a-a736b09f96de
# ╠═ff4826e3-9667-4db6-9a6a-70a67bc7bd90
# ╟─1a3eb77e-740d-447e-9af2-d972ab2bd208
# ╟─6b934743-ce99-4f66-a1c7-cb71b321c3b5
# ╠═292a1193-e6e6-49fa-9785-2ffabadb87b5
# ╟─49f5148a-e552-4896-9e94-a58092e27826
# ╠═37e2363d-ea23-4b21-a8aa-bebbb87d9600
# ╟─8a3d2bc4-2301-4d8b-b43f-fe6fe0435408
# ╟─ed21550e-cb50-43ba-903d-061b5f69975a
# ╟─69376adb-64f8-4204-ad32-10039ac55594
# ╟─ff6806fb-edf0-4b96-b8b6-e6d1d978ca7f
# ╟─7c088c54-afd5-483f-ad52-830baf67ea85
# ╟─ab90dfea-d78f-4d94-a0dc-a12e0388667c
# ╟─e6d90d95-d580-4214-ba7d-755cb27da810
# ╟─2e20900c-696d-44fb-b638-fc53aac18d4e
# ╟─d35cbba2-e247-4469-b07a-125323f05076
# ╟─e3108149-acd3-4128-989d-e484882a3db6
# ╟─b982949f-3307-4225-aac2-96a93f386012
# ╟─d267aaa5-368b-4a7d-a683-9afdeeaa5ba0
# ╟─d9eeddf2-518c-47e5-b96a-4a3326237e2b
# ╟─68a7a366-65fb-4c04-b507-8a5084364859
# ╟─b05737c5-1fbc-4373-9bdc-0f12771ad675
# ╟─adf4105c-c396-4cfd-831a-aeb00943b837
# ╟─b6e7a9be-c319-4771-a4a8-d02514dc3f8a
# ╟─46c38871-9c51-4b9b-8b74-67a2de56f174
# ╟─90e36bb3-5e07-43dd-ab17-d007dda37453
# ╟─d8060b67-fb20-4331-becf-81ff5e954dee
# ╟─4e947fec-7da7-4089-9c13-d63796f4d2a7
# ╟─a61a2ca2-6a0c-4564-837b-dfe01a676878
# ╟─ba91305c-d7a3-4669-95f2-086d3e443eb5
# ╟─f11c45bb-323c-46b9-8b65-50de913bb888
# ╟─6be239a0-9c7b-4b70-93a0-d607679d004e
# ╟─e6f7468f-7d3f-42ec-99c9-faa5c95b8ded
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
