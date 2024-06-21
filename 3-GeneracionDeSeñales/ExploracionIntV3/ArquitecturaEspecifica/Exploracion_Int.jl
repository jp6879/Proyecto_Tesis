### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 322d766c-1576-43c2-b344-fb571e5bd775
import PlutoUI

# ╔═╡ cafda7f0-550f-40bd-8bad-2d82312a16db
md"# Exploración con parámetros continuos

* Una de las últimas estrategias que habíamos explorado fue la posibilidad de utilizar parámetros continuos. Por ejemplo, consideramos encontrar la ecuación de una recta que conectara los puntos inicial y final de la señal.

* Llevemos esta idea algo que le de muchisima mas información a la NeuralODE. Como sabíamos, esta red, para devolver una señal temporal, requiere únicamente un punto inicial $S(0)$ y los valores de la derivada numérica de la señal $S(t)$ que permiten distinguir entre diferentes señales.

* Agreguemos como parámetros una interpolación lineal entre 10 puntos de la señal $S(t)$ que servirán como entrada para la señal, así como también la interpolación lineal de la derivada numérica. Esto parámetros extra al ser continuos permiten que haya información de la señal para todo tiempo $t$ lo cual es útil para la NeuralODE.

* Otra de las cosas que cambiamos es el método de resolución de la ecuaciones diferenciales por Vern6 por la recomendación de ODE Solvers de Julia: _For high accuracy (~1e-8-1e-12), we recommend Vern6, Vern7, or Vern8 as efficient choices_


"

# ╔═╡ 6104761b-b805-48ca-b8eb-50af3874071b
##############################################################################################
# Parámetros fijos
# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de tiempos
N = 5000
time_sample_lenght = 1000

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 45

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lcs = range(l0, lf, length = N)

# Tenemos dos tipos de sampleos uno desde 0 a 0.1 que tiene 0.0001 de paso y otro de 0.1 a 1 que tiene 0.001 de paso
t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))
    
# El tiempo total es concatenar estos dos arreglos
t = vcat(t_short, t_long)

# Guardamos este tiempo total para poder hacer las predicciones
toriginal = vcat(t_short, t_long)

# Path desde donde se leen los datos
# path_read = "/home/juan.morales/datos_PCA"
path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

# Parametros que varían para cada señal
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
# Desviaciones estándar
sigmas = 0.01:0.01:1

##############################################################################################
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
end



# Función que obtiene las derivadas de las señales y las devuelve normalizadas
function get_signals_deriv_valid(tvalid, Signals_valid)
    Signals_derivadas_valid = zeros(size(Signals_valid))

    # Obtenemos las derivadas de las señales de validación
    for i in 1:size(Signals_valid)[1]
        Signals_derivadas_valid[i,:] = derivate_signals(tvalid,Signals_valid[i,:])
    end

    # La transponemos y la convertimos a Float32
    Signals_derivadas_valid = Float32.(Matrix(Signals_derivadas_valid'))

    # Normalizamos las derivadas
    for i in 1:size(Signals_valid)[1]
        Signals_derivadas_valid[:,i] = Signals_derivadas_valid[:,i] ./ maximum(abs.(Signals_derivadas_valid[:,i]))
    end

    return Signals_derivadas_valid
end


# Función que devuelve un interpolador lineal para un conjunto de señales y tiempos dados
# Las señales entran como un arreglo de arreglos y los tiempos como un arreglo
function get_interpolated(t, Signals)
    itp = []
    # Interpolamos las derivadas
    for i in 1:size(Signals)[2]
        push!(itp, Interpolations.linear_interpolation(t, Signals[:,i], extrapolation_bc=Line()))
    end
    return itp
end

##############################################################################################

t_short = Float32.(collect(range(0, 0.1, length = 1000)))
t_long = Float32.(collect(range(0.1, 1, length = 100)))
    
# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto =  25 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 5

# Esto da 60 tiempos 40 puntos desde 0 a 0.1 y 20 puntos desde 0.1 a 1
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]

t = vcat(t_short, t_long)

# Tomamos 1 sigmas y 12 tamaños de compartimientos para cada sigma como entrenamiento
# Probamos en un conjunto pequeño de test de 4 tamaños de compartimiento para el mísmo sigma
sampled_sigmas =  [1.0]
lcm_range = 1:50:600
lcm_test = 1:75:600

lcm_test = [l for l in collect(lcm_test) if l ∉ collect(lcm_range)]
# Lcms: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
# Lcms test: [1.25, 2.75, 4.25, 5.75]

# Obtenemos las señales representativas para un conjunto de sigmas y lcms
Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmas_test = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_test, muestreo_corto, muestreo_largo, t)

# ╔═╡ c2d35967-1c47-49cf-af60-8eb2c47cc881
md"Probemos nuevamente para un único valor de $\sigma = 1$. En este caso, para entrenar la red, utilizamos las señales que corresponden a los 12 valores de $l_{cm} = 0.5,~1.0,~1.5,~2.0,~2.5,~3.0,~3.5,~4.0,~4.5,~5.0,~5.5,~6.0$ $\mu$m. Para el conjunto de pruebas, utilizamos un conjunto pequeño de 4 señales con valores $l_{cm} = 1.25,~2.75,~4.25,~5.75$."

# ╔═╡ 32c9a071-20fc-49cd-95cc-b5945cf9b9f4
PlutoUI.Resource("https://imgur.com/B8Bl3dc.png")

# ╔═╡ 7ab06e65-25bf-4680-819f-34e0c5de2c60
md"Como mencionamos, vamos a utilizar 10 puntos para estos nuevos parámetros, y por lo tanto, para realizar el entrenamiento. La idea sería ingresar a la NeuralODE con el punto inicial $S(0)$ y dos interpolaciones (una de la propia señal y otra de la derivada) con estos 10 puntos que funcionan como parámetros adicionales."

# ╔═╡ 0d169ce8-9ae2-4c10-9afc-816bc065788f
# Numero de puntos para la predicción
n_valid = 10

# Paso para tomar los tiempos de entrenamiento y validación
step = floor(Int, length(t) / n_valid) + 1

tvalid = t[1:step:end]

# En la validación y en el train tenemos que tener el primer y último tiempo
tvalid = vcat(tvalid, t[end])

indexes_valid = [i for i in 1:length(t) if t[i] in tvalid]

Signals_valid = Signals_rep[:,indexes_valid]
Signals_test_puntos = Signals_test[:,indexes_valid]

# Obtenemos las derivadas de las señales de validación
Signals_derivadas_valid = get_signals_deriv_valid(tvalid, Signals_valid)
Signals_derivadas_test = get_signals_deriv_valid(tvalid, Signals_test_puntos)

# Obtenemos las interpolaciones de las señales y de las derivadas
itp_signals = get_interpolated(tvalid, Signals_valid')
itp_derivadas = get_interpolated(tvalid, Signals_derivadas_valid)

itp_signals_test = get_interpolated(tvalid, Signals_test_puntos')
itp_derivadas_test = get_interpolated(tvalid, Signals_derivadas_test)

extra_parameters = itp_signals
extra_parameters2 = itp_derivadas

extra_parameters_test = itp_signals_test
extra_parameters2_test = itp_derivadas_test

# ╔═╡ 76783e8a-9695-48a8-b73c-24f6e8b4adbc
PlutoUI.Resource("https://imgur.com/PYxYfYE.png")

# ╔═╡ 003518cd-8d72-4897-9ffb-011f66267f14
md"Una vez realizado esto, llevamos a cabo nuevamente una exploración de la arquitectura de la red, como en los casos anteriores. Sin embargo, en esta ocasión, la exploración fue limitada y se llevó a cabo únicamente durante 1000 épocas por la alta demanda del clúster estas últimas semanas."

# ╔═╡ 159125e0-5e7d-4e85-99d8-38cc8cb18681
md"

$$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossValid} \\
    \hline
    01 & [3, 32, 32, 16, 1] & \text{relu} & \text{AdamW} & 15 & 8.289 \times 10^{-5} & 9.626 \times 10^{-5} \\
    02 & [3, 32, 32, 16, 1] & \text{relu} & \text{RMSProp} & 15 & 0.00030103 & 0.00025301 \\
    03 & [3, 32, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.0065044 & 0.0071138 \\
    04 & [3, 32, 32, 16, 1] & \text{tanh\_fast} & \text{RMSProp} & 15 & 0.000715 & 0.00080253 \\
    05 & [3, 32, 32, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.00038909 & 0.00041496 \\
    06 & [3, 32, 32, 16, 1] & \text{swish} & \text{RMSProp} & 15 & 0.00061 & 0.00056087 \\
    07 & [3, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.00025692 & 0.00022344 \\
    08 & [3, 32, 64, 16, 1] & \text{relu} & \text{RMSProp} & 15 & 0.00030627 & 0.00030524 \\
    09 & [3, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.00053068 & 0.00059773 \\
    10 & [3, 32, 64, 16, 1] & \text{tanh\_fast} & \text{RMSProp} & 15 & 0.00047955 & 0.00052531 \\
    11 & [3, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.0017002 & 0.0018013 \\
    12 & [3, 32, 64, 16, 1] & \text{swish} & \text{RMSProp} & 15 & 0.00031886 & 0.00033819 \\
    13 & [3, 128, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 6.984 \times 10^{-5} & 8.971 \times 10^{-5} \\
    14 & [3, 128, 64, 16, 1] & \text{relu} & \text{RMSProp} & 15 & 0.00042358 & 0.00040798 \\
    15 & [3, 128, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.0052783 & 0.0057591 \\
    17 & [3, 128, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.00026186 & 0.0002796 \\
    18 & [3, 128, 64, 16, 1] & \text{swish} & \text{RMSProp} & 15 & 0.00071058 & 0.00069441 \\
    19 & [3, 32, 64, 32, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.00011696 & 8.766 \times 10^{-5} \\
    20 & [3, 32, 64, 32, 16, 1] & \text{relu} & \text{RMSProp} & 15 & 0.00034636 & 0.00034905 \\
    21 & [3, 32, 64, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.006295 & 0.0068984 \\
    22 & [3, 32, 64, 32, 16, 1] & \text{swish} & \text{AdamW} & 15 & 5.451 \times 10^{-5} & 4.775 \times 10^{-5} \\
    23 & [3, 32, 64, 32, 16, 1] & \text{swish} & \text{RMSProp} & 15 & 0.00032199 & 0.00025863 \\
    24 & [3, 16, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.00012233 & 0.00013834 \\
    25 & [3, 16, 32, 64, 16, 1] & \text{relu} & \text{RMSProp} & 15 & 0.00049582 & 0.00048687 \\
    26 & [3, 16, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.0066492 & 0.0072771 \\
    27 & [3, 16, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 2.88 \times 10^{-5} & 2.455 \times 10^{-5} \\
    28 & [3, 16, 32, 64, 16, 1] & \text{swish} & \text{RMSProp} & 15 & 0.00014387 & 0.00012604 \\
    29 & [3, 16, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.00032656 & 0.00034397 \\
    30 & [3, 16, 32, 64, 16, 1] & \text{relu} & \text{RMSProp} & 15 & 0.00033571 & 0.00035293 \\
    31 & [3, 16, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.0065869 & 0.0072085 \\
    32 & [3, 16, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.00016695 & 0.00016533 \\
    33 & [3, 16, 32, 64, 16, 1] & \text{swish} & \text{RMSProp} & 15 & 0.0001901 & 0.00019531 \\
    \hline
\end{array}
\end{aligned}$$

"

# ╔═╡ 3c73f87e-2934-428e-a48a-51971075aba9
md"Vemos que el Loss baja considerablemente con respecto a las exploraciones anteriores, seleccionamos en particular la arquitectura 27 que es la que menor loss tiene."

# ╔═╡ 29b1d974-20b1-4d4b-99c6-c4b421d761c4
# Todas las señales tienen la misma condición inicial U0 = 1
U0 = ones32(size(Signals_rep)[1])
U0_test = ones32(size(Signals_test)[1])

# id actual de la red
actual_id = 27

#Definimos el batch size
batch_size = 15

# Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
train_loader = Flux.Data.DataLoader((Signals_rep, t), batchsize = batch_size)

# Función de activación
activation = swish

# 27,"[3, 16, 32, 64, 16, 1]",swish,AdamW,15,2.8820188223710173e-5,2.455452888094309e-5

nn = Chain(Dense(3, 16, activation),
            Dense(16, 32, activation),
            Dense(32, 64, activation),
            Dense(64, 16, activation),
            Dense(16, 1)
            )

# Tomamos un learning rate de 0.001
η = 5e-3

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

# Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
function predict_NeuralODE(u0, parametros, parametros2, time_batch)
    # dSdt = NN(S, parametros_extra) 
    function dSdt(u, p, t; parametros_extra = parametros, parametros_extra2 = parametros2)
        parametros_actuales = parametros(t) # Selecciona los parametros extra en el tiempo t
        parametros_actuales_2 = parametros2(t)
        entrada_red = vcat(u, parametros_actuales, parametros_actuales_2) # Concatena los el valor de S(t) con los parametros extra en el tiempo t
        return re(p)(entrada_red) # Regresa la salida de la red neuronal re creada con los parámetros p
    end

    prob = ODEProblem(dSdt, u0, tspan)

    return Array(solve(prob, Vern6(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch)) # Regresa la solución de la ODE
end

# Función que predice las señales para un conjunto de condiciones iniciales y parámetros extra
function Predict_Singals(U0, parametros_extra, parametros_extra2, time_batch)
    Predicted_Signals = zeros(size(time_batch))
    for i in 1:length(U0)
        u0 = Float32[U0[i]]
        if length(parametros_extra) == length(U0)
            predicted_signal = predict_NeuralODE(u0, parametros_extra[i], parametros_extra2[i], time_batch)[1, :]
        else
            predicted_signal = predict_NeuralODE(u0, parametros_extra, parametros_extra2, time_batch)[1, :]
        end
        Predicted_Signals = hcat(Predicted_Signals, predicted_signal)
    end    
    Predicted_Signals[:,2:end]
end

# Función que calcula el loss de la red neuronal para un batch de datos y sus respectivos tiempos

# Penalization term
function penalization_term(time_batch,y)
    """
    Función de penalización para tratar de mantener la señal monotonamente decrecente
    En caso que la señal no sea monotonamente decreciente, la penalización es la suma de las diferencias positivas entre los tiempos
    """
    pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
end

function loss_node(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0, extra_parameters, extra_parameters2, time_batch)
    return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y))
end

function loss_valid(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0_test, extra_parameters_test, extra_parameters2_test, time_batch)
    return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y))
end

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, epochs), opt)

# ╔═╡ 7d1b873c-e0c1-4f62-beaa-320ad932dfa0
md"Observamos el loss durante el entrenamiento"

# ╔═╡ 25f727c6-34e9-4986-8207-96480a24c255
PlutoUI.Resource("https://imgur.com/JfZmrlP.png")

# ╔═╡ 708d13ef-7751-4ae0-bf85-039ad1dfa32f
md"El Loss de predicción se refiere al error calculado utilizando el conjunto de datos de prueba o validación, ya que no separé en un conjunto adicional. Vemos que ambos siguen disminuyendo en conjunto por lo que no se está llegando a un overfitting."

# ╔═╡ 807af1dd-54cd-49f9-aead-57fd7d9119d4
md"Veamos ahora las predicciones sobre el conjunto de entrenamiento."

# ╔═╡ 178e2c63-33e8-445b-8ad0-b7a80ccbd5b0
PlutoUI.Resource("https://imgur.com/ucOrM00.png")

# ╔═╡ d42752e8-38de-4b70-8e5d-b8ed6e1e23da
md"Lo hace bastante bien. Además, aclaramos que la predicción se realizó para la misma cantidad de puntos que utiliza la red neuronal anteriormente entrenada para obtener el $\sigma$ y $l_{cm}$. Por lo tanto, esperaríamos que el resultado sea bueno.

Veamos un zoom en la región $t \in (0,~0.1)$.

"

# ╔═╡ 5fd2127b-4db7-420e-a704-17b62381514c
PlutoUI.Resource("https://imgur.com/PDLHx7H.png")

# ╔═╡ b25289b7-3661-4c04-8324-98ae21cd34ad
md"En esta parte encontramos pequeños defectos en la predicción pero sigue siendo buena."

# ╔═╡ 4d7bf43e-dc46-41a1-b014-6f3c6b7bb73c
md"Veamos ahora las predicciones realizadas sobre el conjunto de Test/Validación."

# ╔═╡ eaeed495-970b-4364-989b-eef70e1a8ba1
PlutoUI.Resource("https://imgur.com/CPq7t1v.png")

# ╔═╡ 46c58cf8-825f-4264-aed6-71db8ff4913c
md"Nuevamente la NeuralODE logra predecir correctamente veamos en la zona $t \in (0,~0.1)$."

# ╔═╡ e4cf90b5-d139-4022-914c-b19bb03b2dc1
PlutoUI.Resource("https://imgur.com/og1pxqs.png")

# ╔═╡ 58702e1b-c78d-40bf-8fe5-288f8598534a
md"Como esta es una NeuralODE se puede predecir la señal para cualquier tiempo, en particular se puede realizar una extrapolación. Veamos un ejemplo de esto con las señales entrenadas extrapolando hasta $t = 2$."

# ╔═╡ 122e341c-e6db-4ed5-9bae-9fb6af9b033f
PlutoUI.Resource("https://imgur.com/kjUEfzz.png")

# ╔═╡ b8a49a8a-ea19-4f62-8289-2144c5b7086f
md"Resumiendo, nuestra red intenta realizar una interpolación mejor que la lineal de los puntos. Sin embargo, a veces aparecen quiebres porque la red intenta ajustarse a los puntos de las interpolaciones de los parámetros extra. 

El hecho de haber elegido 10 puntos para realizar las predicciones se debe a que proporcionan mucha información sobre la señal en sí. Sin embargo, probemos a ver qué ocurre cuando pasamos estos siguientes 5 puntos a la NeuralODE y entrenamos la red un poco más durante 200 épocas."

# ╔═╡ 8eee656b-a00a-4252-9017-28f5efce57fa
PlutoUI.Resource("https://imgur.com/HvaKWtG.png")

# ╔═╡ 3657f604-4f91-4f4d-9db2-7ae675401999
md"El procedimiento es el mísmo que antes, calculamos la derivada numérica con estos puntos y luego una interpolación lineal tanto para la señal como para la derivada de la señal. Con esto se reentrenó la red en un entrenamiento 'rápido' de 200 épocas.

Veamos el Loss que obtuvimos en función de las épocas.
"

# ╔═╡ a9f783e3-f81e-4df9-9e27-c9e66c8c185d
PlutoUI.Resource("https://imgur.com/tDvOfNc.png")

# ╔═╡ 3000a762-d315-42fa-8854-72529df8bfad
md"En este caso, no es tan claro que vaya a seguir disminuyendo, pero recordemos que esta es una red en particular. Quizás se podría encontrar una que lo haga mejor para estos inputs.

Veamos ahora las predicciones con estos 5 puntos."

# ╔═╡ 3652d7b3-6b16-4000-8425-c6bdd90ce3d2
PlutoUI.Resource("https://imgur.com/Egd5otM.png")

# ╔═╡ 44c6de9d-609a-4107-bb03-11c97b7286a4
md"Tambibién veamos un zoom."

# ╔═╡ bfd00e81-d6d2-4bfa-95dd-e89a0d68df69
PlutoUI.Resource("https://imgur.com/NZUfCqw.png")

# ╔═╡ f74ab92f-b34c-4a43-bff9-6f282693cb9e
md"Ahora para las señales de Validacion/Test"

# ╔═╡ 777a04ce-69e7-416f-9d42-6e0ac98e2d80
PlutoUI.Resource("https://imgur.com/Me2cjOR.png")

# ╔═╡ 560c89f6-f582-470f-8408-f9bfcd5e2826
md"Vemos que, en general, si bien la predicción es peor, cuando realizamos el zoom sobre la zona $t \in (0, 0.1)$, observamos que la red aún logra capturar correctamente el decaimiento en la mayoría de los casos. Sin embargo, es cierto que comete más errores."

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "6e7bcec4be6e95d1f85627422d78f10c0391f199"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

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
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

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
# ╠═322d766c-1576-43c2-b344-fb571e5bd775
# ╟─cafda7f0-550f-40bd-8bad-2d82312a16db
# ╠═6104761b-b805-48ca-b8eb-50af3874071b
# ╟─c2d35967-1c47-49cf-af60-8eb2c47cc881
# ╟─32c9a071-20fc-49cd-95cc-b5945cf9b9f4
# ╟─7ab06e65-25bf-4680-819f-34e0c5de2c60
# ╠═0d169ce8-9ae2-4c10-9afc-816bc065788f
# ╟─76783e8a-9695-48a8-b73c-24f6e8b4adbc
# ╟─003518cd-8d72-4897-9ffb-011f66267f14
# ╠═159125e0-5e7d-4e85-99d8-38cc8cb18681
# ╟─3c73f87e-2934-428e-a48a-51971075aba9
# ╠═29b1d974-20b1-4d4b-99c6-c4b421d761c4
# ╟─7d1b873c-e0c1-4f62-beaa-320ad932dfa0
# ╟─25f727c6-34e9-4986-8207-96480a24c255
# ╟─708d13ef-7751-4ae0-bf85-039ad1dfa32f
# ╟─807af1dd-54cd-49f9-aead-57fd7d9119d4
# ╟─178e2c63-33e8-445b-8ad0-b7a80ccbd5b0
# ╟─d42752e8-38de-4b70-8e5d-b8ed6e1e23da
# ╟─5fd2127b-4db7-420e-a704-17b62381514c
# ╟─b25289b7-3661-4c04-8324-98ae21cd34ad
# ╟─4d7bf43e-dc46-41a1-b014-6f3c6b7bb73c
# ╟─eaeed495-970b-4364-989b-eef70e1a8ba1
# ╟─46c58cf8-825f-4264-aed6-71db8ff4913c
# ╟─e4cf90b5-d139-4022-914c-b19bb03b2dc1
# ╟─58702e1b-c78d-40bf-8fe5-288f8598534a
# ╟─122e341c-e6db-4ed5-9bae-9fb6af9b033f
# ╟─b8a49a8a-ea19-4f62-8289-2144c5b7086f
# ╟─8eee656b-a00a-4252-9017-28f5efce57fa
# ╟─3657f604-4f91-4f4d-9db2-7ae675401999
# ╟─a9f783e3-f81e-4df9-9e27-c9e66c8c185d
# ╟─3000a762-d315-42fa-8854-72529df8bfad
# ╠═3652d7b3-6b16-4000-8425-c6bdd90ce3d2
# ╟─44c6de9d-609a-4107-bb03-11c97b7286a4
# ╟─bfd00e81-d6d2-4bfa-95dd-e89a0d68df69
# ╟─f74ab92f-b34c-4a43-bff9-6f282693cb9e
# ╟─777a04ce-69e7-416f-9d42-6e0ac98e2d80
# ╟─560c89f6-f582-470f-8408-f9bfcd5e2826
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
