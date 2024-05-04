### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ e0d43e36-9410-4d5e-89d5-7ca289b27102
using PlutoUI

# ╔═╡ 054b047d-6eb9-473c-9bba-21adc18c45b9
md"# NOdes para realizar una predicción continua de señales de Hahn

## Predicción para una única señal

Comencemos por lo básico, primero que todo vamos a crear un modelo para predecír una única señal y ver como se comporta. 

Comenzamos intentando utilizar para entrenar una señal con solo 60 puntos, de los cuales vamos a suponer que 10 son puntos medidos para predecir la señal

Visualizemos primero los dos conjuntos que vamos a utilizar, como también las derivadas que se pasan como extra al modelo. En partícular vamos a utilizar la señal con $l_{cm} = 0.5$ $\mu$m y $\sigma = 1$.
"

# ╔═╡ ab94bb01-d059-4d31-9dd0-eef92adb72bd
PlutoUI.Resource("https://imgur.com/XSHXjdP.png")

# ╔═╡ a778665d-503d-4e3c-97cd-e7a81014d3f9
PlutoUI.Resource("https://imgur.com/J8FPbf0.png")

# ╔═╡ e5491fa7-0243-4101-9939-d13511e211bd
md"Veamos ahora las derivadas, porque esto influye bastante en la señal predicha por la red neuronal"

# ╔═╡ 5536bcd2-aa9c-4cd8-8123-94994edfea59
PlutoUI.Resource("https://imgur.com/IVJ2pqt.png")

# ╔═╡ 24c41cc8-3db4-47c9-ac9b-6af8844d88c1
md"Vamos a ver de que esto es probablemente lo que causa problemas en las predicciones"

# ╔═╡ 11f9189d-89ed-40e4-85a6-44191a1158d8
# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 45

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lcs = range(l0, lf, length = N)

t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))
    
# Concatenamos t_short and t_long
t = vcat(t_short, t_long)
toriginal = vcat(t_short, t_long)

# Leeemos los datos
# path_read = "/home/juan.morales/datos_PCA"
path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

# Parametros que se varian
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
sigmas = 0.01:0.01:1

######################################################################################

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

######################################################################################
    
# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto =  25 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 5

# Esto da 60 tiempos, distribuidosen 40 puntos desde 0 a 0.1 y 20 puntos desde 0.1 a 1
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]

t = vcat(t_short, t_long)

# Tomamos 1 sigmas y 1 tamaño de compartimientos para cada sigma o sea 60 señales
sampled_sigmas =  [1.0]
lcm_range = 1:250:250

println("Sigmas: ", sampled_sigmas)
println("Lcms: ", collect(lcms)[lcm_range])

# Obtenemos las señales representativas para un conjunto de sigmas y lcms
Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

# Numero de puntos para la validacion
n_valid = 11 # Terminan siendo 10

# Paso para tomar los tiempos de entrenamiento y validación
step = floor(Int, length(t) / n_valid) + 1

tvalid = t[1:step:end]
ttrain = [t for t in t if t ∉ tvalid]
# Concatenate in the begining
ttrain = vcat(0, ttrain)

indexes_train = [i for i in 1:length(t) if t[i] in ttrain]
indexes_valid = [i for i in 1:length(t) if t[i] in tvalid]

Signals_valid = Signals_rep[:,indexes_valid]
Signals_train = Signals_rep[:,indexes_train]

Signals_derivadas_train = Signals_rep_derivadas[indexes_train,:]
Signals_derivadas_valid = zeros(size(Signals_valid))

for i in 1:size(Signals_valid)[1]
    Signals_derivadas_valid[i,:] = derivate_signals(tvalid,Signals_valid[i,:])
end

Signals_derivadas_valid = Float32.(Matrix(Signals_derivadas_valid'))

# Normalizamos las derivadas
for i in 1:size(Signals_valid)[1]
    Signals_derivadas_valid[:,i] = Signals_derivadas_valid[:,i] ./ maximum(abs.(Signals_derivadas_valid[:,i]))
end

extra_parameters = Signals_derivadas_train
extra_parameters_valid = Signals_derivadas_vali

# ╔═╡ a6c7b7ed-467d-4374-b0bc-5a0c8f5b3256
md"Vamos a usar una red que funcionó anteriormente para pocas señales, la cual tiene 3 capas ocultas con 32, 64 y 16 neuronas con una función de activación tanh.

Además entrenamos este modelo con un batchsize de 5 puntos, un learning rate de 0.001 durante 2000 épocas con el método AdamW, es decir Adam con weight decay.

Seguimos utilizando la función de penalización para garantizar una señal decreciente con un $\lambda = 0.1$ en la función de Loss
"

# ╔═╡ bae54963-a828-45e7-b4f7-31e89d000442
# Las condiciones iniciales son todas 1 para cada señal
U0 = ones32(size(Signals_rep)[1])

#Definimos el batch size
batch_size = 5

# Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
train_loader = Flux.Data.DataLoader((Signals_train, ttrain), batchsize = batch_size)

# Función de activación
activation = tanh_fast

nn = Chain(Dense(2, 32, activation),
            Dense(32, 64, activation),
            Dense(64, 16, activation),
            Dense(16, 1)
            )

# Tomamos un learning rate de 0.001
η = 1e-3

# Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
epochs = 2000

# Parametrizar indices para poder acceder a los parametros extras de la red neuronal
f(x,p) = round(Int, x * (length(p) - 1)) + 1

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

##############################################################################################

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

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

    return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-5, abstol = 1e-5)) # Regresa la solución de la ODE
end

# Función que predice las señales para un conjunto de condiciones iniciales y parámetros extra
function Predict_Singals(U0, parametros_extra, time_batch)
    Predicted_Signals = zeros(size(time_batch))
    for i in 1:length(U0)
        u0 = Float32[U0[i]]
        predicted_signal = predict_NeuralODE(u0, parametros_extra[:, i], time_batch)[1, :]
        Predicted_Signals = hcat(Predicted_Signals, predicted_signal)
    end    
    Predicted_Signals[:,2:end]
end

# Función que calcula el loss de la red neuronal para un batch de datos y sus respectivos tiempos

# Término de pensalización

function penalization_term(time_batch,y)
    """
    Función de penalización para tratar de mantener la señal monotonamente decrecente
    En caso que la señal no sea monotonamente decreciente, la penalización es la suma de las diferencias positivas entre los tiempos
    """
	pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
end


function loss_node(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0, extra_parameters, time_batch)
    return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y))
end


# Hay dos funciones de loss porque le tengo que distinguir las derivadas del entrenamiento con las de validación.

function loss_valid(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0, extra_parameters_valid, time_batch)
    return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y))
end

# Función de callback para guardar el loss en cada época
global iter = 0
loss = []
loss_valid_array = []
callback = function ()
    global iter += 1
    if iter % (length(train_loader)) == 0
        epoch = Int(iter / length(train_loader))
        actual_loss = loss_node(Signals_train, ttrain)
        println("Epoch = $epoch || Loss: $actual_loss")
    end
    return false
end

###############################################################################################################

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, epochs), opt, cb = callback)

# ╔═╡ e5430075-31d9-45a4-afd1-21874efd13b8
md"Si vemos la función de loss del entrenamiento nos damos cuenta de que ya llegamos a un mínimo"

# ╔═╡ 207da352-0a66-4aae-8f85-342d1eebb5e8
PlutoUI.Resource("https://imgur.com/wvXdB9H.png")

# ╔═╡ 53330f4b-efbd-4a03-89bd-84eeeb873fdf
md"Si vemos la predicción con los puntos de entrenamiento, es decir pasandole a la red los púntos de entreanmiento con los de las derivadas de estos obtenemos una señal predicha razonable."

# ╔═╡ cb21a843-dc19-4555-9647-7bd0acf7cf59
PlutoUI.Resource("https://imgur.com/iCRUhU8.png")

# ╔═╡ 8d5251c0-ee65-4b85-8b88-e1c6f720a10a
md"Veamos ahora como se ve la señal predicha con los diez puntos de validación, y con la derivada calculada con estos puntos. Esto es todo lo que necesita la red para encontrar la dinámica. Además le pasamos los tiempos en los que queremos que nos dé los puntos predichos, vamos a pedirle los 60 tiempos originales."

# ╔═╡ e346b709-40b8-449a-8038-c07d87f3d396
PlutoUI.Resource("https://imgur.com/7phyotX.png")

# ╔═╡ d4e5a41c-a3b3-44d8-99d5-15fd39fb7200
md"Vemos que la predicción no es buena, tengamos en cuenta que solo neceistamos 10 puntos para obtener toda esta curva que serían los datos 'medidos'. Veamos que pasa si le damos mas puntos y la derivada de estos."

# ╔═╡ 58ef90bf-891e-410b-b2d0-5426c322eb27
PlutoUI.Resource("https://imgur.com/Dd9Lp1D.png")

# ╔═╡ 8b371f35-2033-4f54-8d3d-68e7a7f0ba8c
PlutoUI.Resource("https://imgur.com/Bjpxd17.png")

# ╔═╡ d4c20590-f0ff-4fbb-bf7d-7634090117cb
md"Ahora con 20 puntos 'medidos' la derivada es esta la cual todavía le falta información del pico pero se parece un poco mas a la derviada con la que se entrenó el modelo."

# ╔═╡ aa32ff80-d549-4b0b-b4b5-3729f11f81a6
PlutoUI.Resource("https://imgur.com/wacHpXL.png")

# ╔═╡ 26566557-e5e1-4e99-a9e1-edb9f8166cf7
md"La señal se pega un poco mas a la señal original y se parece mas a la predicha.

Podrían ser necesarios más paráemtros extras para distinguir las señales, uno interesante sería utilizar una recta que una los puntos inicial y final, esta es fácil de calcular para mediciones y coincide perfectamente con parámetros extra del entrenamiento. Probablemente esto haya sido uno de los problemas que estuvo teniendo la red al entrenar los modelos.

Algo simple que probé es agregar una recta con el valor inicial y final de los datos, en general esto serviría para casos donde la señal no llega a decaer por completo como para $\sigma = 1$ a estos tiempos.
"

# ╔═╡ a6099ec1-7be4-48b5-8a75-f8bb559ba234
PlutoUI.Resource("https://imgur.com/FYpPc5n.png")

# ╔═╡ b76b706b-5996-4766-b9da-344a667aea07
md"Como la recta necesita solo dos puntos podemos usar los mismos para el entrenamiento y la predicción. Hacer esto debería dar a la red mas información. Veamos los resultados que obtuvimos para la misma señal con un entrenamiento mucho mas corto."

# ╔═╡ 58ec0c8f-3dcf-49c2-972a-ee98a3e1bb6a
PlutoUI.Resource("https://imgur.com/7TzWZhE.png")

# ╔═╡ 60af69bb-7ec1-4afe-83cc-d7cfcd166f08
PlutoUI.Resource("https://imgur.com/XdeRb69.png")

# ╔═╡ f4fb8923-92f5-44ef-ba8e-b25af2974b34
md"Mucho mejor que antes la predicción que realiza, igualmente esto no es muy impresionante la idea sería usarlo para mas señales. En este caso usamos solo 10 puntos de la señal para predecir esta señal. Los cambios en el código fueron que ahora usamos 3 valores de entrada a la señal y el cálculo de la recta, que además depende únicamente de un solo punto porque sabemos siempre que empiezan en (0,1)."

# ╔═╡ 71f87e48-c3f8-479d-9db5-3a7b9c4c1928
function calucla_recta(x, x1, y1, x0 = 0, y0 = 1)
    m = (y1 - y0) / (x1 - x0)
    return Float32.(m .* x' .+ b)
end

# Función de activación
activation = tanh_fast

nn = Chain(Dense(3, 32, activation),
            Dense(32, 16, activation),
            Dense(16, 8, activation),
            Dense(8, 1)
            )

# Tomamos un learning rate de 0.001
η = 5e-3

# Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
epochs = 800

# Parametrizar indices para poder acceder a los parametros extras de la red neuronal
f(x,p) = round(Int, x * (length(p) - 1)) + 1

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

##############################################################################################

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

# Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
function predict_NeuralODE(u0, parametros, parametros2, time_batch)
    # dSdt = NN(S, parametros_extra) 
    function dSdt(u, p, t; parametros_extra = parametros, parametros_extra2 = parametros2)
        indx = f(t, parametros)
        parametros_actuales = parametros[indx] # Selecciona los parametros extra en el tiempo t
        parametros_actuales_2 = parametros2[indx]
        entrada_red = vcat(u, parametros_actuales, parametros_actuales_2) # Concatena los el valor de S(t) con los parametros extra en el tiempo t
        return re(p)(entrada_red) # Regresa la salida de la red neuronal re creada con los parámetros p
    end

    prob = ODEProblem(dSdt, u0, tspan)

    return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-5, abstol = 1e-5)) # Regresa la solución de la ODE
end

# Función que predice las señales para un conjunto de condiciones iniciales y parámetros extra
function Predict_Singals(U0, parametros_extra, parametros_extra2, time_batch)
    Predicted_Signals = zeros(size(time_batch))
    for i in 1:length(U0)
        u0 = Float32[U0[i]]
        predicted_signal = predict_NeuralODE(u0, parametros_extra[:, i], parametros_extra2[:, i], time_batch)[1, :]
        Predicted_Signals = hcat(Predicted_Signals, predicted_signal)
    end    
    Predicted_Signals[:,2:end]
end

# ╔═╡ 2b27f6d5-b80d-40aa-abf6-a1d6462887ed
md"## Mas de una señal

Probemos ahora que pasa cuando intentamos con la mísma arquitectura y método predecir más de una única señal. En este caso tomamos 5 señales mantiendo constante el $\sigma = 1$, para distintos valores de $l_{cm} = 0.5,~1.0,~1.5,~2.0,~2.5$ $\mu$m.

Esto nos da las siguientes señales:
"

# ╔═╡ 6cc43322-da4c-4468-a9c0-f65b77ee59f0
PlutoUI.Resource("https://imgur.com/LooUP6t.png")

# ╔═╡ 660faace-2bde-46f9-9bcb-986572cb76f5
md"Para esto hicimos nuevamente hicimos otra exploración de parámetros para poder encontrar cual es el mejor modelo para predecir estas señales.

## Arquitecturas con 4 capas

$$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossPredict} \\
    \hline
    1 & [3, 32, 32, 16, 8, 1] & \text{relu} & \text{AdamW} & 15 & 0.00467 & 0.16700 \\
    2 & [3, 32, 32, 16, 8, 1] & \text{relu} & \text{AdamW} & 30 & 0.02513 & 0.04321 \\
    3 & [3, 32, 32, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.01621 & 0.07001 \\
    4 & [3, 32, 32, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02804 & 0.09460 \\
    5 & [3, 32, 32, 16, 8, 1] & \text{swish} & \text{AdamW} & 15 & 0.01052 & 0.12846 \\
    6 & [3, 32, 32, 16, 8, 1] & \text{swish} & \text{AdamW} & 30 & 0.02816 & 0.03355 \\
    7 & [3, 32, 64, 16, 8, 1] & \text{relu} & \text{AdamW} & 15 & 0.00901 & 0.12464 \\
    8 & [3, 32, 64, 16, 8, 1] & \text{relu} & \text{AdamW} & 30 & 0.00261 & 0.18242 \\
    9 & [3, 32, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02860 & 0.02959 \\
    10 & [3, 32, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02660 & 0.07755 \\
    11 & [3, 32, 64, 16, 8, 1] & \text{swish} & \text{AdamW} & 15 & 0.01181 & 0.11249 \\
    12 & [3, 32, 64, 16, 8, 1] & \text{swish} & \text{AdamW} & 30 & 0.01032 & 0.15790 \\
    13 & [3, 128, 64, 16, 8, 1] & \text{relu} & \text{AdamW} & 15 & 0.03123 & 0.03414 \\
    14 & [3, 128, 64, 16, 8, 1] & \text{relu} & \text{AdamW} & 30 & 0.00902 & 0.21491 \\
    15 & [3, 128, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02581 & 0.09521 \\
    16 & [3, 128, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02362 & 0.05900 \\
    17 & [3, 128, 64, 16, 8, 1] & \text{swish} & \text{AdamW} & 15 & 0.01055 & 0.13962 \\
    18 & [3, 128, 64, 16, 8, 1] & \text{swish} & \text{AdamW} & 30 & 0.01053 & 0.15630 \\
    \hline
\end{array}
\end{aligned}$$

Si vemos parecen overfittear bastante por la diferencias de Loss finales. Veamos modelos mas simples, en este caso la exploración para:
## Arquitecturas con 3 capas

$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossPredict} \\
    \hline
    1 & [3, 32, 32, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.02911 & 0.03140 \\
    2 & [3, 32, 32, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.02980 & 0.03078 \\
    3 & [3, 32, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02321 & 0.06584 \\
    4 & [3, 32, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.03054 & 0.03023 \\
    5 & [3, 32, 32, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.02968 & 0.02929 \\
    6 & [3, 32, 32, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.03007 & 0.02873 \\
    7 & [3, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.01792 & 0.15084 \\
    8 & [3, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.02948 & 0.02958 \\
    9 & [3, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02412 & 0.07201 \\
    10 & [3, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.03021 & 0.03085 \\
    11 & [3, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.02855 & 0.04021 \\
    12 & [3, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.03007 & 0.02893 \\
    13 & [3, 128, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.02904 & 0.03047 \\
    14 & [3, 128, 64, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.02808 & 0.02961 \\
    15 & [3, 128, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02958 & 0.09842 \\
    16 & [3, 128, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.03019 & 0.03787 \\
    17 & [3, 128, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01005 & 0.13325 \\
    18 & [3, 128, 64, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.02911 & 0.03145 \\
    19 & [3, 5, 16, 8, 1] & \text{relu} & \text{AdamW} & 15 & 0.03024 & 0.03007 \\
    20 & [3, 5, 16, 8, 1] & \text{relu} & \text{AdamW} & 30 & 0.02799 & 0.03101 \\
    21 & [3, 5, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02013 & 0.30684 \\
    22 & [3, 5, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02650 & 0.11933 \\
    23 & [3, 5, 16, 8, 1] & \text{swish} & \text{AdamW} & 15 & 0.02956 & 0.03025 \\
    24 & [3, 5, 16, 8, 1] & \text{swish} & \text{AdamW} & 30 & 0.02835 & 0.03804 \\
    25 & [3, 16, 32, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.03031 & 0.04004 \\
    26 & [3, 16, 32, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.02983 & 0.05841 \\
    27 & [3, 16, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.00505 & 0.15298 \\
    28 & [3, 16, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.01589 & 0.08948 \\
    29 & [3, 16, 32, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01116 & 0.13723 \\
    30 & [3, 16, 32, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.00988 & 0.10878 \\
    \hline
\end{array}
\end{aligned}$

Veamos mejor las arquitecturas 9  de cuatro capas y la arquitectura 6 de 3 capas que tienen los menores loss de predicción. 

$$\begin{aligned}
& \begin{array}{ccccccc}
9 & [3, 32, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02860 & 0.02959 \\
6 & [3, 32, 32, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.03007 & 0.02873
\end{array}
\end{aligned}$$
"

# ╔═╡ 37559f37-caa3-40f1-bacb-b7921713f5ab
md"Empecemos con la la arquitectura 6, veamos particularmente la evolución del Loss en función de las épocas de entrenamiento."

# ╔═╡ b5437a4d-9614-4a5b-9cdd-1f16c88619bd
PlutoUI.Resource("https://imgur.com/BLyYYey.png")

# ╔═╡ 0d7a3bfa-70f3-41a7-94a7-d98da5fd3845
md"Por la cantidad de épocas no llega al overfitting pero tampoco predice bien las señales"

# ╔═╡ 3175bee3-910f-48b9-9bca-99d716ae3260
PlutoUI.Resource("https://imgur.com/2k0kQOa.png")

# ╔═╡ 6ba94324-60ae-4f2b-8341-b7e7ce6504cf
md"Por el loss no hace falta ver que la arquitectura con 4 capas da algo similar."

# ╔═╡ 4ecfc829-2e1f-4c77-bdfa-32e5f5842242
md"## Otros

Antes de añadir otro de los parámetros usamos todos los puntos para entrenar pocas señales. En estos casos tomamos el $\sigma = 1$, para distintos valores de $l_{cm} = 0.5,~0.75,~1.0,~1.25,~1.5$ $\mu$m. Estas señales son parecidas a las anteriores. Por lo que el embeeding que tenemos falla claramente. Los resultados para 3 y 4 capas dan lo siguiente:


$$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossPredict} \\
    \hline
    1 & [2, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.01677 & 0.01736 \\
    2 & [2, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.02128 & 0.02475 \\
    3 & [2, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.01898 & 0.01851 \\
    4 & [2, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02189 & 0.02101 \\
    5 & [2, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01655 & 0.01621 \\
    6 & [2, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.01778 & 0.01688 \\
    7 & [2, 128, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.01517 & 0.01640 \\
    8 & [2, 128, 64, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.01671 & 0.01700 \\
    9 & [2, 128, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02049 & 0.01995 \\
    10 & [2, 128, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02105 & 0.02029 \\
    11 & [2, 128, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01689 & 0.01652 \\
    12 & [2, 128, 64, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.01577 & 0.01630 \\
    \hline
\end{array}
\end{aligned}$$

$$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossPredict} \\
    \hline
    1 & [2, 128, 64, 32, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.01487 & 0.01648 \\
    1 & [2, 128, 64, 32, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.01498 & 0.01635 \\
    2 & [2, 128, 64, 32, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.01557 & 0.01699 \\
    3 & [2, 128, 64, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.01636 & 0.01659 \\
    4 & [2, 128, 64, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.01767 & 0.01767 \\
    5 & [2, 128, 64, 32, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01530 & 0.01612 \\
    6 & [2, 128, 64, 32, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.01565 & 0.01665 \\
    \hline
\end{array}
\end{aligned}$$


"

# ╔═╡ 1e2b0e2a-31de-4cde-8a87-6713e6579a7c
md"Si bien los loss de predicción bajan un poquito los de entrenamineto son peores que los anteriores por lo que siguen siendo las mismas pobres predicciones.

Veamos el loss y las predicciones de la arquietctura que mejor lo hizo en este caso la que tiene menor loss en las prediccione es la arquitectura con ID 5 pque tiene cuatro capas ocultas:

$$\begin{aligned}
& \begin{array}{ccccccc}
    5 & [2, 128, 64, 32, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01530 & 0.01612 \\
\end{array}
\end{aligned}$$"

# ╔═╡ 9b2f1a7e-8cd6-4a46-9668-b7b6549b997a
PlutoUI.Resource("https://imgur.com/DR6ZfnS.png")

# ╔═╡ d29f28e4-aa69-408b-8409-1c997571b846
md"Vemos que el error de Loss parece aumentar para la predicción con los puntos de validación mientras que el loss de entrenamiento baja a un ritmo lento."

# ╔═╡ 3fb6b95d-7418-4c46-b37a-b3e6b951b446
md"Veamos ahora las predicciones"

# ╔═╡ 59f81979-3b50-4fe1-9806-2fb91f0ae7c1
PlutoUI.Resource("https://imgur.com/vUOB13i.png")

# ╔═╡ 98365ca7-cf79-49ac-b9db-70947c235c33
md"### ¿Es la cantidad de puntos?
En los anteriores entrenamientos usamos 60 y 50 puntos de las señales para entrenar las redes, probamos esto para únicamente 30 puntos en el mismo caso que antes, útilizando solo como parámetro extra las derivadas calculadas con estos 30 puntos.

Los resultados para una arquitectura de 3 capas da lo siguiente

$$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossPredict} \\
    \hline
    1 & [2, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 5 & 0.00649 & 0.09001 \\
    2 & [2, 32, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.00864 & 0.04351 \\
    3 & [2, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 5 & 0.00618 & 0.06927 \\
    4 & [2, 32, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.00451 & 0.06629 \\
    5 & [2, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 5 & 0.01422 & 0.03350 \\
    6 & [2, 32, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01376 & 0.02555 \\
    7 & [2, 128, 64, 16, 1] & \text{relu} & \text{AdamW} & 5 & 0.00586 & 0.12099 \\
    8 & [2, 128, 64, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.00388 & 0.12966 \\
    9 & [2, 128, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 5 & 0.00691 & 0.09307 \\
    10 & [2, 128, 64, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.00384 & 0.08494 \\
    11 & [2, 128, 64, 16, 1] & \text{swish} & \text{AdamW} & 5 & 0.01013 & 0.05643 \\
    12 & [2, 128, 64, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01049 & 0.11811 \\
    \hline
\end{array}
\end{aligned}$$

Los resultados de entrenamiento mejoran, sin embargo las predicciones siguen siendo igual de malas
"

# ╔═╡ e9bfa74b-fe80-40b8-addc-bce041940774
md"## ¿Tomando conjuntos representativos?
Algo que ya hicimos abordando el problema como predicción de series temporales en vez de tomar todos los puntos es probar si el problema está en que se necesita un conjunto mas amplio de señales para el entrenamiento. Esta se entrenaron solo modelos de 4 capas y se entrenaron combinaciones de las señales con los paraámetros:

$\sigma = 0.01,~0.2,~0.4,~0.6,~0.8,~1.0$
$l_{cm} = 0.5,~0.75,~1.0,~1.25,~1.5,~1.75,~2.0,~2.25,~2.5,~2.75$

Los resultados se resumen en la siguiente tabla:

$$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossPredict} \\
    \hline
    1 & [2, 128, 64, 32, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.01498 & 0.01635 \\
    2 & [2, 128, 64, 32, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.01591 & 0.01699 \\
    3 & [2, 128, 64, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.01883 & 0.01845 \\
    4 & [2, 128, 64, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02058 & 0.02009 \\
    5 & [2, 128, 64, 32, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.01540 & 0.01604 \\
    6 & [2, 128, 64, 32, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.01572 & 0.01669 \\
    \hline
\end{array}
\end{aligned}$$


"

# ╔═╡ 48fe9cc7-2833-440d-943d-fb7d99ed84d9
md"En este caso los resultados son parecido para todas las redes, la que mejor Loss de predicción tiene es la arquitectura 5, que es idéntica a la anterior. Veamos esta vez las predicciones"

# ╔═╡ 6cc46127-d031-433b-ba4c-843fd2be2fc0
PlutoUI.Resource("https://imgur.com/x62o6Ag.png")

# ╔═╡ 28ce16aa-aee5-4fb8-91f6-9adb31d5c745
md"Las predicciones siguien siendo malas para estas exploraciones"

# ╔═╡ 878097ce-a953-4c64-894b-ba691593dfbd
md"## ¿Con parámetros continutos?

Los parámetros extra que estuvimos utilizando son solo puntos a distintos tiempos, ya sea de las señales o de la recta. Cuando la red hace una predicción lo hace en los tiempos que le pedimos que pueden no estar definidos para los parámetros extra. Algo que se puede hacer es tomar es una interpolación con los valores de las derivadas de las señales, así los parámetros extra serán continuos y no se tendrá este problema cunado se resuelve la ODE.

Comenzamos otra vez con pocas señales tomando el entrenamiento para las mismas señales con $\sigma = 1$, para distintos valores de $l_{cm} = 0.5,~1.0,~1.5,~2.0,~2.5$ $\mu$m.

las exploraciones con estos parámetros continuos se resumen en las siguientes tablas.

"

# ╔═╡ 12fae0f1-5cdf-4bfd-8b41-3f4d4da8c0aa
md"#### Para una arquitectura con 3 capas

$$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossPredict} \\
    \hline
    1 & [3, 5, 16, 8, 1] & \text{relu} & \text{AdamW} & 15 & 0.02743 & 0.02495 \\
    2 & [3, 5, 16, 8, 1] & \text{relu} & \text{AdamW} & 30 & 0.02765 & 0.02503 \\
    3 & [3, 5, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02142 & 0.02049 \\
    4 & [3, 5, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02784 & 0.02492 \\
    5 & [3, 5, 16, 8, 1] & \text{swish} & \text{AdamW} & 15 & 0.02736 & 0.02463 \\
    6 & [3, 5, 16, 8, 1] & \text{swish} & \text{AdamW} & 30 & 0.02704 & 0.02435 \\
    7 & [3, 16, 32, 16, 1] & \text{relu} & \text{AdamW} & 15 & 0.02756 & 0.02489 \\
    8 & [3, 16, 32, 16, 1] & \text{relu} & \text{AdamW} & 30 & 0.02003 & 0.01942 \\
    9 & [3, 16, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.01976 & 0.01932 \\
    10 & [3, 16, 32, 16, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02054 & 0.02003 \\
    11 & [3, 16, 32, 16, 1] & \text{swish} & \text{AdamW} & 15 & 0.02183 & 0.02009 \\
    12 & [3, 16, 32, 16, 1] & \text{swish} & \text{AdamW} & 30 & 0.02576 & 0.02226 \\
    \hline
\end{array}
\end{aligned}$$

"

# ╔═╡ f4f8921d-ffb4-44f9-9cc7-402766498473
md"#### Para una arquitectura con 4 capas

$$\begin{aligned}
& \begin{array}{ccccccc}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{Optim} & \text{BatchS} & \text{LossTrain} & \text{LossPredict} \\
    \hline
    1 & [3, 32, 32, 16, 8, 1] & \text{relu} & \text{AdamW} & 15 & 0.02706 & 0.02466 \\
    2 & [3, 32, 32, 16, 8, 1] & \text{relu} & \text{AdamW} & 30 & 0.02738 & 0.02479 \\
    3 & [3, 32, 32, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02063 & 0.01991 \\
    4 & [3, 32, 32, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02228 & 0.02053 \\
    5 & [3, 32, 32, 16, 8, 1] & \text{swish} & \text{AdamW} & 15 & 0.02207 & 0.01948 \\
    6 & [3, 32, 32, 16, 8, 1] & \text{swish} & \text{AdamW} & 30 & 0.02054 & 0.01989 \\
    7 & [3, 32, 64, 16, 8, 1] & \text{relu} & \text{AdamW} & 15 & 0.02241 & 0.01984 \\
    8 & [3, 32, 64, 16, 8, 1] & \text{relu} & \text{AdamW} & 30 & 0.02014 & 0.01990 \\
    9 & [3, 32, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.01540 & 0.01301 \\
    10 & [3, 32, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.01913 & 0.01570 \\
    11 & [3, 32, 64, 16, 8, 1] & \text{swish} & \text{AdamW} & 15 & 0.02143 & 0.01879 \\
    12 & [3, 32, 64, 16, 8, 1] & \text{swish} & \text{AdamW} & 30 & 0.02059 & 0.01982 \\
    13 & [3, 128, 64, 16, 8, 1] & \text{relu} & \text{AdamW} & 15 & 0.02061 & 0.01894 \\
    14 & [3, 128, 64, 16, 8, 1] & \text{relu} & \text{AdamW} & 30 & 0.02053 & 0.02049 \\
    15 & [3, 128, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 15 & 0.02158 & 0.02047 \\
    16 & [3, 128, 64, 16, 8, 1] & \text{tanh\_fast} & \text{AdamW} & 30 & 0.02166 & 0.01974 \\
    17 & [3, 128, 64, 16, 8, 1] & \text{swish} & \text{AdamW} & 15 & 0.02090 & 0.01894 \\
    18 & [3, 128, 64, 16, 8, 1] & \text{swish} & \text{AdamW} & 30 & 0.01544 & 0.01488 \\
    \hline
\end{array}
\end{aligned}$$
"

# ╔═╡ 0046f7de-c9fb-44df-91ae-d78fa8993c2a
md"De estas la red que mejor predice es la 9 con 4 capas, vamos a ver el loss y las predicciones: "

# ╔═╡ 384a261c-8560-4bbd-b0f3-09a2444685f3
PlutoUI.Resource("https://imgur.com/fk7UswI.png")

# ╔═╡ 2e57b008-99a1-415a-937e-f6832dfcd4ce
PlutoUI.Resource("https://imgur.com/68KRQu8.png")

# ╔═╡ 82876351-6189-416f-b9f5-05e219f89bc2
function recta(x1, y1, x0 = 0, y0 = 1)
    m = (y1 - y0) / (x1 - x0)
    b = y0
    function evalue_recta(x)
        return m .* x .+ b
    end
    return evalue_recta
end

recta_funcs = []

for i in 1:size(Signals_rep)[1]
    push!(recta_funcs, recta(t[end], Signals_rep[i,end], t[1], Signals_rep[i,1]))
end

# Derivadas de las señales con pocos puntos para predecir
Signals_derivadas_valid = zeros(size(Signals_valid))

for i in 1:size(Signals_valid)[1]
    Signals_derivadas_valid[i,:] = derivate_signals(tvalid,Signals_valid[i,:])
end

Signals_derivadas_valid = Float32.(Matrix(Signals_derivadas_valid'))

# Normalizamos las derivadas
for i in 1:size(Signals_valid)[1]
    Signals_derivadas_valid[:,i] = Signals_derivadas_valid[:,i] ./ maximum(abs.(Signals_derivadas_valid[:,i]))
end

itp_derivadas = []

# Interpolamos las derivadas
for i in 1:size(Signals_rep)[1]
    push!(itp_derivadas, CubicSpline(tvalid, Signals_derivadas_valid[:,i], extrapl = [2,], extrapr=[2,]))
end

extra_parameters = itp_derivadas
extra_parameters_valid = itp_derivadas
extra_parameters2 = recta_funcs

# ╔═╡ c7c6511c-d0ea-401e-b8dd-76628a65c8ce
md"Hay un parámetro de las NODE que hubo que bajar para que no consuma tanta memoria entrenarlas y es el método y tolerancias:

* abstol (absolute tolerance in changes of the objective value)
* reltol (relative tolerance in changes of the objective value)

que estaban en $1 \times 10^{-7}$, que pasa si bajamos esto a $1 \times 10^{-9}$ esto debería dar pedir menos diferencia entre los valores que predice la red y los objetivos. Además de esto sacamos el término de penalización que ya no está contribuyendo al loss para que las predicciones tengan la libertad de subir y ajustarse a las soluciones.

entrenando unas 200 épocas mas con estas configuraciones la red anterior tenemos lo siguiente:
"

# ╔═╡ 4d24e3a2-8c30-4f78-9d88-895ad8427dc6
PlutoUI.Resource("https://imgur.com/LzEYEHQ.png")

# ╔═╡ ce7a7fb3-339e-4c26-8f9e-b9d7bc314984
md"Esto podría ser un punto de partida para entrenar un buen modelo que generalice los resultados."

# ╔═╡ 28cd6d2e-f493-4b26-9d5f-35d0b766d890


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
# ╠═e0d43e36-9410-4d5e-89d5-7ca289b27102
# ╟─054b047d-6eb9-473c-9bba-21adc18c45b9
# ╟─ab94bb01-d059-4d31-9dd0-eef92adb72bd
# ╟─a778665d-503d-4e3c-97cd-e7a81014d3f9
# ╟─e5491fa7-0243-4101-9939-d13511e211bd
# ╟─5536bcd2-aa9c-4cd8-8123-94994edfea59
# ╟─24c41cc8-3db4-47c9-ac9b-6af8844d88c1
# ╠═11f9189d-89ed-40e4-85a6-44191a1158d8
# ╟─a6c7b7ed-467d-4374-b0bc-5a0c8f5b3256
# ╠═bae54963-a828-45e7-b4f7-31e89d000442
# ╟─e5430075-31d9-45a4-afd1-21874efd13b8
# ╟─207da352-0a66-4aae-8f85-342d1eebb5e8
# ╟─53330f4b-efbd-4a03-89bd-84eeeb873fdf
# ╟─cb21a843-dc19-4555-9647-7bd0acf7cf59
# ╟─8d5251c0-ee65-4b85-8b88-e1c6f720a10a
# ╟─e346b709-40b8-449a-8038-c07d87f3d396
# ╟─d4e5a41c-a3b3-44d8-99d5-15fd39fb7200
# ╟─58ef90bf-891e-410b-b2d0-5426c322eb27
# ╟─8b371f35-2033-4f54-8d3d-68e7a7f0ba8c
# ╟─d4c20590-f0ff-4fbb-bf7d-7634090117cb
# ╟─aa32ff80-d549-4b0b-b4b5-3729f11f81a6
# ╟─26566557-e5e1-4e99-a9e1-edb9f8166cf7
# ╟─a6099ec1-7be4-48b5-8a75-f8bb559ba234
# ╟─b76b706b-5996-4766-b9da-344a667aea07
# ╟─58ec0c8f-3dcf-49c2-972a-ee98a3e1bb6a
# ╟─60af69bb-7ec1-4afe-83cc-d7cfcd166f08
# ╟─f4fb8923-92f5-44ef-ba8e-b25af2974b34
# ╠═71f87e48-c3f8-479d-9db5-3a7b9c4c1928
# ╟─2b27f6d5-b80d-40aa-abf6-a1d6462887ed
# ╠═6cc43322-da4c-4468-a9c0-f65b77ee59f0
# ╟─660faace-2bde-46f9-9bcb-986572cb76f5
# ╟─37559f37-caa3-40f1-bacb-b7921713f5ab
# ╟─b5437a4d-9614-4a5b-9cdd-1f16c88619bd
# ╟─0d7a3bfa-70f3-41a7-94a7-d98da5fd3845
# ╠═3175bee3-910f-48b9-9bca-99d716ae3260
# ╟─6ba94324-60ae-4f2b-8341-b7e7ce6504cf
# ╟─4ecfc829-2e1f-4c77-bdfa-32e5f5842242
# ╟─1e2b0e2a-31de-4cde-8a87-6713e6579a7c
# ╟─9b2f1a7e-8cd6-4a46-9668-b7b6549b997a
# ╟─d29f28e4-aa69-408b-8409-1c997571b846
# ╟─3fb6b95d-7418-4c46-b37a-b3e6b951b446
# ╟─59f81979-3b50-4fe1-9806-2fb91f0ae7c1
# ╟─98365ca7-cf79-49ac-b9db-70947c235c33
# ╟─e9bfa74b-fe80-40b8-addc-bce041940774
# ╟─48fe9cc7-2833-440d-943d-fb7d99ed84d9
# ╟─6cc46127-d031-433b-ba4c-843fd2be2fc0
# ╟─28ce16aa-aee5-4fb8-91f6-9adb31d5c745
# ╟─878097ce-a953-4c64-894b-ba691593dfbd
# ╟─12fae0f1-5cdf-4bfd-8b41-3f4d4da8c0aa
# ╟─f4f8921d-ffb4-44f9-9cc7-402766498473
# ╟─0046f7de-c9fb-44df-91ae-d78fa8993c2a
# ╟─384a261c-8560-4bbd-b0f3-09a2444685f3
# ╟─2e57b008-99a1-415a-937e-f6832dfcd4ce
# ╠═82876351-6189-416f-b9f5-05e219f89bc2
# ╟─c7c6511c-d0ea-401e-b8dd-76628a65c8ce
# ╟─4d24e3a2-8c30-4f78-9d88-895ad8427dc6
# ╟─ce7a7fb3-339e-4c26-8f9e-b9d7bc314984
# ╠═28cd6d2e-f493-4b26-9d5f-35d0b766d890
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
