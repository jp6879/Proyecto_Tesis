### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ bb3273bf-1387-45f6-83dc-5c4d734e4ed5
import PlutoUI

# ╔═╡ bd5741c9-2c04-479a-8a66-ca374c66a758
md"# Exploración con todas las señales

Después de haber encontrado los hiperparámetros para una red que predice señales de RMN pesadas por una distribución de tamaños log-normal con $\sigma = 1$ y variando $l_{cm}$, buscamos encontrar los hiperparámetros de una red más general. Esta red busca predecir señales que se obtienen al variar $\sigma$ y $l_{cm}$ en la distribución de tamaños.

* Se utilizaron las señales de los siguientes parámetros dividiendo señales para entrenamiento y otras para validación.

- $\sigma = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]$

- $l_{cm} = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]~(\text{entrenamiento})$

- $l_{cm} = [1.75, 3.0, 4.25]~(\text{validación})$


Dado que el hecho de agregar muchas mas señales al set de entrenamiento relentiza el tiempo en que tarda el mismo se realizó un grid-search con un barrido grueso sobre los hiperparámetros.

En primer lugar debido a que obtuvimos buenos resultados anteriormente al utilizar redes con 3 y 4 capas y con funciones de activación relu y swish respectivamente intentamos únicamente variar el número de neuoronas en la exploración. 

Se utilizó además el optimizador AdamW con dos learning rates distintos 0.001 y 0.1. La función de costo utilizada fue la MSE con una regularización L2 utilizando tres valores distintos de $\lambda$ = 0.0, 0.01, 0.1. 

Se utilizaron mini-batchs para el entrenamiento donde se varío el tamaño del mismo en 10, 20 y 30 señales por batch.

Por ultimo se añadió dropout a las capas ocultas variando la fracción p en 0.0, 0.2 y 0.5.

La exploración de estos hiperparámetros nos deja con 216 redes por entrenar.

Las señales fueron muestreadas únicamente con 40 púntos para mejorar la rapidez del entrenamiento. Las señales de entrenamiento y validación junto con los puntos utilizados se muestran en las siguientes figuras:"

# ╔═╡ 6770b2db-b083-432a-8f7c-63822df3e22b
PlutoUI.Resource("https://imgur.com/t1hKULS.png")

# ╔═╡ 22c5fa8b-aa4d-4c1e-83bf-057863559d0f
PlutoUI.Resource("https://imgur.com/gtJ5rQa.png")

# ╔═╡ 2af67fbb-53f1-42ba-9dd0-45eece09efea
# Progarma para la exploración de hiperparámetros en una Neural Ordinary Differential Equations (NODE) con Mini-Batchs
using Flux
using Flux: train!
using DataFrames
using CSV
using DifferentialEquations
using SciMLSensitivity
using ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux
using Interpolations
using OrdinaryDiffEq
using IterTools: ncycle
using BSplineKit

###################################################################################
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

# path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"
path_read = "/home/juan.morales/datos_PCA"

# Parametros que se varian
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
sigmas = 0.01:0.01:1

###################################################################################
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

###################################################################################

# Funcion para leer las señales desde el path_read
function GetSignals(path_read)
    dataSignals = CSV.read(path_read * "/dataSignals.csv", DataFrame)
    dataSignals = Matrix(dataSignals)
    return dataSignals
end

# Funcion que toma las señales representativas para un conjunto de sigmas y lcms además las muestrea según el muestreo corto y largo
function Get_Rep_Signals(indexes, sampled_sigmas, lcm_range, path_read, muestreo_corto, muestreo_largo)
    dataSignals = Float32.(GetSignals(path_read))
    Signals_rep = Float32.(Matrix(dataSignals[:,indexes]'))
    Signals_rep_short = Signals_rep[:,1:muestreo_corto:1000]
    Signals_rep_long = Signals_rep[:,1001:muestreo_largo:end]
    Signals_rep = hcat(Signals_rep_short, Signals_rep_long)
    return Signals_rep
end

###################################################################################
# Función que idientifca las señales que se van a utilizar para el entrenamiento de la NODE con lcm y σ
function Get_Signals_Test_Parameters(lcms,sigmas)
    dim1 = dimlcm = length(lcms)
    dim2 = dimsigma = length(sigmas)

    column_lcm = zeros(dim1*dim2)
    column_sigmas = zeros(dim1*dim2)
    aux_lcm = collect(lcms)
    aux_sigmas = collect(sigmas)

    for i in 1:dim1
        for j in 1:dim2
            column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
            column_sigmas[(i - 1)*dim2 + j] = aux_sigmas[j]
        end
    end
    
    return column_lcm, column_sigmas
end

###################################################################################

# Función que devuelve señales de prueba, sus derivadas y los parámetros con los que se generaron
function Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)
    # Obtenemos primero los parámetros con los que se generaron las señales
    column_lcm, column_sigmas = Get_Signals_Test_Parameters(lcms,sigmas)
    df_SignalsParams = DataFrame(
        sigmas = column_sigmas,
	    lcm = column_lcm,
	)
    indexes = []
    # A partir de ciertos sigmas dados por sampled_sigmas y un rango de lcm dado por lcm_range, obtenemos las señales representativas
    for sigma in sampled_sigmas
        find_rows = findall(x -> x == sigma, df_SignalsParams.sigmas)[lcm_range]
        push!(indexes, find_rows)
    end
    # Indices de estas señales
    indexes = vcat(indexes...)
    # Obtenemos las señales representativas ya muestreadas
    Signals_rep = Get_Rep_Signals(indexes, sampled_sigmas, lcm_range, path_read, muestreo_corto, muestreo_largo)
    # Derivadas de las señales
    Signals_rep_derivadas = zeros(size(Signals_rep))
    for i in 1:size(Signals_rep)[1]
        Signals_rep_derivadas[i,:] = derivate_signals(t,Signals_rep[i,:])
    end
    Signals_rep_derivadas = Float32.(Matrix(Signals_rep_derivadas'))
    # Normalizamos las derivadas
    for i in 1:size(Signals_rep)[1]
        Signals_rep_derivadas[:,i] = Signals_rep_derivadas[:,i] ./ maximum(abs.(Signals_rep_derivadas[:,i]))
    end
    column_lcm_rep = column_lcm[indexes]
    column_sigmas_rep = column_sigmas[indexes]
    return Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep
end

###################################################################################
# Función que obtiene las derivadas de las señales y las devuelve normalizadas
function get_signals_deriv(t, Signals)
    Signals_derivadas= zeros(size(Signals))

    # Obtenemos las derivadas de las señales de validación
    for i in 1:size(Signals)[1]
        Signals_derivadas[i,:] = derivate_signals(t,Signals[i,:])
    end

    # La transponemos y la convertimos a Float32
    Signals_derivadas = Float32.(Matrix(Signals_derivadas'))

    # Normalizamos las derivadas
    for i in 1:size(Signals)[1]
        Signals_derivadas[:,i] = Signals_derivadas[:,i] ./ maximum(abs.(Signals_derivadas[:,i]))
    end

    return Signals_derivadas
end

# Función que devuelve un interpolador lineal para un conjunto de señales y tiempos dados
# Las señales entran como un arreglo de arreglos y los tiempos como un arreglo
function get_interpolated(t, Signals)
    itp = []
    # Interpolamos las derivadas
    for i in 1:size(Signals)[2]
        interpol = BSplineKit.interpolate(t, Signals[:,i], BSplineOrder(2))
        extrapol = BSplineKit.extrapolate(interpol, Smooth())
        push!(itp, extrapol)
        # push!(itp, cubic_spline_interpolation(t, Signals[:,i], extrapolation_bc=Line())) # Interpolación cúbica
        # push!(itp, extrapolate(interpolate((t, ), Signals[:,i], BSpline(Quadratic())), Line()))
    end
    return itp
end

###################################################################################
# Función que crea el modelo de la red neuronal que va a estar dentro de la ODE
# Se añade además un dropout en cada capa oculta
function create_model(layers::Vector{Int}, activation::Function, dropout_rate)
    activations = [activation for i in 1:length(layers) - 2]
    startlayer = Dense(layers[1], layers[2])
    hiddenlayers = hcat([[Dense(layers[i], layers[i+1], activations[i]), Dropout(dropout_rate)] for i in 2:length(layers) - 2]...)
    endlayer = Dense(layers[end-1], layers[end])
    return Chain(startlayer, hiddenlayers..., endlayer)
end

###################################################################################

# Función que entrena la NODE con mini-batchs
function Train_Neural_ODE(nn, U0, U0_valid, extra_parameters, extra_parameters2, 
                            extra_parameters_valid, extra_parameters2_valid,
                            epochs, train_loader, opt, eta, Signals_rep, Signals_valid, t, lambd, actual_id)
    
    # Tiempo sobre el cual resolver
    tspan = (0f0, 1f0)
    
    # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial
    p, re = Flux.destructure(nn) 

    # Si existe el archivo con los parámetros de la red previamente los cargamos
    if isfile("/home/juan.morales/ExploracionIntV3/M_layers/Parameters/$(actual_id)_Parameters.csv")
        theta = CSV.read("/home/juan.morales/ExploracionIntV3/M_layers/Parameters/$(actual_id)_Parameters.csv", DataFrame)
        p = Float32.(theta[:,1])
    else
        println("No se encontraron los parámetros de la red neuronal")
    end
    
    # Optimizardor
    opt = opt(eta)

    # Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
    function predict_NeuralODE(u0, parametros, parametros2, time_batch)
        # dSdt = NN(S, parametros_extra) 
        function dSdt(u, p, t; parametros_extra = parametros, parametros_extra2 = parametros2)
            parametros_actuales = parametros.(t) # Selecciona los parametros extra en el tiempo t
            parametros_actuales_2 = parametros2.(t)
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

    # Función de penalización para tratar de mantener la señal monotonamente decrecente
    function penalization_term(time_batch,y)
        pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
        return pen
    end

    # Función de pérdida que vamos a minimizar, recibe un batch de señales y un batch de tiempos
    function loss_node(batch, time_batch)
        y = Predict_Singals(U0, extra_parameters, extra_parameters2, time_batch)
        return Flux.mse(y, batch') + lambd * sum(p.^2)
        #+ lamb * (penalization_term(time_batch, y))
    end
    
    # Función de Loss de validación, acá usamos las derivadas de la señal con pocos puntos
    function loss_valid(batch, time_batch)
        y = Predict_Singals(U0_valid, extra_parameters_valid, extra_parameters2_valid, time_batch)
        return Flux.mse(y, batch') + lambd * sum(p.^2)
        #+ lamb * (penalization_term(time_batch, y))
    end

    # Función de callback para guardar el loss cada epoch
    global iter = 0
    loss = []
    loss_forecast = []
    callback = function ()
        global iter += 1
        if iter % (length(train_loader)) == 0
            epoch = Int(iter / length(train_loader))
            actual_loss = loss_node(Signals_rep, t)
            forecast_loss = loss_valid(Signals_valid, t)
            println("Epoch = $epoch || Loss: $actual_loss || Loss valid: $forecast_loss")
            push!(loss, actual_loss)
            push!(loss_forecast, forecast_loss)
        end
        return false
    end

    # Entrenamos la red neuronal con mini-batchs
    Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, epochs), opt, cb = callback)
    
    # Devolvemos el loss final y los parametros de la red neuronal
    return loss, p, loss_forecast

end

function main()
    # Arquitecturas que vamos a utilizar
    architectures = [
        [[3, 32, 32, 16, 1], relu], # Tres capas ocultas simple
        [[3, 128, 64, 16, 1], relu], # Tres capas ocultas con mas neuronas
        [[3, 16, 32, 64, 16, 1], swish], # Cuatro capas ocultas simple
        [[3, 32, 64, 32, 16, 1], swish], # Cuatro capas ocultas con mas neuronas
        ]

    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [AdamW]]

    # Numero de mini-batchs que vamos a utilizar 
    batchs_size = [10, 20, 30]

    # Dropout rates
    dropout_rates = [0.0, 0.2, 0.5]

    # Lambdas regularizacion L2
    lambdas = [0.0, 0.01, 0.1]

    # Learning rates
    learning_rates = [0.001, 0.1]

    # Vector de configuraciones que vamos a utilizar
    configuraciones = []

    for arch in architectures, opt in optimizers, batch_size in batchs_size, lambd in lambdas, eta in learning_rates, dropout_rate in dropout_rates
        push!(configuraciones, (arch, opt, batch_size, lambd, eta, dropout_rate))
    end

    path_read = "/home/juan.morales/datos_PCA"
    # path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"
    
    # Distintos muestreos de tiempo
    t_short = Float32.(collect(range(0, 0.1, length = 1000)))
    t_long = Float32.(collect(range(0.1, 1, length = 100)))
        
    # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
    muestreo_corto = 30 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
    muestreo_largo = 19

    # Esto da 100 tiempos 50 puntos desde 0 a 0.1 y 25 puntos desde 0.1 a 1
    t_short = t_short[1:muestreo_corto:end]
    t_long = t_long[1:muestreo_largo:end]
    
    t = vcat(t_short, t_long)

    # Tomamos 1 sigmas y 5 tamaños de compartimientos para cada sigma o sea 60 señales
    sampled_sigmas =  [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
    lcm_range = 1:100:600
    lcm_valid = 1:125:600
    lcm_test = 1:130:600
 
    println("Sigmas: ", sampled_sigmas)
    println("Lcms: ", collect(lcms)[lcm_range])
    lcm_valid = [l for l in collect(lcm_valid) if l ∉ collect(lcm_range)]
    lcm_test = [l for l in collect(lcm_test) if l ∉ collect(lcm_range)]
    println("Lcms validacion: ", collect(lcms)[lcm_valid])
    println("Lcms test: ", collect(lcms)[lcm_test])
    
    # Obtenemos las señales representativas para un conjunto de sigmas y lcms
    # Obtenemos las señales representativas para un conjunto de sigmas y lcms
    Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)
    Signals_valid, Signals_valid_derivadas, column_lcm_valid, column_sigmas_valid = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_valid, muestreo_corto, muestreo_largo, t)
    # Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmas_test = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_test, muestreo_corto, muestreo_largo, t)

    # Numero de puntos para la predicción
    n_valid = 10

    # Paso para tomar los tiempos de entrenamiento y validación
    step = floor(Int, length(t) / n_valid) + 1

    t_usados = t[1:step:end]

    # En la validación y en el train tenemos que tener el primer y último tiempo
    if t[end] ∉ t_usados
        t_usados = vcat(t_usados, t[end])
    end

    indexes_usados = [i for i in 1:length(t) if t[i] in t_usados]

    Signals_rep_puntos = Signals_rep[:,indexes_usados]
    Signals_valid_puntos = Signals_valid[:,indexes_usados]
    # Signals_test_puntos = Signals_test[:,indexes_usados]

    # Obtenemos las derivadas de las señales de validación
    Signals_rep_derivadas_puntos = get_signals_deriv(t_usados, Signals_rep_puntos)
    Signals_valid_derivadas_puntos = get_signals_deriv(t_usados, Signals_valid_puntos)
    # Signals_test_derivadas_puntos = get_signals_deriv(t_usados, Signals_test_puntos)

    # Obtenemos las interpolaciones de las señales y de las derivadas
    itp_signals = get_interpolated(t_usados, Signals_rep_puntos')
    itp_derivadas = get_interpolated(t_usados, Signals_rep_derivadas_puntos)

    itp_signals_valid = get_interpolated(t_usados, Signals_valid_puntos')
    itp_derivadas_valid = get_interpolated(t_usados, Signals_valid_derivadas_puntos)

    # itp_signals_test = get_interpolated(t_usados, Signals_test_puntos')
    # itp_derivadas_test = get_interpolated(t_usados, Signals_test_derivadas_puntos)

    extra_parameters = itp_signals
    extra_parameters2 = itp_derivadas

    extra_parameters_valid = itp_signals_valid
    extra_parameters2_valid = itp_derivadas_valid

    # extra_parameters_test = itp_signals_test
    # extra_parameters2_test = itp_derivadas_test

    # Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
    epochs = 500

    # Todas las señales tienen la misma condición inicial U0 = 1
    U0 = ones32(size(Signals_rep)[1])
    U0_valid = ones32(size(Signals_valid)[1])
    # U0_test = ones32(size(Signals_test)[1])

    # Para el entrenamiento en el cluster vamos iterando sobre las configuraciones y guardamos los resultados en archivos csv
    # architecture, opt, batch_size, lambd, eta, dropout_rate = configuraciones[1]
    architecture, opt, batch_size, lambd, eta, dropout_rate = configuraciones[parse(Int128, ARGS[1])]

    layers = architecture[1]
    activation = architecture[2]

    if activation == tanh_fast
        activation_string = "tanh_fast"
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
    train_loader = Flux.Data.DataLoader((Signals_rep, t), batchsize = batch_size)

    # Vamos a crear el modelo de la red neuronal que va a estar dentro de la ODE
    nn = create_model(layers, activation, dropout_rate)

    # Número de modelo
    # actual_id = 1
    actual_id = parse(Int128,ARGS[1])

    # Entrenamos una NODE con mini-batchs para cada arquitectura, optimizador y tamaño de mini-batch y guardamos el loss y los parametros de la red neuronal
    architecture_loss, theta, loss_forecast = Train_Neural_ODE(nn, U0, U0_valid, extra_parameters, extra_parameters2, 
                                                                extra_parameters_valid, extra_parameters2_valid,
                                                                epochs, train_loader, opt, eta, Signals_rep, Signals_valid, t, lambd, actual_id)

    # Guardamos los hiperparámetros del entrenamiento y el loss final
    actual_layer = string(layers)
    actual_activation = activation_string
    actual_optimizer = opt_string
    actual_batch_size = batch_size
    actual_lambda = lambd
    actual_eta = eta
    actual_dropout_rate = dropout_rate
    actual_loss_final_train = architecture_loss[end]
    actual_loss_final_forecast = loss_forecast[end]


    # Guardamos los resultados en un archivo csv
    df_results_total = DataFrame(ID = actual_id, 
                                Arquitectura = actual_layer, 
                                Activación = actual_activation,
                                Optimizador = actual_optimizer,
                                Batch_Size = actual_batch_size,
                                Lambda = actual_lambda,
                                Eta = actual_eta,
                                Loss_Final_Entrenamiento = actual_loss_final_train,
                                Loss_Final_Predicción = actual_loss_final_forecast,
                                Dropout_rate = actual_dropout_rate)

    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    CSV.write("/home/juan.morales/ExploracionIntV3/M_layers/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size)_$(actual_lambda)_$(actual_eta)_$(actual_dropout_rate).csv", df_results_total)

    # Guardamos los loss y los parametros de la red neuronal en archivos csv
    Loss_Matrix = zeros((length(architecture_loss), 2))

    for i in 1:length(architecture_loss)
        Loss_Matrix[i,1] = architecture_loss[i]
        Loss_Matrix[i,2] = loss_forecast[i]
    end

    df_losses = DataFrame(Loss_Matrix, :auto)

    rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
    rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

    # Chequeamos si existe previamente un archivo CSV y si existe concatenamos al actual
    if isfile("/home/juan.morales/ExploracionIntV3/M_layers/Losses/$(actual_id)_losses.csv")
        df_losses = vcat(CSV.read("/home/juan.morales/ExploracionIntV3/M_layers/Losses/$(actual_id)_losses.csv", DataFrame), df_losses)
    end

    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Losses/$(actual_id)_losses.csv", df_losses)
    CSV.write("/home/juan.morales/ExploracionIntV3/M_layers/Losses/$(actual_id)_losses.csv", df_losses)

    df_theta = DataFrame(reshape(theta, length(theta), 1), :auto)
    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Parameters/$(actual_id)_Parameters.csv", df_losses)
    CSV.write("/home/juan.morales/ExploracionIntV3/M_layers/Parameters/$(actual_id)_Parameters.csv", df_theta)
end

main()

# ╔═╡ 56087631-4282-4aa5-a83b-cde277924d68
md"Los resultados obtenidos son los siguientes: 
$$\begin{aligned}
& \begin{array}{}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{BatchS} & \lambda & \eta & \text{LossTrain} & \text{LossValid} & \text{DP} \\
    \hline
        1 & [3, 32, 32, 16, 1] & relu & 10 & 0.0 & 0.001 & 0.001891 & 0.001631 & 0.0 & ~ \\ 
        4 & [3, 32, 32, 16, 1] & relu & 10 & 0.0 & 0.1 & 0.001226 & 0.0009682 & 0.0 & ~ \\ 
        7 & [3, 32, 32, 16, 1] & relu & 10 & 0.01 & 0.001 & 0.288 & 0.2741 & 0.0 & ~ \\ 
        10 & [3, 32, 32, 16, 1] & relu & 10 & 0.01 & 0.1 & 0.2907 & 0.2798 & 0.0 & ~ \\ 
        13 & [3, 32, 32, 16, 1] & relu & 10 & 0.1 & 0.001 & 0.6132 & 0.7135 & 0.0 & ~ \\ 
        16 & [3, 32, 32, 16, 1] & relu & 10 & 0.1 & 0.1 & 0.6156 & 0.7157 & 0.0 & ~ \\ 
        17 & [3, 32, 32, 16, 1] & relu & 10 & 0.1 & 0.1 & 0.6141 & 0.7141 & 0.2 & ~ \\ 
        19 & [3, 32, 32, 16, 1] & relu & 20 & 0.0 & 0.001 & 0.002078 & 0.00177 & 0.0 & ~ \\ 
        25 & [3, 32, 32, 16, 1] & relu & 20 & 0.01 & 0.001 & 0.2878 & 0.2735 & 0.0 & ~ \\ 
        28 & [3, 32, 32, 16, 1] & relu & 20 & 0.01 & 0.1 & 0.288 & 0.2754 & 0.0 & ~ \\ 
        31 & [3, 32, 32, 16, 1] & relu & 20 & 0.1 & 0.001 & 0.6132 & 0.7136 & 0.0 & ~ \\ 
        34 & [3, 32, 32, 16, 1] & relu & 20 & 0.1 & 0.1 & 0.6136 & 0.7137 & 0.0 & ~ \\ 
        35 & [3, 32, 32, 16, 1] & relu & 20 & 0.1 & 0.1 & 0.6137 & 0.7139 & 0.2 & ~ \\ 
        37 & [3, 32, 32, 16, 1] & relu & 30 & 0.0 & 0.001 & 0.002308 & 0.001969 & 0.0 & ~ \\ 
        40 & [3, 32, 32, 16, 1] & relu & 30 & 0.0 & 0.1 & 0.0008262 & 0.0007238 & 0.0 & ~ \\ 
        43 & [3, 32, 32, 16, 1] & relu & 30 & 0.01 & 0.001 & 0.2938 & 0.2944 & 0.0 & ~ \\ 
        46 & [3, 32, 32, 16, 1] & relu & 30 & 0.01 & 0.1 & 0.2958 & 0.2976 & 0.0 & ~ \\ 
        49 & [3, 32, 32, 16, 1] & relu & 30 & 0.1 & 0.001 & 0.617 & 0.7138 & 0.0 & ~ \\ 
        52 & [3, 32, 32, 16, 1] & relu & 30 & 0.1 & 0.1 & 0.6175 & 0.7139 & 0.0 & ~ \\ 
        55 & [3, 128, 6, 16, 1] & relu & 10 & 0.0 & 0.001 & 0.003357 & 0.002532 & 0.0 & ~ \\ 
        61 & [3, 128, 6, 16, 1] & relu & 10 & 0.01 & 0.001 & 0.2883 & 0.2743 & 0.0 & ~ \\ 
        64 & [3, 128, 6, 16, 1] & relu & 10 & 0.01 & 0.1 & 0.2917 & 0.2812 & 0.0 & ~ \\ 
        67 & [3, 128, 6, 16, 1] & relu & 10 & 0.1 & 0.001 & 0.6134 & 0.7145 & 0.0 & ~ \\ 
        70 & [3, 128, 6, 16, 1] & relu & 10 & 0.1 & 0.1 & 0.617 & 0.7171 & 0.0 & ~ \\ 
        73 & [3, 128, 6, 16, 1] & relu & 20 & 0.0 & 0.001 & 0.005246 & 0.003728 & 0.0 & ~ \\ 
        82 & [3, 128, 6, 16, 1] & relu & 20 & 0.01 & 0.1 & 0.2881 & 0.276 & 0.0 & ~ \\ 
        85 & [3, 128, 6, 16, 1] & relu & 20 & 0.1 & 0.001 & 0.6177 & 0.7202 & 0.0 & ~ \\ 
        88 & [3, 128, 6, 16, 1] & relu & 20 & 0.1 & 0.1 & 0.6133 & 0.7134 & 0.0 & ~ \\ 
        91 & [3, 128, 6, 16, 1] & relu & 30 & 0.0 & 0.001 & 0.006963 & 0.005239 & 0.0 & ~ \\ 
        97 & [3, 128, 6, 16, 1] & relu & 30 & 0.01 & 0.001 & 0.2953 & 0.2961 & 0.0 & ~ \\ 
        100 & [3, 128, 64, 16, 1] & relu & 30 & 0.01 & 0.1 & 0.2956 & 0.2979 & 0.0 & ~ \\ 
        103 & [3, 128, 64, 16, 1] & relu & 30 & 0.1 & 0.001 & 0.6306 & 0.7283 & 0.0 & ~ \\ 
        109 & [3, 16, 32, 64, 16, 1] & swish & 10 & 0.0 & 0.001 & 0.003628 & 0.0047 & 0.0 \\ 
        112 & [3, 16, 32, 64, 16, 1] & swish & 10 & 0.0 & 0.1 & 0.00133 & 0.001161 & 0.0 \\ 
        115 & [3, 16, 32, 64, 16, 1] & swish & 10 & 0.01 & 0.001 & 0.2816 & 0.2676 & 0.0 \\ 
        118 & [3, 16, 32, 64, 16, 1] & swish & 10 & 0.01 & 0.1 & 0.273 & 0.2648 & 0.0 \\ 
        121 & [3, 16, 32, 64, 16, 1] & swish & 10 & 0.1 & 0.001 & 0.6134 & 0.7145 & 0.0 \\ 
        124 & [3, 16, 32, 64, 16, 1] & swish & 10 & 0.1 & 0.1 & 0.6147 & 0.7148 & 0.0 \\ 
        125 & [3, 16, 32, 64, 16, 1] & swish & 10 & 0.1 & 0.1 & 0.616 & 0.716 & 0.2 \\ 
        127 & [3, 16, 32, 64, 16, 1] & swish & 20 & 0.0 & 0.001 & 0.00361 & 0.00297 & 0.0 \\ 
        133 & [3, 16, 32, 64, 16, 1] & swish & 20 & 0.01 & 0.001 & 0.2896 & 0.2729 & 0.0 \\ 
        136 & [3, 16, 32, 64, 16, 1] & swish & 20 & 0.01 & 0.1 & 0.2895 & 0.2762 & 0.0 \\ 
        139 & [3, 16, 32, 64, 16, 1] & swish & 20 & 0.1 & 0.001 & 0.6115 & 0.7125 & 0.0 \\ 
        142 & [3, 16, 32, 64, 16, 1] & swish & 20 & 0.1 & 0.1 & 0.6117 & 0.7131 & 0.0 \\ 
        143 & [3, 16, 32, 64, 16, 1] & swish & 20 & 0.1 & 0.1 & 0.612 & 0.7134 & 0.2 \\ 
        145 & [3, 16, 32, 64, 16, 1] & swish & 30 & 0.0 & 0.001 & 0.004226 & 0.00323 & 0.0 \\ 
        148 & [3, 16, 32, 64, 16, 1] & swish & 30 & 0.0 & 0.1 & 0.00143 & 0.001233 & 0.0 \\ 
        151 & [3, 16, 32, 64, 16, 1] & swish & 30 & 0.01 & 0.001 & 0.2835 & 0.2675 & 0.0 \\ 
        154 & [3, 16, 32, 64, 16, 1] & swish & 30 & 0.01 & 0.1 & 0.2746 & 0.2653 & 0.0 \\ 
        157 & [3, 16, 32, 64, 16, 1] & swish & 30 & 0.1 & 0.001 & 0.6128 & 0.7127 & 0.0 \\ 
        160 & [3, 16, 32, 64, 16, 1] & swish & 30 & 0.1 & 0.1 & 0.6143 & 0.7151 & 0.0 \\ 
        161 & [3, 16, 32, 64, 16, 1] & swish & 30 & 0.1 & 0.1 & 0.6175 & 0.7139 & 0.2 \\ 
        162 & [3, 16, 32, 64, 16, 1] & swish & 30 & 0.1 & 0.1 & 0.6177 & 0.7141 & 0.5 \\ 
        163 & [3, 32, 64, 32, 16, 1] & swish & 10 & 0.0 & 0.001 & 0.00216 & 0.001807 & 0.0 \\ 
        169 & [3, 32, 64, 32, 16, 1] & swish & 10 & 0.01 & 0.001 & 0.2847 & 0.26995 & 0.0 \\ 
        172 & [3, 32, 64, 32, 16, 1] & swish & 10 & 0.01 & 0.1 & 0.2723 & 0.2639 & 0.0 \\ 
        175 & [3, 32, 64, 32, 16, 1] & swish & 10 & 0.1 & 0.001 & 0.6135 & 0.7145 & 0.0 \\ 
        178 & [3, 32, 64, 32, 16, 1] & swish & 10 & 0.1 & 0.1 & 0.6153 & 0.7154 & 0.0 \\ 
        179 & [3, 32, 64, 32, 16, 1] & swish & 10 & 0.1 & 0.1 & 0.6152 & 0.7153 & 0.2 \\ 
        181 & [3, 32, 64, 32, 16, 1] & swish & 20 & 0.0 & 0.001 & 0.003022 & 0.002308 & 0.0 \\ 
        187 & [3, 32, 64, 32, 16, 1] & swish & 20 & 0.01 & 0.001 & 0.288 & 0.2727 & 0.0 \\ 
        190 & [3, 32, 64, 32, 16, 1] & swish & 20 & 0.01 & 0.1 & 0.5205 & 0.5999 & 0.0 \\ 
        193 & [3, 32, 64, 32, 16, 1] & swish & 20 & 0.1 & 0.001 & 0.6165 & 0.720 & 0.0 \\ 
        196 & [3, 32, 64, 32, 16, 1] & swish & 20 & 0.1 & 0.1 & 0.6133 & 0.7134 & 0.0 \\ 
        197 & [3, 32, 64, 32, 16, 1] & swish & 20 & 0.1 & 0.1 & 0.6134 & 0.7136 & 0.2 \\ 
        198 & [3, 32, 64, 32, 16, 1] & swish & 20 & 0.1 & 0.1 & 0.6133 & 0.7134 & 0.5 \\ 
        199 & [3, 32, 64, 32, 16, 1] & swish & 30 & 0.0 & 0.001 & 0.003787 & 0.002967 & 0.0 \\ 
        205 & [3, 32, 64, 32, 16, 1] & swish & 30 & 0.01 & 0.001 & 0.2995 & 0.2962 & 0.0 \\ 
        208 & [3, 32, 64, 32, 16, 1] & swish & 30 & 0.01 & 0.1 & 0.5216 & 0.6039 & 0.0 \\ 
        211 & [3, 32, 64, 32, 16, 1] & swish & 30 & 0.1 & 0.001 & 0.6175 & 0.7194 & 0.0 \\ 
        214 & [3, 32, 64, 32, 16, 1] & swish & 30 & 0.1 & 0.1 & 0.6174 & 0.7138 & 0.0 \\ 
        215 & [3, 32, 64, 32, 16, 1] & swish & 30 & 0.1 & 0.1 & 0.6178 & 0.7142 & 0.2 \\ 
\end{array}
\end{aligned}$$
"

# ╔═╡ 95557f25-0218-44e4-bd8f-03e940b4a548
md"De esta exploración podemos notar que las técnicas de regularización no mejoran el entrenamiento porque no estamos llegando al overfitting, además el dropout por alguna razón dificulta mucho el proceso de entrenamiento llegando al punto en que el cluster termina los procesos antes de terminar perdinendo muchos de los entrenamientos.

Sobre resultados los loss mas bajos que obtivimos se muestran en la siguiente tabla:

Al haber sido entrenado durante 500 épocas debería haber llegado el punto en el que no obtengan mejores reslutados los learning rates altos al converger mas rápido como se observa en las arquitecturas 1 y 19 donde se obtienen resultados similares.
"

# ╔═╡ c0c1e4d0-7b58-4dec-b39f-8270bb2a8f8f
md"$$\begin{aligned}
& \begin{array}{|c|c|c|c|c|c|c|c|c|}
    \hline
    \text{ID} & \text{Arq} & \text{Activ} & \text{BatchS} & \lambda & \eta & \text{LossTrain} & \text{LossValid} & \text{DP} \\
    \hline
    40 & [3, 32, 32, 16, 1] & \text{relu} & 30 & 0.0 & 0.1 & 0.0008262 & 0.0007238 & 0.0 \\
    4 & [3, 32, 32, 16, 1] & \text{relu} & 10 & 0.0 & 0.1 & 0.001226 & 0.0009682 & 0.0 \\
    112 & [3, 16, 32, 64, 16, 1] & \text{swish} & 10 & 0.0 & 0.1 & 0.001330 & 0.001161 & 0.0 \\
    1 & [3, 32, 32, 16, 1] & \text{relu} & 10 & 0.0 & 0.001 & 0.001891 & 0.001631 & 0.0 \\
    19 & [3, 32, 32, 16, 1] & \text{relu} & 20 & 0.0 & 0.001 & 0.002078 & 0.001770 & 0.0 \\
    \hline
\end{array}
\end{aligned}$$"

# ╔═╡ bf45e496-8a4c-4305-8a24-42908e2cec20
md"El hecho de que la red 40 tenga tan bajo loss para ambos conjuntos de señales es algo llamativo así que voy a intentar realizar una exploración quedandome con estas 5 redes para una cantidad mas amplia de señales que serían:


- $\sigma = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]$

- $l_{cm} = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9]~(\text{entrenamiento})$

- $l_{cm} = [0.54, 1.29, 2.04, 2.79, 3.54, 4.29, 5.04, 5.79]~(\text{validación})$

- $l_{cm} = [0.62, 1.37, 2.12, 2.87, 3.62, 4.37, 5.12, 5.87]~(\text{test})$

Con esto abarcaríamos casi todo el espectro de señales simuladas:"

# ╔═╡ b6054a4f-8791-4358-ba72-f96b06464b8c
PlutoUI.Resource("https://imgur.com/xr4k94j.png")

# ╔═╡ 46977020-2101-46b8-9191-e4cda127ab71
PlutoUI.Resource("https://imgur.com/OHX3mux.png")

# ╔═╡ ff1a4f26-02ce-4840-8aae-120221319217
md"Algo que también me sorprendió fueron las predicciones sin haber entrenado con estos nuevos datos la red 40, que si bien no son excelentes predicciones tampoco parecen estar tan lejos de converger a una buena predicción."

# ╔═╡ 0b33676a-50bb-4dd9-8456-e5e2372cf2ca
md"## Prediccion inicial para algunas señales de entrenamiento"

# ╔═╡ 837e89c7-19e6-4768-927f-4899039253d7
PlutoUI.Resource("https://imgur.com/2STFQOX.png")

# ╔═╡ 17b816fb-11a5-4d14-894e-5cf4ce8edfb4
PlutoUI.Resource("https://imgur.com/yz7tvAA.png")

# ╔═╡ 67acee35-5e3e-4c9b-af11-62914966a574
md"## Predicción inicial para las señales de validación"

# ╔═╡ 0845bdf6-63ba-43e6-809a-42639fadc3d2
PlutoUI.Resource("https://imgur.com/KdTIFgr.png")

# ╔═╡ 5f2e7189-b647-4776-a5e9-ea53679af510
PlutoUI.Resource("https://imgur.com/qH2ErRU.png")

# ╔═╡ ab4160cd-64d8-412a-9887-db144efb8151
md"## Predicción inicial para las señales de test"

# ╔═╡ 0f9e8ef9-5cc8-4c3d-9e29-09ae65729e0d
PlutoUI.Resource("https://imgur.com/evF1Bn8.png")

# ╔═╡ 6cfb6cee-3637-49c9-a53e-4b38fb3364b8
PlutoUI.Resource("https://imgur.com/SqcQ1cY.png")

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

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
# ╠═bb3273bf-1387-45f6-83dc-5c4d734e4ed5
# ╠═bd5741c9-2c04-479a-8a66-ca374c66a758
# ╟─6770b2db-b083-432a-8f7c-63822df3e22b
# ╟─22c5fa8b-aa4d-4c1e-83bf-057863559d0f
# ╠═2af67fbb-53f1-42ba-9dd0-45eece09efea
# ╟─56087631-4282-4aa5-a83b-cde277924d68
# ╟─95557f25-0218-44e4-bd8f-03e940b4a548
# ╟─c0c1e4d0-7b58-4dec-b39f-8270bb2a8f8f
# ╟─bf45e496-8a4c-4305-8a24-42908e2cec20
# ╟─b6054a4f-8791-4358-ba72-f96b06464b8c
# ╟─46977020-2101-46b8-9191-e4cda127ab71
# ╟─ff1a4f26-02ce-4840-8aae-120221319217
# ╟─0b33676a-50bb-4dd9-8456-e5e2372cf2ca
# ╟─837e89c7-19e6-4768-927f-4899039253d7
# ╟─17b816fb-11a5-4d14-894e-5cf4ce8edfb4
# ╟─67acee35-5e3e-4c9b-af11-62914966a574
# ╟─0845bdf6-63ba-43e6-809a-42639fadc3d2
# ╟─5f2e7189-b647-4776-a5e9-ea53679af510
# ╟─ab4160cd-64d8-412a-9887-db144efb8151
# ╟─0f9e8ef9-5cc8-4c3d-9e29-09ae65729e0d
# ╟─6cfb6cee-3637-49c9-a53e-4b38fb3364b8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
