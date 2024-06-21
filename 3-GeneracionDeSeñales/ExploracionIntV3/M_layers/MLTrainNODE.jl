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