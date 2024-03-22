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

t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))
    
# # Concatenate t_short and t_long
# t = vcat(t_short, t_long)
    
# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 4
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]
    
t = vcat(t_short, t_long)

# path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"
path_read = "/home/juan.morales/datos_PCA"

# Parametros que se varian
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
sigmas = 0.01:0.01:1

# sampled_sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1]
# lcm_range = 1:25:250
# Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

# pl = plot(t, Signals_rep[1,:], label = "lcm = $(column_lcm_rep[1]) sigma = $(column_sigmas_rep[1])")
# for i in 2:size(Signals_rep)[1]
#     plot!(pl, t, Signals_rep[i,:], label = false)
# end

# pl

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

# Función que crea el modelo de la red neuronal que va a estar dentro de la ODE
function create_model(layers::Vector{Int}, activation)
    activations = [activation for i in 1:length(layers) - 2]
    return Chain([Dense(layers[i], layers[i+1], activations[i]) for i in 1:length(layers) - 2]..., Dense(layers[end-1], layers[end]))
end

###################################################################################

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
        # y_forecasted = Predict_Singals(U0, extra_parameters, tforecast)
        # return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y) + penalization_term(tforecast, y_forecasted))
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

function main()
    # Arquitecturas que vamos a utilizar
    architectures = [
        [[2, 8, 1], relu], # Una capa oculta
        [[2, 8, 1], tanh], # Misma con activación tanh
        [[2, 8, 1], swish], # Misma con activación swish
        
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

        [[2, 128, 256, 32, 64, 32, 16, 8, 1], relu], # Siete capas ocultas
        [[2, 128, 256, 32, 64, 32, 16, 8, 1], tanh], # Misma con activación tanh
        [[2, 128, 256, 32, 64, 32, 16, 8, 1], swish], # Misma con activación swish
        ]

    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [AdamW, RMSProp]]

    # Numero de mini-batchs que vamos a utilizar 
    batchs_size = [30, 60] # En este caso cada bath es la mitda de una señal completa o la señal completa

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

    # Esto da 100 tiempos 50 puntos desde 0 a 0.1 y 25 puntos desde 0.1 a 1
    t_short = t_short[1:muestreo_corto:end]
    t_long = t_long[1:muestreo_largo:end]
    
    t = vcat(t_short, t_long)

    # Tomamos 6 sigmas y 10 tamaños de compartimientos para cada sigma o sea 60 señales
    sampled_sigmas = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    lcm_range = 1:25:125
    
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

    # Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
    epochs = 1000

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
    CSV.write("/home/juan.morales/Model1_Representative_trainingNODE/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    

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
    CSV.write("/home/juan.morales/Model1_Representative_trainingNODE/Losses/$(actual_id)_losses.csv", df_losses)
    

    df_theta = DataFrame(reshape(theta, length(theta), 1), :auto)
    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Parameters/$(actual_id)_Parameters.csv", df_losses)
    CSV.write("/home/juan.morales/Model1_Representative_trainingNODE/Parameters/$(actual_id)_Parameters.csv", df_theta)

end

main()


###################################TESTING#########################################
# using Plots

# t_short = collect(range(0, 0.1, length = 1000))
# t_long = collect(range(0.1, 1, length = 100))
    
# # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
# muestreo_corto = 100 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
# muestreo_largo = 2
# t_short = t_short[1:muestreo_corto:end]
# t_long = t_long[1:muestreo_largo:end]

# t = vcat(t_short, t_long)

# rango = 1:55100
# Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmass_test = Get_Signals_Data_Training(path_read, rango, lcms, sigmas, muestreo_corto, muestreo_largo, t)

# idx_forecast = 40

# tforecast = t[idx_forecast:end]
# ttrain = t[1:idx_forecast-1]

# Signals_test_train = Signals_test[:,1:idx_forecast-1]
# Signals_test_valid = Signals_test[:,idx_forecast:end]

# Signals_test_derivadas_train = Signals_test_derivadas[1:idx_forecast-1,:]
# Signals_test_derivadas_valid = Signals_test_derivadas[idx_forecast:end,:]

# size(Signals_test)[1]

##################################################################################
# using Plots

# pl = scatter(ttrain, Signals_test_train[1,:], label = "σ = $(column_sigmass_test[1]), lcm = $(column_lcm_test[1])", xlabel = "Tiempo", ylabel = "S")
# # scatter!(pl, tforecast, Signals_test_valid[1,:], label = "σ = $(column_sigmass_test[1]), lcm = $(column_lcm_test[1])")

# for i in 2:size(Signals_test)[1]
#     scatter!(pl, ttrain, Signals_test_train[i,:], label = false)
#     # scatter!(pl, tforecast, Signals_test_valid[i,:], label = false)
    
#     if i == round(size(Signals_test)[1]//2)
#         scatter!(pl, ttrain, Signals_test_train[i,:], label = "σ = $(column_sigmass_test[i]), lcm = $(column_lcm_test[i])")
#         # scatter!(pl, tforecast, Signals_test_valid[i,:], label = "σ = $(column_sigmass_test[i]), lcm = $(column_lcm_test[i])")
#     end

#     if i == size(Signals_test)[1]
#         scatter!(pl, ttrain, Signals_test_train[i,:], label = "σ = $(column_sigmass_test[i]), lcm = $(column_lcm_test[i])")
#         # scatter!(pl, tforecast, Signals_test_valid[i,:], label = "σ = $(column_sigmass_test[i]), lcm = $(column_lcm_test[i])")
#     end
# end

# pl

# scatter!(pl, ttrain, Signals_test_train[end,:], label = "Señal 2 entrenamiento")
# scatter!(pl, tforecast, Signals_test_valid[end,:], label = "Señal 2 predicciones")

# scatter!(pl, ttrain, Signals_test_train[end-5,:], label = "Señal 3 entrenamiento")
# scatter!(pl, tforecast, Signals_test_valid[end-20,:], label = "Señal 3 predicciones")

###################################################################################