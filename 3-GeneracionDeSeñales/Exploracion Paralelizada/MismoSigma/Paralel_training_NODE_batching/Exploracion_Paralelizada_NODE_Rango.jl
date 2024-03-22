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
    
# Concatenate t_short and t_long
t = vcat(t_short, t_long)
    
# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto = 50 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 4
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]
    
t = vcat(t_short, t_long)

# Seleccionamos el path según sea para el cluster o para la computadora personal
# path_read = "/home/juan.morales/datos_PCA"
path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

# Parametros que se varian
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
sigmas = 0.01:0.01:1
###################################################################################

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
end

###################################################################################

# Función que crea el modelo de la red neuronal que va a estar dentro de la ODE
function create_model(layers::Vector{Int}, activation)
    # Creamos la red neuronal con las capas y la activación que nos pasan
    activations = [activation for i in 1:length(layers) - 2]
    # Regresamos la red neuronal
    return Chain([Dense(layers[i], layers[i+1], activations[i]) for i in 1:length(layers) - 2]..., Dense(layers[end-1], layers[end]))
end

###################################################################################

# Función que entrena la NODE con mini-batchs
function Train_Neural_ODE(nn, U0, extra_parameters ,num_epochs, train_loader, opt, eta, Signals, Signals_forecast, t, tforecast)
    # Tiempo sobre el cual resolver
    tspan = (0f0, 1f0)

    # Parametrizar indices para poder acceder a los parametros extras de la red neuronal
    f(x,p) = round(Int, x * (length(p) - 1)) + 1

    # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial
    p, re = Flux.destructure(nn) 

    # Optimizardor
    opt = opt(eta)

    # Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
    function predict_NeuralODE(u0, parametros, time_batch)
        # dSdt = NN(S, parametros_extra) 
        function dSdt(u, p, t; parametros_extra = parametros)
            indx = f(t, parametros)
             # Selecciona los parametros extra en el tiempo t
            parametros_actuales = parametros[indx]
            # Concatena los el valor de S(t) con los parametros extra en el tiempo t
            entrada_red = vcat(u, parametros_actuales)
            # Regresa la salida de la red neuronal re creada con los parámetros p
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

        [[2, 50, 64, 16, 1], relu], # Tres capas ocultas con mas neuronas
        [[2, 50, 64, 16, 1], tanh], # Misma con activación tanh
        [[2, 50, 64, 16, 1], swish], # Misma con activación swish
        
        [[2, 32, 64, 64, 32, 16, 1], relu], # Cinco capas ocultas
        [[2, 32, 64, 64, 32, 16, 1], tanh], # Misma con activación tanh
        [[2, 32, 64, 64, 32, 16, 1], swish], # Misma con activación swish
        ]

    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [AdamW, RMSProp]]

    # Numero de mini-batchs que vamos a utilizar
    batchs_size = [5, 10]

    # Vector de configuraciones que vamos a utilizar
    configuraciones = []

    # Guardamos todas las configuraciones en un vector que va a ser el que iteramos con el cluster
    for arch in architectures
        for opt in optimizers
            for batch_size in batchs_size
                push!(configuraciones, (arch, opt, batch_size))
            end
        end
    end

    path_read = "/home/juan.morales/datos_PCA"

    t_short = collect(range(0, 0.1, length = 1000))
    t_long = collect(range(0.1, 1, length = 100))
    
    # Concatenate t_short and t_long
    t = vcat(t_short, t_long)
    
    # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
    muestreo_corto = 50 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
    muestreo_largo = 4
    t_short = t_short[1:muestreo_corto:end]
    t_long = t_long[1:muestreo_largo:end]
    
    t = vcat(t_short, t_long)

    # Voy a tomar 100 señales elegidas tomando lcm de 5 en 5 hasta 5.45 μm y manteniendo σ = 1.0
    rango = 100:500:50000
    Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmass_test = Get_Signals_Data_Training(path_read, rango, lcms, sigmas, muestreo_corto, muestreo_largo)

    # Tiempo desde el cual vamos a hacer la predicción
    idxforecast = 23

    tforecast = t[idxforecast:end]
    ttrain = t[1:idxforecast-1]

    Signals_test_train = Signals_test[:,1:22]
    Signals_test_valid = Signals_test[:,23:end]

    Signals_test_derivadas_train = Signals_test_derivadas[1:22,:]

    # Tomamos un learning rate de 0.005
    eta = 5e-3

    # Vamos a tomar 1500 épocas para entrenar todas las arquitecturas
    epochs = 1500

    # Todas las señales tienen la misma condición inicial U0 = 1
    U0 = ones32(size(Signals_test)[1])

    # Vamos a iterar sobre todas las configuraciones y vamos a entrenar una NODE con mini-batchs para cada una de ellas
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
    train_loader = Flux.Data.DataLoader((Signals_test_train, ttrain), batchsize = batch_size)

    # Vamos a crear el modelo de la red neuronal que va a estar dentro de la ODE
    nn = create_model(layers, activation)
    
    # Entrenamos una NODE con mini-batchs para cada arquitectura, optimizador y tamaño de mini-batch y guardamos el loss y los parametros de la red neuronal
    architecture_loss, theta, loss_forecast = Train_Neural_ODE(nn, U0, Signals_test_derivadas_train, epochs, train_loader, opt, eta, Signals_test_train, Signals_test_valid, ttrain, tforecast)

    # println("Arquitectura: $architecture", " || Optimizador: $opt", " || Tamaño de mini-batch: $batch_size", " || Loss: $(architecture_loss[end])", " || Loss Forecast: $(loss_forecast[end])")

    actual_id = parse(Int128,ARGS[1])
    actual_layer = string(layers)
    actual_activation = activation_string
    actual_optimizer = opt_string
    actual_loss_final_train = architecture_loss[end]
    actual_loss_final_forecast = loss_forecast[end]
    actual_batch_size = batch_size

    # Guardamos los resultados en un archivo csv

    df_results_total = DataFrame(ID = actual_id, Arquitectura = actual_layer, Activación = actual_activation, Optimizador = actual_optimizer, Batch_Size = actual_batch_size, Loss_Final_Entrenamiento = actual_loss_final_train, Loss_Final_Predicción = actual_loss_final_forecast)

    CSV.write("/home/juan.morales/Paralel_training_NODE_batching/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    
    # Guardamos los loss y los parametros de la red neuronal en archivos csv

    Loss_Matrix = zeros((length(architecture_loss), 2))

    for i in 1:length(architecture_loss)
        Loss_Matrix[i,1] = architecture_loss[i]
        Loss_Matrix[i,2] = loss_forecast[i]
    end

    df_losses = DataFrame(Loss_Matrix, :auto)

    rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
    rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

    CSV.write("/home/juan.morales/Paralel_training_NODE_batching/Losses/$(actual_id)_losses.csv", df_losses)

    df_theta = DataFrame(reshape(theta, length(theta), 1), :auto)

    CSV.write("/home/juan.morales/Paralel_training_NODE_batching/Parameters/$(actual_id)_Parameters.csv", df_theta)

end

main()

#######################################################################################################