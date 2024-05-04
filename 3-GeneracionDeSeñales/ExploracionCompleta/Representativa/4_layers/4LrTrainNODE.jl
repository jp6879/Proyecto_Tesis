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
    
## Testing ##
# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 10
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]
    
t = vcat(t_short, t_long)

path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"
# path_read = "/home/juan.morales/datos_PCA"

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

###################################################################################

# Función que crea el modelo de la red neuronal que va a estar dentro de la ODE
function create_model(layers::Vector{Int}, activation)
    activations = [activation for i in 1:length(layers) - 2]
    return Chain([Dense(layers[i], layers[i+1], activations[i]) for i in 1:length(layers) - 2]..., Dense(layers[end-1], layers[end]))
end

###################################################################################

# Función que entrena la NODE con mini-batchs
function Train_Neural_ODE(nn, U0, extra_parameters, extra_parameters_valid ,num_epochs, train_loader, opt, eta, Signals, Signals_forecast, t, tforecast, lamb, actual_id)
    
    # Tiempo sobre el cual resolver
    tspan = (0f0, 1f0)

    # Parametrizar indices para poder acceder a los parametros extras de la red neuronal
    f(x,p) = round(Int, x * (length(p) - 1)) + 1
    
    # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial
    p, re = Flux.destructure(nn) 

    # Si existe el archivo con los parámetros de la red previamente los cargamos
    if isfile("/home/juan.morales/ExploracionCompleta/Mini/4_layers/Parameters/$(actual_id)_Parameters.csv")
        theta = CSV.read("/home/juan.morales/ExploracionCompleta/Mini/4_layers/Parameters/$(actual_id)_Parameters.csv", DataFrame)
        p = Float32.(theta[:,1])
    end

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

        return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-9, abstol = 1e-9)) # Regresa la solución de la ODE
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
        return Flux.mse(y, batch') #+ lamb * (penalization_term(time_batch, y))
    end
    
    # Función de Loss de validación, acá usamos las derivadas de la señal con pocos puntos
    function loss_node_valid(batch, time_batch)
        y = Predict_Singals(U0, extra_parameters_valid, time_batch)
        return Flux.mse(y, batch') #+ lamb * (penalization_term(time_batch, y))
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
            forecast_loss = loss_node_valid(Signals_forecast, tforecast)
            println("Epoch = $epoch || Loss: $actual_loss || Loss valid: $forecast_loss")
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
        [[2, 32, 64, 32, 16, 1], relu], # Cuatro capas ocultas
        [[2, 32, 64, 32, 16, 1], tanh_fast], # Misma con activación tanh_fast
        [[2, 32, 64, 32, 16, 1], swish], # Misma con activación swish

        [[2, 128, 64, 32, 16, 1], relu], # Cuatro capas ocultas con mas neuronas
        [[2, 128, 64, 32, 16, 1], tanh_fast], # Misma con activación tanh_fast
        [[2, 128, 64, 32, 16, 1], swish], # Misma con activación swish
        ]

    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [AdamW]]

    # Numero de mini-batchs que vamos a utilizar 
    batchs_size = [15, 30] 

    # Vector de configuraciones que vamos a utilizar

    configuraciones = []

    for arch in architectures
        for opt in optimizers
            for batch_size in batchs_size
                push!(configuraciones, (arch, opt, batch_size))
            end
        end
    end

    # path_read = "/home/juan.morales/datos_PCA"
    path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

    t_short = collect(range(0, 0.1, length = 1000))
    t_long = collect(range(0.1, 1, length = 100))
        
    # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
    muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
    muestreo_largo = 10

    # Esto da 100 tiempos 50 puntos desde 0 a 0.1 y 25 puntos desde 0.1 a 1
    t_short = t_short[1:muestreo_corto:end]
    t_long = t_long[1:muestreo_largo:end]
    
    t = vcat(t_short, t_long)

    # Tomamos 1 sigma y 5 tamaños de compartimientos para cada sigma o sea 60 señales
    sampled_sigmas = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    lcm_range = 1:75:600
    
    # Obtenemos las señales representativas para un conjunto de sigmas y lcms
    Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

    # Numero de puntos en el conjunto de validación
    n_valid = 10
    
    # Paso para tomar los tiempos y señales de entrenamiento y de validación
    step = floor(Int, length(t) / n_valid) + 1

    # Tomamos los tiempos de entrenamiento y de validacion
    tvalid = t[1:step:end]
    ttrain = [t for t in t if t ∉ tvalid]

    indexes_train = [i for i in 1:length(t) if t[i] in ttrain]
    indexes_valid = [i for i in 1:length(t) if t[i] in tvalid]

    # Puntos de las señales a predecir
    Signals_valid = Signals_rep[:,indexes_valid]

    # Derivadas de las señales de entrenamiento que son sacadas de los 10 puntos de validación
    Signals_derivadas_valid = get_signals_deriv_valid(tvalid, Signals_valid)

    # Tomamos un learning rate de 0.005
    eta = 5e-3

    # Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
    epochs = 100

    # Todas las señales tienen la misma condición inicial U0 = 1
    U0 = ones32(size(Signals_rep)[1])

    # Para el entrenamiento en el cluster vamos iterando sobre las configuraciones y guardamos los resultados en archivos csv
    
    architecture, opt, batch_size = configuraciones[1]
    # architecture, opt, batch_size = configuraciones[parse(Int128,ARGS[1])]

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
    nn = create_model(layers, activation)
    
    # Parámetro de penalización
    lambd = 0.1

    # println("Arquitectura: $architecture", " || Optimizador: $opt", " || Tamaño de mini-batch: $batc_size", " || Loss: $(architecture_loss[end])", " || Loss Forecast: $(loss_forecast[end])")

    # Número de modelo
    actual_id = 1
    # actual_id = parse(Int128,ARGS[1])

    # Entrenamos una NODE con mini-batchs para cada arquitectura, optimizador y tamaño de mini-batch y guardamos el loss y los parametros de la red neuronal
    architecture_loss, theta, loss_forecast = Train_Neural_ODE(nn, U0, Signals_derivadas_valid, Signals_derivadas_valid, epochs, train_loader, opt, eta, Signals_rep, Signals_valid, t, tvalid, lambd, actual_id)

    # Guardamos los hiperparámetros del entrenamiento y el loss final
    actual_layer = string(layers)
    actual_activation = activation_string
    actual_optimizer = opt_string
    actual_batch_size = batch_size
    actual_loss_final_train = architecture_loss[end]
    actual_loss_final_forecast = loss_forecast[end]

    # Guardamos los resultados en un archivo csv
    df_results_total = DataFrame(ID = actual_id, Arquitectura = actual_layer, Activación = actual_activation, Optimizador = actual_optimizer, Batch_Size = actual_batch_size, Loss_Final_Entrenamiento = actual_loss_final_train, Loss_Final_Predicción = actual_loss_final_forecast)

    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    CSV.write("/home/juan.morales/ExploracionCompleta/Mini/4_layers/Resultados/$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size).csv", df_results_total)
    
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
    if isfile("/home/juan.morales/ExploracionCompleta/Mini/4_layers/Losses/$(actual_id)_losses.csv")
        df_losses = vcat(CSV.read("/home/juan.morales/ExploracionCompleta/Mini/4_layers/Losses/$(actual_id)_losses.csv", DataFrame), df_losses)
    end

    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Losses/$(actual_id)_losses.csv", df_losses)
    CSV.write("/home/juan.morales/ExploracionCompleta/Mini/4_layers/Losses/$(actual_id)_losses.csv", df_losses)
    
    df_theta = DataFrame(reshape(theta, length(theta), 1), :auto)
    # CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/RepresentativeTrain_NODE/Parameters/$(actual_id)_Parameters.csv", df_losses)
    CSV.write("/home/juan.morales/ExploracionCompleta/Mini/4_layers/Parameters/$(actual_id)_Parameters.csv", df_theta)
end

main()