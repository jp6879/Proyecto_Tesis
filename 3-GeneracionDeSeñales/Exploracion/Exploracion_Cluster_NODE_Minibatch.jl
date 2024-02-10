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
    
path_read = "/home/juan.morales/datos_PCA"

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
    dataSignals = Float32.(GetSignals(path_read))
    Signals_test = Float32.(Matrix(dataSignals[:,rango]'))
    Signals_test_short = Signals_test[:,1:muestreo_corto:1000]
    Signals_test_long = Signals_test[:,1001:muestreo_largo:end]
    Signals_test = hcat(Signals_test_short, Signals_test_long)
    return Signals_test
end

###################################################################################
# Función que idientifca las señales que se van a utilizar para el entrenamiento de la NODE con lcm y σ

function Get_Signals_Test_Parameters(rango,lcms,sigmas)
    dim1 = dimlcm = length(lcms)
    dim2 = dimσ = length(sigmas)

    column_lcm = zeros(dim1*dim2)
    column_σs = zeros(dim1*dim2)
    aux_lcm = collect(lcms)
    aux_σs = collect(sigmas)

    for i in 1:dim1
        for j in 1:dim2
            column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
            column_σs[(i - 1)*dim2 + j] = aux_σs[j]
        end
    end

    column_lcm_test = column_lcm[rango]
    column_sigmass_test = column_σs[rango]
    
    return column_lcm_test, column_sigmass_test
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
    Signals_test = Get_Signals_Test(rango, path_read, muestreo_corto, muestreo_largo)
    column_lcm_test, column_sigmass_test = Get_Signals_Test_Parameters(rango,lcms,sigmas)
    Signals_test_derivadas = zeros(size(Signals_test))
    for i in 1:size(Signals_test)[1]
        Signals_test_derivadas[i,:] = derivate_signals(t,Signals_test[i,:])
    end
    Signals_test_derivadas = Float32.(Matrix(Signals_test_derivadas'))
    for i in 1:size(Signals_test)[1]
        Signals_test_derivadas[:,i] = Signals_test_derivadas[:,i] ./ maximum(abs.(Signals_test_derivadas[:,i]))
    end
    return Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmass_test
end

###################################################################################


# Voy a tomar 100 se�ales elegidas tomando lcm de 5 en 5 hasta 5.45 �m y manteniendo s = 1.0
# rango = 100:500:50000
# Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmass_test = Get_Signals_Data_Training(path_read, rango, lcms, sigmas, muestreo_corto, muestreo_largo)

# tforecast = t[23:end]
# t = t[1:22]

# Signals_test_train = Signals_test[:,1:29]
# Signals_test_valid = Signals_test[:,30:end]

# Signals_test_derivadas_train = Signals_test_derivadas[1:29,:]
# Signals_test_derivadas_valid = Signals_test_derivadas[30:end,:]

# println(size(Signals_test))


###################################################################################

# Voy a establecer un rango saltando lcm de 5 en 5 hasta 5.45 μm y manteniendo σ = 1.0
# rango = 100:500:50000
# Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmass_test = Get_Signals_Data_Training(path_read, rango, lcms, sigmas, muestreo_corto, muestreo_largo)

# # using Plots
# # Signals_test

# pl = plot(t, Signals_test[1,:], label = "l = $(column_lcm_test[1]), σ = $(column_sigmass_test[1])", xlabel = "t", ylabel = "S(t)", title = "Señales de Hahn", lw = 2, color = :blue)

# constant = ones(length(range(-0.05, 1.1, length = 1000)))*t[25]

# for i in range(2, size(Signals_test)[1], step = 10)
#     plot!(pl, t, Signals_test[i,:], label = "l = $(column_lcm_test[i]), σ = $(column_sigmass_test[i])", xlabel = "t", ylabel = "S(t)", title = "Señales de Hahn", lw = 2)
# end

# plot!(constant, range(-0.05, 1.1, length = 1000), label = false, lw = 2, ls = :dash, color = :gray)

# pl


###################################################################################

# Función que crea el modelo de la red neuronal que va a estar dentro de la ODE
function create_model(layers::Vector{Int}, activation)
    activations = [activation for i in 1:length(layers) - 2]
    return Chain([Dense(layers[i], layers[i+1], activations[i]) for i in 1:length(layers) - 2]..., Dense(layers[end-1], layers[end]))
end

###################################################################################

# Función que entrena la NODE con mini-batchs
function Train_Neural_ODE(nn, U0, extra_parameters ,num_epochs, train_loader, opt, eta, Signals, Signals_forecast, t, tforecast)
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

    # Función de pérdida que vamos a minimizar, recibe un batch de señales y un batch de tiempos
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

# Función principal con la exploración de los parámetros

function main()
    # Arquitecturas que vamos a utilizar
    architectures = [
        [[2, 8, 1], relu], # Una capa oculta
        # [[2, 8, 1], tanh], # Misma con activación tanh
        # [[2, 8, 1], swish], # Misma con activación swish
        
        # [[2, 16, 16, 1], relu], # Dos capas ocultas
        # [[2, 16, 16, 1], tanh], # Misma con activación tanh
        # [[2, 16, 16, 1], swish], # Misma con activación swish
        
        # [[2, 32, 64, 16, 1], relu], # Tres capas ocultas
        # [[2, 32, 64, 16, 1], tanh], # Misma con activación tanh
        # [[2, 32, 64, 16, 1], swish], # Misma con activación swish
        
        # [[2, 32, 64, 64, 32, 16, 1], relu], # Cinco capas ocultas
        # [[2, 32, 64, 64, 32, 16, 1], tanh], # Misma con activación tanh
        # [[2, 32, 64, 64, 32, 16, 1], swish], # Misma con activación swish
        ]

    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [AdamW]]

    # Numero de mini-batchs que vamos a utilizar
    baths_size = [5, 10]

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

    tforecast = t[23:end]
    t = t[1:22]

    Signals_test_train = Signals_test[:,1:22]
    Signals_test_valid = Signals_test[:,23:end]

    Signals_test_derivadas_train = Signals_test_derivadas[1:22,:]

    # Tomamos un learning rate de 0.005
    eta = 5e-3

    # Vamos a tomar 2000 épocas para entrenar todas las arquitecturas
    epochs = 2

    # Todas las señales tienen la misma condición inicial U0 = 1
    U0 = ones32(size(Signals_test)[1])

    losses = []
    losses_forecast = []
    thetas = []

    id = 1

    id_column = []
    layers_column = []
    activation_column = []
    optimizer_column = []
    loss_final_train_column = []
    loss_final_forecast_column = []
    batch_size_column = []

    # Vamos a hacer un loop sobre las arquitecturas, los optimizadores y los tamaños de los mini-batchs
    for architecture in architectures
        for opt in optimizers
            for bath_size in baths_size
                id += 1

                string_id = string(id)
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
                train_loader = Flux.Data.DataLoader((Signals_test_train, t), batchsize = bath_size)
                # Vamos a crear el modelo de la red neuronal que va a estar dentro de la ODE
                nn = create_model(layers, activation)
                # Entrenamos una NODE con mini-batchs para cada arquitectura, optimizador y tamaño de mini-batch y guardamos el loss y los parametros de la red neuronal
                architecture_loss, theta, loss_forecast = Train_Neural_ODE(nn, U0, Signals_test_derivadas_train, epochs, train_loader, opt, eta, Signals_test_train, Signals_test_valid, t, tforecast)

                push!(id_column, id)
                push!(layers_column, layers)
                push!(activation_column, activation_string)
                push!(optimizer_column, opt_string)
                push!(loss_final_train_column, architecture_loss[end])
                push!(loss_final_forecast_column, loss_forecast[end])
                push!(losses, architecture_loss)
                push!(losses_forecast, loss_forecast)
                push!(batch_size_column, bath_size)
                push!(thetas, theta)
            end
        end
    end

    # Guardamos los resultados en el dataframe
    df_results_total = DataFrame(ID = id_column, Layers = layers_column, Activation = activation_column, Optimizer = optimizer_column, Loss_Train = loss_final_train_column, Loss_forecast = loss_final_forecast_column, batchsize = batch_size_column)

    existing_csv_file = "/home/juan.morales/Testing_NODE_batchs/Prediction_results/Cl_Registro_arquitecturasNODE.csv"
    if isfile(existing_csv_file)
        df_old = CSV.read(existing_csv_file, DataFrame)
        df_results_total = vcat(df_old, df_results)
    end
    CSV.write("/home/juan.morales/Testing_NODE_batchs/Prediction_results/Cl_Registro_arquitecturasNODE.csv", df_results_total)

    # Guardamos los loss en un dataframe

    Loss_Matrix = zeros(length(losses[1]), length(losses))
    Loss_Matrix_forecast = zeros(length(losses_forecast[1]), length(losses_forecast))
    for i in 1:length(losses)
        Loss_Matrix[:,i] = losses[i]
        Loss_Matrix_forecast[:,i] = losses_forecast[i]
    end
    df_losses = DataFrame(Loss_Matrix, :auto)
    df_losses_forecasts = DataFrame(Loss_Matrix_forecast, :auto)

    # Cambiamos el nombre predeterminado por el id de la arquitectura
    for i in 1:(id-2)
        rename!(df_losses, Symbol("x$i") => Symbol("id $(i)"))
    end

    # Nos fijmamos si habia un archivo previo con las exploraciones, si existe lo leemos y lo concatenamos con el nuevo, sinó creamos un nuevo archivo
    existing_csv_file = "/home/juan.morales/Testing_NODE_batchs/Loss_list/Cl_Losses_arquitecturasNODE.csv"
    if isfile(existing_csv_file)
        df_old = CSV.read(existing_csv_file, DataFrame)
        df_losses = vcat(df_old, df_losses)
    end

    CSV.write("/home/juan.morales/Testing_NODE_batchs/Loss_list/Cl_Losses_arquitecturasNODE.csv", df_losses)

    exisiting_csv_file_forecast = "/home/juan.morales/Testing_NODE_batchs/Loss_list/Cl_Losses_forecast_arquitecturasNODE.csv"
    if isfile(exisiting_csv_file_forecast)
        df_old_forecast = CSV.read(exisiting_csv_file_forecast, DataFrame)
        df_losses_forecasts = vcat(df_old_forecast, df_losses_forecasts)
    end

    CSV.write("/home/juan.morales/Testing_NODE_batchs/Loss_list/Cl_Losses_forecast_arquitecturasNODE.csv", df_losses_forecasts)

    # Guardamos los parametros de la red neuronal en distintos DataFrames
    for i in 1:length(thetas)
        df_theta = DataFrame(reshape(thetas[i], length(thetas[i]), 1), :auto)
        CSV.write("/home/juan.morales/Testing_NODE_batchs/Parameters/Cl_Parameters_arquitectura_$(i).csv", df_theta)
    end
end

main()