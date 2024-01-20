# Programa que explora los parámetros de una red neuronal feedfoward para la identificación de lcm y σ a partir de señales en PCA de Hahn
# Se varían la arquitectura de la red, la función de activación y el optimizador, se analizan las redes con la métrica RMAE tanto globalmente como punto a punto de los datos PCA

# Paquetes necesarios
using Flux
using Statistics
using Flux: train!
using Plots
using Distributions
using ProgressMeter
using MultivariateStats
using DataFrames
using CSV
using StatsPlots
using LaTeXStrings
using LinearAlgebra
using PlotlyJS
using CUDA
using Random

############################################################################################################################################################################
# Traemos los mismos datos de los parametros utilizados para generar los datos, deberiamos hacer una función grande en la proxima función que genere los datos donde les pasamos
# Todos estos parámetros desde otro programa, como ahora generamos pocos datos me quedo con esto
# Esta exploración fue realizada con datos generados con una menor resolución tanto en lc como en t, es por estos que estos parámetros son distintos a los de los otros programas

###################################################ACORDARSE DE ESTO#################################################################
# Parámetros fijos
# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 2000
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 50

lcs = Float32.(collect(range(l0, lf, length = N)))

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.01:1

##########################################################################################

# Distribucion de probabilidad log-normal se puede utilizar para añadir a la función de costo final, toma demasiado tiempo.

function Pln(lcm::Float32, σ::Float32)
    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
end

##########################################################################################
# Metricas de validacion de la red neuronal, error medio absoluto el cual se divide por la media de los datos reales para obtener el error relativo

# Relative Mean Absolute Error
function RMAE(predicted, real)
    return mean(abs.(predicted .- real)) / mean(real)
end

##########################################################################################

# Regularizaciones L1 y L2 para la red neuronal
pen_l2(x::AbstractArray) = Float32.(sum(abs2, x) / 2)
pen_l1(x::AbstractArray) = Float32.(sum(abs, x) / 2)

##########################################################################################

# Función para crear los modelos de la red neuronal
function create_model(layers::Vector{Int}, activation)
    activations = [activation for i in 1:length(layers) - 2]
    return Chain([Dense(layers[i], layers[i+1], activations[i]) for i in 1:length(layers) - 2]..., Dense(layers[end-1], layers[end], softplus))
end

##########################################################################################

# Función para cargar los datos de entrenamiento, validacion
function load_data(x_train, y_train, x_valid, y_valid, batchsize::Int, shuffle::Bool)
    data = Flux.Data.DataLoader((x_train, y_train), batchsize = batchsize, shuffle = shuffle)
    data_valid = Flux.Data.DataLoader((x_valid, y_valid), batchsize = batchsize, shuffle = shuffle)
    return data, data_valid
end

##########################################################################################

# Función para entrenar la red neuronal

function train_model(model, id::String, epochs::Int, learning_rate, opt, data, data_valid)
    η = learning_rate

    if opt == ADAM
        opt = ADAM(η)
    elseif opt == Descent
        opt = Descent(η)
    elseif opt == RMSProp
        opt = RMSProp(η)
    end

    # Funciones de costo para utilizar
    
    # Función de costo MSE
    function loss_mse(x,y)
        y_hat = model(x)
        return Flux.mse(y_hat, y)
    end

    losses = []
    losses_valid = []
    
    # Parámetros de la red neuronal
    params = Flux.params(model)
    
    # Definimos una funcion de callback para ver el progreso del entrenamiento cada 1000 épocas
    global iter = 0
    cb = function()
        global iter += 1
        if iter % length(data) == 0
            epoch = iter ÷ length(data)
            actual_loss = loss_mse(data.data[1], data.data[2])
            actual_valid_loss = loss_mse(data_valid.data[1], data_valid.data[2])
            if epoch % 1000 == 0
                println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
            end
            push!(losses, actual_loss)
            push!(losses_valid, actual_valid_loss)
        end
    end;
    
    for epoch in 1:epochs
        train!(loss_mse, params, data, opt, cb = cb)
        if epoch % 500 == 0
            η = η * 0.2
            if opt == ADAM
                opt = ADAM(η)
            elseif opt == Descent
                opt = Descent(η)
            elseif opt == RMSProp
                opt = RMSProp(η)
            end
        end
    end

    # Guardamos la función costo en cada época
    save_loss(losses, "loss_train_"*id*".csv")
    save_loss(losses_valid, "loss_valid_"*id*".csv")

    # Evaluamos el modelo en los datos de entrenamiento y validación
    rmae_global_train = eval_model(model, data.data[1], data.data[2])
    rmae_global_valid = eval_model(model, data_valid.data[1], data_valid.data[2])

    rmae_scores_train = eval_model_point(model, data.data[1], data.data[2])
    rmae_scores_valid = eval_model_point(model, data_valid.data[1], data_valid.data[2])

    # Guardamos las predicciones del modelo y el RMAE en cada punto
    save_predictions(model(data.data[1]), rmae_scores_train, "predictions_train_"*id*".csv")
    save_predictions(model(data_valid.data[1]), rmae_scores_valid, "predictions_valid_"*id*".csv")

    # Devolvemos los RMAE globales para globales

    return rmae_global_train, rmae_global_valid

end

##########################################################################################
# Función que evalua el RMAE global

function eval_model(model, x, y)
    y_pred = model(x)
    rmae = RMAE(y_pred, y)
    return rmae
end

##########################################################################################
# Función que evalua el RMAE punto a punto

function eval_model_point(model, x, y)
    y_pred = model(x)
    N = length(y_pred[1,:])

    rmae_scores = zeros(N)

    for i in 1:N
        rmae_scores[i] = RMAE(y_pred[:,i], y[:,i])
    end
    
    return rmae_scores
end

##########################################################################################

# Funciones de guardado de datos

function save_loss(loss_vector, filename::String)
    CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\2-RedNeruronal_Identificacion_Hahn\\NN(S)_Params_Exploración\\Funciones_loss\\"*filename, DataFrame(loss = loss_vector))
end

# Función para guardar las predicciones del modelo y el RMAE en cada punto

function save_predictions(predictions, rmae_scores, filename::String)
    df = DataFrame(x1 = predictions[1,:], x2 = predictions[2,:], rmae_scores = rmae_scores)
    CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\2-RedNeruronal_Identificacion_Hahn\\NN(S)_Params_Exploración\\Predicciones\\"*filename, df)
end

# Función que lee los datos de PCA
function read_data_PCA(path_read::String, step_valid::Int64)
    df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)
    num_datos = size(df_datasignals)[1]
    k = 7 # Comienzo de los datos de validación
    datasignals_valid = Float32.(Matrix(df_datasignals[k^2:step_valid:num_datos,1:3])')
    datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),1:3])')

    σ_valid = df_datasignals[k^2:step_valid:num_datos,4]
    lcm_valid = df_datasignals[k^2:step_valid:num_datos,5]
    
    σ_col = df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),4]
    lcm_col = df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),5]
    
    dataparams = hcat(lcm_col, σ_col)'
    dataparams_valid = hcat(lcm_valid, σ_valid)'
    
    return datasignals, dataparams, datasignals_valid, dataparams_valid
end

##########################################################################################

# Función principal que realiza la exploración de los parámetros de la red neuronal

function main()
    # Arquitecturas que vamos a utilizar
    architectures = [
        [[3, 8, 2], relu], # Una capa oculta con pocas neuronas
        [[3, 16, 2], relu], # Una capa oculta con más neuronas
        [[3, 16, 8, 2], relu], # Dos capas ocultas
        [[3, 16, 16, 2], relu], # Dos capas ocultas con aún más neuronas
        [[3, 8, 16, 8, 2], relu], # Tres capas ocultas
        [[3, 16, 32, 16, 2], relu], # Tres capas ocultas con más neuronas
        [[3, 32, 64, 16, 2], relu], # Tres capas ocultas con aun más neuonras
        [[3, 16, 32, 16, 8, 2], relu], # Cuatro capas ocultas
        [[3, 32, 64, 8, 8, 2], relu], # Cuatro capas ocultas mas neuronas
        [[3, 32, 64, 32, 16, 2], relu], # Cuatro capas ocultas con aun mas neuronas
        [[3, 30, 25, 20, 15, 10, 2], relu], # Cinco capas ocultas, mayor complejidad
        [[3, 16, 8, 2], tanh], # Variando función de activación a tanh
        [[3, 16, 32, 16, 2], tanh], # Tres capas ocultas con más neuronas
        [[3, 32, 64, 16, 2], tanh], # Tres capas ocultas con aun más neuonras
        [[3, 32, 64, 8, 8, 2], tanh], # Cuatro capas ocultas mas neuronas
        [[3, 32, 64, 32, 16, 2], tanh], # Cuatro capas ocultas con aun mas neuronas
        [[3, 30, 25, 20, 15, 10, 2], σ], # Cinco capas ocultas σ
        [[3, 32, 64, 8, 8, 2], swish], # Cuatro capas ocultas mas neuronas
        [[3, 32, 64, 32, 16, 2], swish], # Cuatro capas ocultas con aun mas neuronas
        [[3, 30, 25, 20, 15, 10, 2], swish], # Cinco capas ocultas σ
        ]

    # Metodos de optimización que vamos a utilizar
    optimizers = [opt for opt in [ADAM, RMSProp]]

    # Lectura de los datos de PCA de las señales de Hahn
    path_read = "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\1-GeneracionDeDatos\\Datos_Final\\Datos2\\Datos_PCA2"
    
    # Fraccion de datos que se van a utilizar para validación
    percent_valid = 0.1
    step_valid = Int(1 / percent_valid)

    train_signals, train_params, valid_signals, valid_params = read_data_PCA(path_read, step_valid)

    # Cargamos los datos de entrenamiento y validación
    batchsize = 100
    shuffle = true
    data, data_valid = load_data(train_signals, train_params, valid_signals, valid_params, batchsize, shuffle)

########### Si se van a hacer mas exploraciones cambiar esto por el id de la ultima arquitectura usada.#################
    id = 43 # Ultima arquitectura usada
   
    id_column = []
    layers_column = []
    activation_column = []
    optimizer_column = []
    rmae_global_train_column = []
    rmae_global_valid_column = []

    for architecture in architectures
        for opt in optimizers
            id += 1
            string_id = string(id)
            layers = architecture[1]
            activation = architecture[2]
            
            if activation == σ
                activation_string = "σ"
            elseif activation == tanh
                activation_string = "tanh"
            elseif activation == relu
                activation_string = "relu"
            elseif activation == swish
                activation_string = "swish"
            end

            if opt == ADAM
                opt_string = "ADAM"
            elseif opt == Descent
                opt_string = "Descent"
            elseif opt == RMSProp
                opt_string = "RMSProp"
            end

            
            # Creamos el modelo
            model = create_model(layers, activation)

            # Definimos el learning rate inicial, ya que este va a ser variable cada 500 épocas

            learning_rate = 1e-4

            # Definimos el número de épocas

            epochs = 3000

            # Entrenamos el modelo

            rmae_global_train, rmae_global_valid = train_model(model, string_id, epochs, learning_rate, opt, data, data_valid)

            # Guardamos los datos de la arquitectura

            push!(id_column, id)
            push!(layers_column, layers)
            push!(activation_column, activation_string)
            push!(optimizer_column, opt_string)
            push!(rmae_global_train_column, rmae_global_train)
            push!(rmae_global_valid_column, rmae_global_valid)
        end
    end
    df = DataFrame(id = id_column, layers = layers_column, activation = activation_column, optimizer = optimizer_column, rmae_global_train = rmae_global_train_column, rmae_global_valid = rmae_global_valid_column)
    existing_csv_file = "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\2-RedNeruronal_Identificacion_Hahn\\NN(S)_Params_Exploración\\Arquitecturas\\Registro_arquitecturas2.csv"
    if isfile(existing_csv_file)
        df_old = CSV.read(existing_csv_file, DataFrame)
        df = vcat(df_old, df)
    end
    CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\2-RedNeruronal_Identificacion_Hahn\\NN(S)_Params_Exploración\\Arquitecturas\\Registro_arquitecturas2.csv", df)

end

# Llamamos a la función principal
main()

# Fin del programa