### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 0b977bfa-d61a-4181-8fcd-e91ec519e376
md"Exploración de modelos para la predicción de parámetros a partir de las componentes principales de las señales"

# ╔═╡ 5949d9f3-0b40-44ca-9cf7-e9a2c9d8960a
begin
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
	
end

# ╔═╡ 614b4953-3017-4229-bf36-0ba4c1880bea
begin
	#------------------------------------------------------------------------------------------
	
	# Distribucion de probabilidad log-normal
	
	function Pln(lcm::Float32, σ::Float32)
	    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
	end
	
	#------------------------------------------------------------------------------------------
	# Metricas de validacion de la red neuronal
	
	# Relative Mean Absolute Error
	function RMAE(predicted, real)
	    return mean(abs.(predicted .- real)) / mean(real)
	end
	
	#------------------------------------------------------------------------------------------
	
	# Regularizaciones L1 y L2 para la red neuronal
	pen_l2(x::AbstractArray) = Float32.(sum(abs2, x) / 2)
	pen_l1(x::AbstractArray) = Float32.(sum(abs, x) / 2)
	
	#------------------------------------------------------------------------------------------
	
	# Función para crear los modelos de la red neuronal
	
	function create_model(layers::Vector{Int}, activation)
	    activations = [activation for i in 1:length(layers) - 2]
	    return Chain([Dense(layers[i], layers[i+1], activations[i]) for i in 1:length(layers) - 2]..., Dense(layers[end-1], layers[end], softplus))
	end
	
	#------------------------------------------------------------------------------------------
end

# ╔═╡ 72cd538b-4959-4585-9671-b39b7939bcfd
# Función para cargar los datos de entrenamiento, validacion

function load_data(x_train, y_train, x_valid, y_valid, batchsize::Int, shuffle::Bool)
    data = Flux.Data.DataLoader((x_train, y_train), batchsize = batchsize, shuffle = shuffle)
    data_valid = Flux.Data.DataLoader((x_valid, y_valid), batchsize = batchsize, shuffle = shuffle)
    return data, data_valid
end


# ╔═╡ 28baca08-f505-45d2-bd65-cd75d10dc647
# Función que evalua la red neuronal globalmente

function eval_model(model, x, y)
    y_pred = model(x)
    rmae = RMAE(y_pred, y)
    return rmae
end

# ╔═╡ 8d25b0f8-752e-46e0-ba05-307f48465af6
# Función que evalua la red neuronal punto a punto

function eval_model_point(model, x, y)
    y_pred = model(x)
    N = length(y_pred[1,:])

    rmae_scores = zeros(N)

    for i in 1:N
        rmae_scores[i] = RMAE(y_pred[:,i], y[:,i])
    end
    
    return rmae_scores
end

# ╔═╡ 992e3c61-f9f3-469b-aeee-e5fcf4c6d886
begin
	# Funciones de guardado de datos
	
	function save_loss(loss_vector, filename::String)
	    CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params_Exploración\\Funciones_loss\\"*filename, DataFrame(loss = loss_vector))
	end
	
	# Función para guardar las predicciones del modelo y el RMAE en cada punto
	
	function save_predictions(predictions, rmae_scores, filename::String)
	    df = DataFrame(x1 = predictions[1,:], x2 = predictions[2,:], rmae_scores = rmae_scores)
	    CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params_Exploración\\Predicciones\\"*filename, df)
	end
end

# ╔═╡ 1beead44-7e79-40b5-8d25-a05d5f55d8b5
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
    
    function loss_mse(x,y)
        y_hat = model(x)
        return Flux.mse(y_hat, y)
    end

    losses = []
    losses_valid = []
    
    # Parámetros de la red neuronal
    params = Flux.params(model)
    
    # Definimos una funcion de callback para ver el progreso del entrenamiento
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

# ╔═╡ 27184396-240c-475b-9d2c-6057e5af005c
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

# ╔═╡ 9b3ace8d-ecd7-4afb-9a2a-82997d567c1a
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
        [[3, 30, 25, 20, 15, 10, 2], σ], # Cinco capas ocultas σ
        ]
	
    optimizers = [opt for opt in [ADAM, RMSProp]]

    # Lectura de los datos
    path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA2"
    
    # Fraccion de datos que se van a utilizar para validación
    percent_valid = 0.1
    step_valid = Int(1 / percent_valid)

    train_signals, train_params, valid_signals, valid_params = read_data_PCA(path_read, step_valid)

    # Cargamos los datos de entrenamiento y validación
    batchsize = 64
    shuffle = true
    data, data_valid = load_data(train_signals, train_params, valid_signals, valid_params, batchsize, shuffle)

########### Si se van a hacer mas exploraciones cambiar esto por el id de la ultima arquitectura usada.#################
    id = 0

    # Función de costo que vamos a utilizar
    
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

            # Definimos el learning rate inicial

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
    existing_csv_file = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params_Exploración\\Arquitecturas\\Registro_arquitecturas.csv"
    if isfile(existing_csv_file)
        df_old = CSV.read(existing_csv_file, DataFrame)
        df = vcat(df_old, df)
    end
    CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params_Exploración\\Arquitecturas\\Registro_arquitecturas2.csv", df)

end

# ╔═╡ 7b5d50c2-8788-47f0-a031-1c0c22a22f4f
md"Se realizó una exploración de 15 arquitecturas distintas, en las que se realizó un aumento secuencial de complejidad variando el número de capas y número de neuronas. Adempás se exploraron algunas de estas redes cambiando la función de activación no lineal en las capas ocultas, dejando softmax en la ultima capa para obtener valores positivios.

Todas las arquitecturas fueron entrenadas durante 3000 épocas para mini-batches de 64 elementos, donde se utilizaron los optimizadores ADAM y RNSProp con una tasa de aprendizaje inicial de $1\times 10^{-4}$ disminuyendola un 80% cada 500 épocas.

Al finalizar cada entrenamiento las redes se evaluaron utilizando como métrica global el error relativo medio absoluto (RMAE) definido como

$\begin{equation}
\frac{\frac{1}{N} \sum_i |σ_{real,i} - σ_{pred,i}| + |{l_{cm}}_{real,i} - {l_{cm}}_{pred,i}|}{\frac{1}{N} \sum_i σ_{real,i} + {l_{cm}}_{real,i}},
\end{equation}$

donde $N$ es la cantidad de señales de entrenamiento o validación y los sub-indices $real$ y $pred$ inidican los valores rales y predichos por la red neuronal respectivamente.
" 

# ╔═╡ ad4c7345-d780-4905-9fac-8d8f079de673
md"Los resultados obtenidos se muestran en la siguiente tabla

$$\begin{aligned}
& \begin{array}{cccc}
\hline \hline
\text{layers} & \text{activation} & \text{optimizer} & \text{RMAE Train} & \text{RMAE Valid} \\
\hline
3, 8, 2 & relu & ADAM & 0.07033 & 0.06862 \\
3, 8, 2 & relu & RMSProp & 0.08425 & 0.08284 \\
3, 16, 2 & relu & ADAM & 0.04570 & 0.04320 \\
3, 16, 2 & relu & RMSProp & 0.03814 & 0.03624 \\
3, 16, 8, 2 & relu & ADAM & 0.02236 & 0.02089 \\
3, 16, 8, 2 & relu & RMSProp & 0.02724 & 0.02613 \\
3, 16, 16, 2 & relu & ADAM & 0.02243 & 0.02067 \\
3, 16, 16, 2 & relu & RMSProp & 0.02481 & 0.02295 \\
3, 8, 16, 8, 2 & relu & ADAM & 0.02432 & 0.02274 \\
3, 8, 16, 8, 2 & relu & RMSProp & 0.02947 & 0.02815 \\
3, 16, 32, 16, 2 & relu & ADAM & 0.01414 & 0.01263 \\
3, 16, 32, 16, 2 & relu & RMSProp & 0.01724 & 0.01582 \\
3, 32, 64, 16, 2 & relu & ADAM & 0.01365 & 0.01185 \\
3, 32, 64, 16, 2 & relu & RMSProp & 0.01585 & 0.01436 \\
3, 16, 32, 16, 8, 2 & relu & ADAM & 0.01355 & 0.01210 \\
3, 16, 32, 16, 8, 2 & relu & RMSProp & 0.01568 & 0.01368 \\
3, 32, 64, 8, 8, 2 & relu & ADAM & 0.01221 & 0.0100 \\
3, 32, 64, 8, 8, 2 & relu & RMSProp & 0.0180 & 0.01622 \\
3, 32, 64, 32, 16, 2 & relu & ADAM & 0.01326 & 0.01085 \\
3, 32, 64, 32, 16, 2 & relu & RMSProp & 0.02003 & 0.01761 \\
3, 30, 25, 20, 15, 10, 2 & relu & ADAM & 0.01316 & 0.01152 \\
3, 30, 25, 20, 15, 10, 2 & relu & RMSProp & 0.02073 & 0.01859 \\
3, 16, 8, 2 & tanh & ADAM & 0.02568 & 0.02328 \\
3, 16, 8, 2 & tanh & RMSProp & 0.02363 & 0.02106 \\
3, 16, 32, 16, 2 & tanh & ADAM & 0.01575 & 0.01317 \\
3, 16, 32, 16, 2 & tanh & RMSProp & 0.01707 & 0.01591 \\
3, 32, 64, 16, 2 & tanh & ADAM & 0.01385 & 0.01244 \\
3, 32, 64, 16, 2 & tanh & RMSProp & 0.01597 & 0.01286 \\
3, 30, 25, 20, 15, 10, 2 & \sigma & ADAM & 0.02373 & 0.02217 \\
3, 30, 25, 20, 15, 10, 2 & \sigma & RMSProp & 0.03188 & 0.02899 \\
3, 32, 64, 8, 8, 2 & swish & ADAM & 0.01682 & 0.01423 \\
3, 32, 64, 8, 8, 2 & swish & RMSProp & 0.01639 & 0.01359 \\
3, 32, 64, 32, 16, 2 & swish & ADAM & 0.01267 & 0.01056 \\
3, 32, 64, 32, 16, 2 & swish & RMSProp & 0.01571 & 0.01250 \\
3, 30, 25, 20, 15, 10, 2 & swish & ADAM & 0.01406 & 0.01250 \\
3, 30, 25, 20, 15, 10, 2 & swish & RMSProp & 0.01541 & 0.01273 \\
\hline
\end{array}
\end{aligned}$$

De este análisis se puede ver que la red que da un mayor rendimiento según la métrica de evaluación es aquella con 4 capas ocultas de 32, 64, 8 y 8 neuronas respectivamente, con la función de activación relu y entrenada con un optimizador ADAM. Otra de las redes con un rendimiento similar es aquella que tiene una mayor cantidad de neuronas en las 4 capas ocultas de 32, 64, 32 y 16 neuronas respectivamente. Estas redes son correspondientes con las filas



$$\begin{aligned}
& \begin{array}{cccc}
\hline \hline
\text{layers} & \text{activation} & \text{optimizer} & \text{RMAE Train} & \text{RMAE Valid} \\
\hline
3, 32, 64, 8, 8, 2 & \text{relu} & \text{ADAM} & 0.01221 & 0.0100 \\
3, 32, 64, 32, 16, 2 & relu & ADAM & 0.01326 & 0.01085 \\
\hline
\end{array}
\end{aligned}$$

Estas puden ser potenciales arquitecuras ópimas para la tarea que buscamos realizar donde se puede hacer una exploración para encontrar los hiperparámetros ópitmos.

Para continuar con esta exploración se realizó además una exploración de estas mismas redes usando esta vez los datos agrupados en mini-batchs de 100 elementos obteniendo

$$\begin{aligned}
& \begin{array}{ccccccc}
\hline \hline
& \text{Layers} & \text{Activation} & \text{Optimizer} & \text{RMAE Train} & \text{RMAE Valid} \\
\hline
& 3, 16, 32, 16, 8, 2 & relu & ADAM & 0.01757 & 0.01521 \\
& 3, 32, 64, 8, 8, 2 & relu & ADAM & 0.01704 & 0.01593 \\
& 3, 32, 64, 32, 16, 2 & relu & ADAM & 0.01837 & 0.01591 \\
& 3, 30, 25, 20, 15, 10, 2 & relu & ADAM & 0.01377 & 0.01222 \\
& 3, 32, 64, 8, 8, 2 & tanh & ADAM & 0.01494 & 0.01340 \\
& 3, 32, 64, 32, 16, 2 & tanh & ADAM & 0.01289 & 0.01096 \\
\hline
\end{array}
\end{aligned}$$

En este caso se obtuvo un rendimiento peor para las redes con las funciones de activación relu pero un rendimiento similar para la última red con función de activación tanh.

Es necesario realizar una exploración para los hiperparámetros restantes pero por lo observado aumentar el tamaño de los mini-batches no mejora el funcionamiento de las redes.

" 

# ╔═╡ Cell order:
# ╟─0b977bfa-d61a-4181-8fcd-e91ec519e376
# ╠═5949d9f3-0b40-44ca-9cf7-e9a2c9d8960a
# ╠═614b4953-3017-4229-bf36-0ba4c1880bea
# ╠═72cd538b-4959-4585-9671-b39b7939bcfd
# ╠═1beead44-7e79-40b5-8d25-a05d5f55d8b5
# ╠═28baca08-f505-45d2-bd65-cd75d10dc647
# ╠═8d25b0f8-752e-46e0-ba05-307f48465af6
# ╠═992e3c61-f9f3-469b-aeee-e5fcf4c6d886
# ╠═27184396-240c-475b-9d2c-6057e5af005c
# ╠═9b3ace8d-ecd7-4afb-9a2a-82997d567c1a
# ╠═7b5d50c2-8788-47f0-a031-1c0c22a22f4f
# ╠═ad4c7345-d780-4905-9fac-8d8f079de673
