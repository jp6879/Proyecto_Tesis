# Programa que entrena la red neuronal con los datos reducidos de las señales y las distribuciones de probabilidad
# Tiene como salida lcm y σ con las que fueron generadas
# Se utiliza el método K-Folds para prevenir el overfitting de la red neuronal

# Importamos las librerias necesarias
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
using Measures

####################################################################################################################
# Traemos los mismos datos de los parametros utilizados para generar los datos, deberiamos hacer una función grande en la proxima función que genere los datos donde les pasamos
# Todos estos parámetros desde otro programa, como ahora generamos pocos datos me quedo con esto

###################################################ACORDARSE DE ESTO#################################################################
# Datos 1
# # Parámetros fijos

# # Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
# N = 2000
# time_sample_lenght = 100

# # Rango de tamaños de compartimientos en μm
# l0 = 0.01
# lf = 50

# lcs = Float32.(collect(range(l0, lf, length = N)))

# # Tiempo final de simulación en s
# tf = 1

# # Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
# lc = range(l0, lf, length = N)
# t = range(0, tf, length = time_sample_lenght)

# # Parametros que se varian

# # Rango de tamaños medios de correlación en μm
# lcms = 0.5:0.01:6
# σs = 0.01:0.01:1

####################################################################################################################
# Datos Final
# Parámetros fijos

# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 5000
time_sample_lenght = 1000

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 50

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lcs = range(l0, lf, length = N)
t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))

# Concatenate t_short and t_long

t = vcat(t_short, t_long)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.01:1
####################################################################################################################
# Distribucion de probabilidad log-normal que se puede agregar a la función costo de la red neuronal, lleva mucho tiempo de entrenamiento

function Pln(lcm, σ)
    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
end

####################################################################################################################

# Leemos los datos a los que les realizamos PCA de las señales, hay dos archivos porque son dos generaciones de datos distintas, con diferente resolución

# path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA2"
path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData_PCA_Final\\Datos_Final\\datos_PCA"

df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)

####################################################################################################################
# Funciones de pre procesamiento para escalar los datos y estandarizarlos

# Normalización Max-Min
function MaxMin(data)
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)
    scaled_data = (data .- min_vals) ./ (max_vals .- min_vals)
    return scaled_data

end

# Estandarización Z
function Standarize(data)
    mean_vals = mean(data, dims=1)
    std_devs = std(data, dims=1)
    standardized_data = (data .- mean_vals) ./ std_devs
    return standardized_data
end

####################################################################################################################
# Metricas de validacion de la red neuronal, solo utilice RMAE

# Root Mean Squared Error
function RMSE(predicted, real)
    return sqrt(sum((predicted .- real).^2) / length(predicted))
end

# Mean Absolute Error
function MAE(predicted, real)
    return sum(abs.(predicted .- real)) / length(predicted)
end

# R2 score
function R2_score(predicted, real)
    return 1 - sum((predicted .- real).^2) / sum((real .- mean(real)).^2)
end

# Realetive Root Mean Squared Error
function RRMSE(predicted, real)
    return sqrt(mean((predicted .- real).^2)) / mean(real)
end

# Relative Mean Absolute Error
function RMAE(predicted, real)
    return mean(abs.(predicted .- real)) / mean(real)
end

# Mean Absolute Percentaje Error
function MAPE(predicted, real)
    return mean(abs.((predicted .- real) ./ real))
end

####################################################################################################################

# Regularizaciones L1 y L2
pen_l2(x::AbstractArray) = Float32.(sum(abs2, x) / 2)
pen_l1(x::AbstractArray) = Float32.(sum(abs, x) / 2)

####################################################################################################################
# Utilizamos la técnica k-fold de validación cruzada para prevenir el overfitting de la red neuronal
# Definimos el número de folds
folds = 5
step_valid = 20
num_datos = Int(size(df_datasignals, 1))

# Guardamos los datos de validacion de cada NN en cada fold
out_of_sample_data = []
out_of_sample_pred = []

# Guardamos la metrica de validación de cada NN en cada fold
scores_MAE = []

# Primero sacamos los datos de testing de los datos de señales, estos seran un 3er conjunto de datos que no se usara para entrenar ni validar la red
df_datasignals_out = df_datasignals[6 + 1:step_valid:num_datos,:]
# Datos de Testing
df_datasignals_minus_out = df_datasignals[setdiff(1:num_datos, 6 + 1:step_valid:num_datos),:]
# Nuevo numero de datos que tenemos para entrenamiento + validacion
num_datos_new = Int(size(df_datasignals_minus_out, 1))

for k in 1:folds
    # Usamos 5 conjuntos disjuntos de datos de validación del 5% total de los datos para cada fold

    datasignals_valid = Float32.(Matrix(df_datasignals_minus_out[k^2 + 10:step_valid:num_datos_new,1:3])')
    datasignals = Float32.(Matrix(df_datasignals_minus_out[setdiff(1:num_datos_new, k^2 + 10:step_valid:num_datos_new),1:3])')

    σ_valid = df_datasignals_minus_out[k^2 + 10:step_valid:num_datos_new,4]
    lcm_valid = df_datasignals_minus_out[k^2 + 10:step_valid:num_datos_new,5]
    
    σ_col = df_datasignals_minus_out[setdiff(1:num_datos_new, k^2 + 10:step_valid:num_datos_new),4]
    lcm_col = df_datasignals_minus_out[setdiff(1:num_datos_new, k^2 + 10:step_valid:num_datos_new),5]
    
    dataparams = hcat(lcm_col, σ_col)'
    dataparams_valid = hcat(lcm_valid, σ_valid)'

    # Definimos la red neuronal
    model = Chain(
        Dense(3, 32, swish),
        Dense(32, 64, relu),
        Dense(64, 32, swish),
        Dense(32, 16, relu),
        Dense(16, 2, softplus),
    )

    # Función de loss
    function loss(x,y)
        return Flux.mse(model(x), y)
    end

    # Definimos el metodo de aprendizaje y la tasa de aprendizaje
    η = 1e-4
    opt = ADAM(η)

    # Definimos el número de épocas
    epochs = 3000

    # Definimos el tamaño del batch
    batch_size = 64

    # Usamos dataloader para cargar los datos
    data = Flux.DataLoader((datasignals, dataparams), batchsize = batch_size, shuffle = true)
    data_valid = Flux.DataLoader((datasignals_valid, dataparams_valid), batchsize = batch_size, shuffle = true)

    # Definimos una funcion de callback para ver el progreso del entrenamiento
    iter = 0
    cb = function()
        global iter += 1
        if iter % length(data) == 0
            epoch = iter ÷ length(data)
            if epoch % 500 == 0
                actual_loss = loss(data.data[1], data.data[2])
                actual_valid_loss = loss(data_valid.data[1], data_valid.data[2])
                println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
            end
            # push!(losses, actual_loss)
            # push!(losses_valid, actual_valid_loss)
        end
    end;

    # Entrenamos la red neuronal con el loss mse variando la tasa de aprendizaje cada 500 épocas
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model, opt), data, opt, cb=cb)
        if epoch % 500 == 0
            η = η * 0.2
            opt = ADAM(η)
        end
    end

    # Predicción de la red en la validacion
    predictions_valid = model(datasignals_valid)

    # Métricas de validación de la red
    RMAE_valid = RMAE(predictions_valid, dataparams_valid)

    push!(scores_MAE, RMAE_valid)

    # Guardamos los datos de validación y las predicciones de la red
    push!(out_of_sample_data, dataparams_valid)
    push!(out_of_sample_pred, predictions_valid)

    println("Fold $k terminado con score de validación RMAE = $RMAE_valid")

end

####################################################################################################################
#Ahora sacamos el promedio de la metrica de validacion
println("El promedio de la metrica de validación es $(mean(scores_MAE)) y el desvio estandar es $(std(scores_MAE))")

# Al haber testeado en todos estos datos y seguir obteniendo valores similares podemos asegurar que el modelo no está sobreajustando
# Ahora podemos re entrenar el modelo tranquilamente

num_datos = Int(size(df_datasignals_minus_out, 1))
step_valid = 20

k = 7

datasignals_test = Float32.(Matrix(df_datasignals_out[:,1:3])')
datasignals_valid = Float32.(Matrix(df_datasignals_minus_out[k^2:step_valid:num_datos_new,1:3])')
datasignals = Float32.(Matrix(df_datasignals_minus_out[setdiff(1:num_datos_new, k^2:step_valid:num_datos_new),1:3])')

σ_test = df_datasignals_out[:,4]
lcm_test = df_datasignals_out[:,5]

σ_valid = df_datasignals_minus_out[k^2:step_valid:num_datos_new,4]
lcm_valid = df_datasignals_minus_out[k^2:step_valid:num_datos_new,5]

σ_col = df_datasignals_minus_out[setdiff(1:num_datos_new, k^2:step_valid:num_datos_new),4]
lcm_col = df_datasignals_minus_out[setdiff(1:num_datos_new, k^2:step_valid:num_datos_new),5]


dataparams = hcat(lcm_col, σ_col)'
dataparams_valid = hcat(lcm_valid, σ_valid)'
dataparams_test = hcat(lcm_test, σ_test)'

model = Chain(
    Dense(3, 32, swish),
    Dense(32, 64, relu),
    Dense(64, 32, swish),
    Dense(32, 16, relu),
    Dense(16, 2, softplus),
)

# Función de loss
function loss(x,y)
    y_hat = model(x)
    return Flux.mse(y_hat, y)
end

# Loss compuesto
function composed_loss(x,y)
    y_hat = model(x)
    Pln_predicted = Pln.(y_hat[1,:], y_hat[2,:])
    Pln_real = Pln.(y[1,:], y[2,:])
    return mean(Flux.mse.(Pln_predicted,Pln_real)) + Flux.mse(y_hat, y)
end

# Definimos el batch size
batch_size = 64

# Usamos dataloader para cargar los datos
data = Flux.DataLoader((datasignals, dataparams), batchsize = batch_size, shuffle = true) 
data_valid = Flux.DataLoader((datasignals_valid, dataparams_valid), batchsize = batch_size, shuffle = true)

# Definimos el vector donde guardamos la pérdida
losses = []
losses_valid = []

# Parámetros de la red neuronal
params = Flux.params(model)

# Definimos una funcion de callback para ver el progreso del entrenamiento
iter = 0
cb = function()
    global iter += 1
    if iter % length(data) == 0
        epoch = iter ÷ length(data)
        actual_loss = loss(data.data[1], data.data[2])
        actual_valid_loss = loss(data_valid.data[1], data_valid.data[2])
        if epoch%100 == 0
            println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
        end
        push!(losses, actual_loss)
        push!(losses_valid, actual_valid_loss)
    end
end;

losses_composed = []
losses_composed_valid = []
cb2 = function()
    global iter += 1
    epoch = iter ÷ length(data)
    actual_loss = composed_loss(data.data[1], data.data[2])
    actual_valid_loss = composed_loss(data_valid.data[1], data_valid.data[2])
    println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
    push!(losses_composed, actual_loss)
    push!(losses_composed_valid, actual_valid_loss)
end;

# Definimos el modo de aprendizaje y la tasa de aprendizaje
η = 1e-5
opt = ADAM(η)

# Definimos el número de épocas
epochs = 5000

# Entrenamos la red neuronal con el loss mse
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), data, opt, cb=cb)
end

# for epoch in 1:1
#     Flux.train!(composed_loss, Flux.params(model, opt), data, opt, cb=cb2)
# end
####################################################################################################################

# Plots de la función de loss
pl_loss = Plots.plot(losses, xlabel = "Épocas", ylabel = "Loss", label = "Datos de entrenamiento", legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
Plots.plot!(losses_valid, xlabel = "Épocas", ylabel = "Loss", label = "Datos de validación", legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
#xlims!(100, length(losses))
ylims!(0, 1e-1)
# Plots.savefig(pl_loss,"C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Series\\Graficos\\LossS1.png")
# Plots.savefig(pl_loss,"C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Series\\Graficos\\LossS1.pdf")

####################################################################################################################
# Guardado del modelo
# using BSON: @save
# @save "S2.bson" model

# Load model
# using BSON: @load
# @load "S1.bson" model
####################################################################################################################

# Predicciones de la red
predictions = model(datasignals)
predictions_valid = model(datasignals_valid)
predictions_test = model(datasignals_test)

####################################################################################################################
# Medidas de error globales

# R2_train = R2_score(predictions, dataparams)
# RMSE_train = RMSE(predictions, dataparams)
MAE_train = MAE(predictions, dataparams)
# RRMSE_train = RRMSE(predictions, dataparams)
RMAE_train = RMAE(predictions, dataparams)

# R2_valid = R2_score(predictions_valid, dataparams_valid)
# RMSE_valid = RMSE(predictions_valid, dataparams_valid)
MAE_valid = MAE(predictions_valid, dataparams_valid)
# RRMSE_valid = RRMSE(predictions_valid, dataparams_valid)
RMAE_valid = RMAE(predictions_valid, dataparams_valid)

####################################################################################################################
# Medidas de error puntuales
N = length(predictions[1,:])
N_valid = length(predictions_valid[1,:])
N_test = length(predictions_test[1,:])

RMAE_scores = zeros(N)
RMAE_scores_valid = zeros(N_valid)
RMAE_scores_test = zeros(N_test)
MAE_scores = zeros(N)
MAE_scores_valid = zeros(N_valid)
MAE_scores_test = zeros(N_test)

for i in 1:N
    # RMSE_scores[i] = RMSE(predictions[:,i], dataparams[:,i])
    MAE_scores[i] = MAE(predictions[:,i], dataparams[:,i])
    # RRMSE_scores[i] = RRMSE(predictions[:,i], dataparams[:,i])
    RMAE_scores[i] = RMAE(predictions[:,i], dataparams[:,i])
end

for i in 1:N_valid
    # RMSE_scores_valid[i] = RMSE(predictions_valid[:,i], dataparams_valid[:,i])
    MAE_scores_valid[i] = MAE(predictions_valid[:,i], dataparams_valid[:,i])
    # RRMSE_scores_valid[i] = RRMSE(predictions_valid[:,i], dataparams_valid[:,i])
    RMAE_scores_valid[i] = RMAE(predictions_valid[:,i], dataparams_valid[:,i])
end

for i in 1:N_test
    MAE_scores_test[i] = MAE(predictions_test[:,i], dataparams_test[:,i])
    RMAE_scores_test[i] = RMAE(predictions_test[:,i], dataparams_test[:,i])
end

####################################################################################################################
# Evaluamos la red neuronal con los datos de testeo
best = 0
worst = 0
for i in 1:N_test
    if minimum(RMAE_scores_test) == RMAE_scores_test[i]
        println("El mínimo RMAE es $(RMAE_scores_test[i]) y se encuentra en la posición $i")
        best = i
    end
    if maximum(RMAE_scores_test) == RMAE_scores_test[i]
        println("El máximo RMAE es $(RMAE_scores_test[i]) y se encuentra en la posición $i")
        worst = i
    end
end

P_real_best = Pln.(dataparams_test[1,best], dataparams_test[2,best])
P_predict_best = Pln.(predictions_test[1,best], predictions_test[2,best])

P_real_worst = Pln.(dataparams_test[1,worst], dataparams_test[2,worst])
P_predict_worst = Pln.(predictions_test[1,worst], predictions_test[2,worst])

Plots.plot(lcs, P_real_best, label = L"Real $l_{cm} = $" * " $(dataparams_test[1,best]) " * L"$\mu m$" * L" $σ = $" * " $(dataparams_test[2,best])", xlabel = L"l_c", ylabel = L"P(l_c)", lw = 4, legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
Plots.plot!(lcs, P_predict_best, label = L"Predicción $l_{cm} = $" * " $(round(predictions_test[1,best],digits = 4)) " * L"$\mu m$" * L" $σ =$" * " $(round(predictions_test[2,best],digits = 4))",lw = 2, legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
# Plots.savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Series\\Graficos\\P_real_vs_P_predS2.png")
# Plots.savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Series\\Graficos\\P_real_vs_P_predS2.pdf")

Plots.plot(lcs, P_real_worst, label = L"Real $l_{cm} = $" * " $(dataparams_test[1,worst]) " * L"$\mu m$ " * L" $σ = $" * " $(dataparams_test[2,worst])", xlabel = L"l_c", ylabel = L"P(l_c)", lw = 4, legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
Plots.plot!(lcs, P_predict_worst, label = L"Predicción $l_{cm} = $" * " $(round(predictions_test[1,worst],digits = 4)) " * L"$\mu m$" * L" $σ = $" * " $(round(predictions_test[2,worst],digits = 4))", xlabel = L"l_c", ylabel = L"P(l_c)", lw = 2, legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
xlims!(0, 3)
# Plots.savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Series\\Graficos\\Worst_P_real_vs_P_predS2.pdf")
# Plots.savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Series\\Graficos\\Worst_P_real_vs_P_predS2.png")

####################################################################################################################
# Plots de los errores de la predicion de la red
params_error = PlotlyJS.scatter(
    x = datasignals_valid[1,1:end],
    y = datasignals_valid[2,1:end],
    mode = "markers",
    hoverinfo = "text",
    hovertext = RMAE_scores[1:end],
    marker = attr(
        color = RMAE_scores[1:end],  # Use the color_vector for color mapping
        colorscale = "Hot",  # Choose a predefined colormap (e.g., "Viridis")
        colorbar_title = "RMAE",
    ),
)

params_layout = Layout(
    title = "Datos de validación",
    xaxis = attr(title = "PC1"),
    yaxis = attr(title = "PC2"),
    font = attr(size = 15),
)

params_plot = PlotlyJS.plot([params_error], params_layout)

# Final del programa