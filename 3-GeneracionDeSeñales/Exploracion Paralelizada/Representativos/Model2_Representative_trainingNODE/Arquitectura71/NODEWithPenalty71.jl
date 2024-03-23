# Programa para visualizar los resultados de cierta arquitectura ya entrenada
# Importamos los paquetes necesarios
using Plots
using Flux
using Flux: train!
using DataFrames
using CSV
using LaTeXStrings
using DifferentialEquations
using DiffEqFlux
using Statistics
using Random
using LinearAlgebra
using SciMLSensitivity
using ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux
using Interpolations
using OrdinaryDiffEq
using IterTools: ncycle
using Measures
include("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Representativos\\Model2_Representative_trainingNODE\\Arquitectura71\\UtilsRepresentative.jl")

##############################################################################################
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

path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

# Parametros que se varian
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
sigmas = 0.01:0.01:1

##############################################################################################
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
##############################################################################################

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
lcm_range = 126:25:250

# Obtenemos las señales representativas para un conjunto de sigmas y lcms
Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

# A partir de que indice de tiempo vamos a tomar predicciones
# Los anteriroes tiempos se usan para entrenar la NODE
idx_forecast = 61

# Definimos el tamaño del batch
batch_size = 30

# Tiempos de entrenamiento y de predicción
tforecast = t[idx_forecast:end]
ttrain = t[1:idx_forecast-1]

# Señaes de entrenamiento y de predicción
Signals_train = Signals_rep[:,1:idx_forecast-1]
Signals_valid = Signals_rep[:,idx_forecast:end]

# Derivadas de las señales de entrenamiento y de predicción
Signals_derivadas_train = Signals_rep_derivadas[1:idx_forecast-1,:]
Signals_derivadas_valid = Signals_rep_derivadas[idx_forecast:end,:]
extra_parameters = Signals_derivadas_train

# Todas las señales tienen la misma condición inicial U0 = 1
U0 = ones32(size(Signals_rep)[1])

# Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
train_loader = Flux.Data.DataLoader((Signals_train, ttrain), batchsize = batch_size)

# ID,   Arquitectura,                           Activación,     Optimizador,        Batch_Size,     Loss_Final_Entrenamiento,       Loss_Final_Predicción
# 71,   "[2, 128, 256, 32, 64, 32, 16, 8, 1]",    swish,        RMSProp,            30,             0.0074902220891056035,          0.001033835837930216

nn = Chain(Dense(2, 128, swish),
            Dense(128, 256, swish),
            Dense(256, 32, swish),
            Dense(32, 64, swish),
            Dense(64, 32, swish),
            Dense(32, 16, swish),
            Dense(16, 8, swish),
            Dense(8, 1)
            )

# Tomamos un learning rate de 0.001
η = 1e-3

# Vamos a tomar 30 épocas para entrenar todas las arquitecturas
epochs = 5

# Parametrizar indices para poder acceder a los parametros extras de la red neuronal
f(x,p) = round(Int, x * (length(p) - 1)) + 1

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

# Leemos parámetros de la arquitectura 17
θ = CSV.read("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Representativos\\Model2_Representative_trainingNODE\\Parameters\\71_Parameters.csv", DataFrame)
p = Float32.(θ[:,1])
actual_loss = loss_node(Signals_train, ttrain)
forecast_loss = loss_node(Signals_valid, tforecast)
##############################################################################################

# Optimizardor
opt = RMSProp(η)

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

    return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-7, abstol = 1e-7)) # Regresa la solución de la ODE
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

# Penalization term

function penalization_term(time_batch,y)
    """
    Función de penalización para tratar de mantener la señal monotonamente decrecente
    En caso que la señal no sea monotonamente decreciente, la penalización es la suma de las diferencias positivas entre los tiempos
    """
    # pen = 0
    # for i in 1:size(y)[1]
    #     pen += sum(max.(y[i,2:end] .- y[i,1:end-1], 0))
    # end
    # return pen
    pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
end

# pen1 = penalization_term(ttrain, y)
# plot(ttrain, y[:,1], label = "Predicción", lw = 2, color = :red)
# plot!(ttrain, Signals_test_train[1,:], label = "Real", lw = 2, color = :blue)

# y = Predict_Singals(U0, extra_parameters, ttrain)
# y_forecasted = Predict_Singals(U0, extra_parameters, tforecast)
# t_total = vcat(ttrain, tforecast)
# y_total = vcat(y, y_forecasted)

function loss_node(batch, time_batch, tforecast = tforecast , lamb = 1.0)
    y = Predict_Singals(U0, extra_parameters, time_batch)
    # y_forecasted = Predict_Singals(U0, extra_parameters, tforecast)
    return Flux.mse(y, batch') #+ lamb * (penalization_term(time_batch, y) + penalization_term(tforecast, y_forecasted))
end

# Función de callback para guardar el loss en cada época
global iter = 0
loss = []
loss_forecast = []
callback = function ()
    global iter += 1
    if iter % (length(train_loader)) == 0
        epoch = Int(iter / length(train_loader))
        actual_loss = loss_node(Signals_train, ttrain)
        forecast_loss = loss_node(Signals_valid, tforecast)
        println("Epoch = $epoch || Loss: $actual_loss || Loss Forecast: $forecast_loss")
        push!(loss, actual_loss)
        push!(loss_forecast, forecast_loss)
    end
    return false
end

# println("Loss inicial: ", loss_node(Signals_test_train, ttrain))

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p,U0), ncycle(train_loader, epochs), opt, cb = callback)

##############################################################################################
# Vamos a leer los loss del entrenamiento que ya teníamos
# 
read_loss_path = "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Representativos\\Model2_Representative_trainingNODE\\Losses\\71_losses.csv"
df_loss = CSV.read(read_loss_path, DataFrame)

Loss_Entrenamiento = vcat(df_loss[!,"Loss_Entrenamiento"], loss)
Loss_Prediccion = vcat(df_loss[!,"Loss_Predicción"], loss_forecast)

println("Loss Entrenamiento: ", Loss_Entrenamiento[end])
println("Loss Predicción: ", Loss_Prediccion[end])

plot(Loss_Entrenamiento, label = "Loss entrenamiento", xlabel = "Épocas", ylabel = "Loss", lw = 2,tickfontsize=12, labelfontsize=15, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, rightmargin = 5mm)
plot!(Loss_Prediccion, label = "Loss predicción", lw = 2, color = :red)

# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\13_losses.pdf")

# ##############################################################################################
# # Vamos a hacer un plot de las señales de entrenamiento y sus predicciones

linspace = range(-0.05, 1.1, length = 1000)
constant = ones(length(linspace))*t[idx_forecast]


idx_mitad = Int(size(Signals_train)[1]//2)

plot_signals = plot(vcat(t_short,t_long), Signals_rep[1,:], label = "Señal σ = $(column_sigmas_rep[1]) lcm = $(column_lcm_rep[1])", xlabel = L"t", ylabel = L"S(t)", title = "Predicción de señales", lw = 2, color = :blue, markershape = :circle, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
plot!(ttrain, Predict_Singals(U0[1], Signals_derivadas_train[:,1], ttrain), color = :red, label = "Entrenamiento", markershape = :star6, ls = :dash, lw = 2)
plot!(tforecast, Predict_Singals(U0[1], Signals_derivadas_train[:,1], tforecast), color = :orange, label = "Predicción", markershape = :utriangle, ls = :dash, lw = 2)
plot!(constant, linspace, label = false, lw = 2, ls = :dash, color = :gray)

# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\Arquitectura_13_WP_Signal_1.pdf")
# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\Arquitectura_13_WP_Signal_1.png")

plot(vcat(t_short,t_long), Signals_rep[idx_mitad,:], label = "Señal σ = $(column_sigmas_rep[idx_mitad]) lcm = $(column_lcm_rep[idx_mitad])", xlabel = L"t", ylabel = L"S(t)", title = "Predicción de señales", lw = 2, color = :blue, markershape = :circle, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
plot!(ttrain, Predict_Singals(U0[idx_mitad], Signals_derivadas_train[:,idx_mitad], ttrain), color = :red, label = "Entrenamiento", markershape = :star6, ls = :dash, lw = 2)
plot!(tforecast, Predict_Singals(U0[idx_mitad], Signals_derivadas_train[:,idx_mitad], tforecast), color = :orange, label = "Predicción", markershape = :utriangle, ls = :dash, lw = 2)

# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\Arquitectura_13_WP_Signal_$(idx_mitad).pdf")
# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\Arquitectura_13_WP_Signal_$(idx_mitad).png")

plot(vcat(t_short,t_long), Signals_rep[end,:], label = "Señal σ = $(column_sigmas_rep[end]) lcm = $(column_lcm_rep[end])", xlabel = L"t", ylabel = L"S(t)", title = "Predicción de señales", lw = 2, color = :blue, markershape = :circle, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
plot!(ttrain, Predict_Singals(U0[end], Signals_derivadas_train[:,end], ttrain), color = :red, label = "Entrenamiento", markershape = :star6, ls = :dash, lw = 2)
plot!(tforecast, Predict_Singals(U0[end], Signals_derivadas_train[:,end], tforecast), color = :orange, label = "Predicción", markershape = :utriangle, ls = :dash, lw = 2)

# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\Arquitectura_13_WP_Signal_$(size(Signals_test)[1]).pdf")
# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\Arquitectura_13_WP_Signal_$(size(Signals_test)[1]).png")

plot_predictions = plot(ttrain, Predict_Singals(U0[1], Signals_derivadas_train[:,1], ttrain), label = "Entrenamiento", xlabel = L"t", ylabel = L"S(t)", title = "Predicción de señales", lw = 2, color = :red, markershape = :circle, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
plot!(tforecast, Predict_Singals(U0[1], Signals_derivadas_train[:,1], tforecast), label = "Predicción", lw = 2, color = :orange, markershape = :utriangle, ls = :dash)

for i in 2:2:size(Signals_rep)[1]
    # plot!(plot_signals,vcat(t_short,t_long), Signals_test[i,:], label = false, xlabel = L"t", ylabel = L"S(t)", title = "Predicción de señales", lw = 2, color = :blue, markershape = :circle, tickfontsize=12, labelfontsize=15, legendfontsize=5, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
    plot!(ttrain, Predict_Singals(U0[i], Signals_derivadas_train[:,i], ttrain), color = :red, label = false, markershape = :star6, ls = :dash, lw = 2)
    plot!(tforecast, Predict_Singals(U0[i], Signals_derivadas_train[:,i], tforecast), color = :orange, label = false, markershape = :utriangle, ls = :dash, lw = 2)
end

plot_predictions
# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\Arquitectura_13_WP_Predictions.pdf")
# savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Arquitectura-13\\Arquitectura_13_WP_Predictions.png")