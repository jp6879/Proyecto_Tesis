# Programa para visualizar los resultados de cierta arquitectura ya entrenada
# Importamos los paquetes necesarios
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
using CubicSplines

using Plots
# include("/home/juan.morales/ExploracionCompleta/UtilsRepresentative.jl")
include("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionCompleta/UtilsRepresentative.jl")

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
toriginal = vcat(t_short, t_long)
# path_read = "/home/juan.morales/datos_PCA"
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


function calucla_recta(x, x1, y1, x0 = 0, y0 = 1)
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return Float32.(m .* x' .+ b)
end


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


# Función que devuelve un interpolador de la derivada calculada utilizando los puntos
# de la señal que se tienen
function get_interpolated_deriv(tvalid, Signals_derivadas_valid)
    itp_derivadas = []
    # Interpolamos las derivadas
    for i in 1:size(Signals_derivadas_valid)[2]
        # push!(itp_derivadas, CubicSpline(tvalid, Signals_derivadas_valid[:,i], extrapl = [2,], extrapr=[2,]))
        push!(itp_derivadas, Interpolations.linear_interpolation(tvalid, Signals_derivadas_valid[:,i], extrapolation_bc=Line()))
    end
    return itp_derivadas
end

##############################################################################################

t_short = Float32.(collect(range(0, 0.1, length = 1000)))
t_long = Float32.(collect(range(0.1, 1, length = 100)))
    
# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto =  25 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 5

# Esto da 100 tiempos 50 puntos desde 0 a 0.1 y 25 puntos desde 0.1 a 1
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]

t = vcat(t_short, t_long)

# Tomamos 1 sigmas y 5 tamaños de compartimientos para cada sigma o sea 60 señales
sampled_sigmas =  [1.0]
lcm_range = 1:600:600
lcm_test = 1:250:600

println("Sigmas: ", sampled_sigmas)
println("Lcms: ", collect(lcms)[lcm_range])
lcm_test = [l for l in collect(lcm_test) if l ∉ collect(lcm_range)]
println("Lcms test: ", collect(lcms)[lcm_test])

# Obtenemos las señales representativas para un conjunto de sigmas y lcms
Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)
Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmas_test = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_test, muestreo_corto, muestreo_largo, t)

#################################################################################################################################################
# Grafiquemos las señales
plot_signals = scatter(t, Signals_rep[1,:], label = "Señales", xlabel = "t", ylabel = "S(t)", title = "Varias Señales", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
for i in 2:size(Signals_rep)[1]
    scatter!(plot_signals, t, Signals_rep[i,:], label = false, lw = 2)
end

scatter!(t, Signals_test[1,:], label = "Señales Test", lw = 2, color = :orange)
for i in 2:size(Signals_test)[1]
    scatter!(plot_signals, t, Signals_test[i,:], label = false, lw = 2, color = :orange)
end

plot_signals
#################################################################################################################################################

# Numero de puntos para la predicción
n_valid = 3

# Paso para tomar los tiempos de entrenamiento y validación
step = floor(Int, length(t) / n_valid) + 1

tvalid = t[1:step:end]

# En la validación y en el train tenemos que tener el primer y último tiempo
tvalid = vcat(tvalid, t[end])

indexes_valid = [i for i in 1:length(t) if t[i] in tvalid]

Signals_valid = Signals_rep[:,indexes_valid]
Signals_test_puntos = Signals_test[:,indexes_valid]

# Obtenemos las derivadas de las señales de validación
Signals_derivadas_valid = get_signals_deriv_valid(tvalid, Signals_valid)
Signals_derivadas_test = get_signals_deriv_valid(tvalid, Signals_test_puntos)

#################################################################################################################################################
plot_puntos_usados = scatter(tvalid, Signals_valid[1,:], label = "Puntos usados", xlabel = "t", ylabel = "S(t)", title = "Varias Señales", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
for i in 2:size(Signals_valid)[1]
    scatter!(plot_puntos_usados, tvalid, Signals_valid[i,:], label = false, lw = 2)
end

scatter!(tvalid, Signals_test_puntos[1,:], label = "Puntos usados Test", lw = 2, color = :orange)

for i in 2:size(Signals_test_puntos)[1]
    scatter!(plot_puntos_usados, tvalid, Signals_test_puntos[i,:], label = false, lw = 2, color = :orange)
end

display(plot_puntos_usados)

plot_derivadas_train = scatter(tvalid, Signals_derivadas_valid[:,1], label = "Derivadas usadas", xlabel = "t", ylabel = "S'(t)", title = "Derivadas de las señales de entrenamiento", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
for i in 2:size(Signals_valid)[1]
    scatter!(plot_derivadas_train, tvalid, Signals_derivadas_valid[:,i], label = false, lw = 2)
end

scatter!(tvalid, Signals_derivadas_test[:,1], label = "Derivadas usadas Test", lw = 2, color = :orange)
for i in 2:size(Signals_test_puntos)[1]
    scatter!(plot_derivadas_train, tvalid, Signals_derivadas_test[:,i], label = false, lw = 2, color = :orange)
end

display(plot_derivadas_train)

#################################################################################################################################################

# Obtenemos las interpolaciones de las señales y de las derivadas
itp_signals = get_interpolated_deriv(tvalid, Signals_valid')
itp_derivadas = get_interpolated_deriv(tvalid, Signals_derivadas_valid)

itp_signals_test = get_interpolated_deriv(tvalid, Signals_test_puntos')
itp_derivadas_test = get_interpolated_deriv(tvalid, Signals_derivadas_test)

extra_parameters = itp_signals
extra_parameters2 = itp_derivadas

extra_parameters_test = itp_signals_test
extra_parameters2_test = itp_derivadas_test

#################################################################################################################################################
# Veamos las interpolaciones que acabamos de hacer

plot_interpolaciones = scatter(toriginal, itp_signals[1][toriginal], label = "Interpolación Señales", xlabel = "t", ylabel = "S(t)", title = "Interpolaciones de las señales", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
for i in 2:size(Signals_valid)[1]
    scatter!(plot_interpolaciones, toriginal, itp_signals[i][toriginal], label = false, lw = 2)
end

display(plot_interpolaciones)

scatter!(t, itp_signals_test[1][t], label = "Interpolación Señales Test", lw = 2, color = :orange)
for i in 2:size(Signals_test_puntos)[1]
    scatter!(plot_interpolaciones, t, itp_signals_test[i][t], label = false, lw = 2, color = :orange)
end

display(plot_interpolaciones)
#################################################################################################################################################

# Todas las señales tienen la misma condición inicial U0 = 1
U0 = ones32(size(Signals_rep)[1])
U0_test = ones32(size(Signals_test)[1])

# id actual de la red
actual_id = 17

#Definimos el batch size
batch_size = 10

# Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
train_loader = Flux.Data.DataLoader((Signals_rep, t), batchsize = batch_size)

# Función de activación
activation = relu

# 7,"[3, 32, 64, 16, 1]",relu,AdamW,15,0.015440804844603407,0.37564877375723504

nn = Chain(Dense(2, 32, activation),
            Dense(32, 64, activation),
            Dense(64, 16, activation),
            Dense(16, 1)
            )

# Tomamos un learning rate de 0.001
η = 5e-3

# Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
epochs = 500

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial
p
# Leemos los parámetros de la red ya entrenada si es que existen
if isfile("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/ArquitecturaEspecifica/Parameters/$(actual_id)_ParametersV2.csv")
    theta = CSV.read("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/ArquitecturaEspecifica/Parameters/$(actual_id)_ParametersV2.csv", DataFrame)
    p = Float32.(theta[:,1])
else
    println("No se encontraron los parámetros de la red neuronal")
end

p

##############################################################################################

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

# Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
function predict_NeuralODE(u0, parametros, parametros2, time_batch)
    # dSdt = NN(S, parametros_extra) 
    function dSdt(u, p, t; parametros_extra = parametros, parametros_extra2 = parametros2)
        # parametros_actuales = parametros(t) # Selecciona los parametros extra en el tiempo t
        # parametros_actuales_2 = parametros2(t)
        # entrada_red = vcat(u, parametros_actuales, parametros_actuales_2) # Concatena los el valor de S(t) con los parametros extra en el tiempo t
        # entrada_red = u
        entrada_red = vcat(u, parametros_extra(t))
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
        predicted_signal = predict_NeuralODE(u0, parametros_extra[i], parametros_extra2[i], time_batch)[1, :]
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
    pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
end

function L2pen(y)
    return sum(y .^ 2)
end

function loss_node(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0, extra_parameters, extra_parameters2, time_batch)
    return Flux.mse(y, batch') #+ lamb * (penalization_term(time_batch, y))
end

function loss_valid(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0_test, extra_parameters_test, extra_parameters2_test, time_batch)
    return Flux.mse(y, batch') #+ lamb * (penalization_term(time_batch, y))
end


# Función de callback para guardar el loss en cada época
global iter = 0
loss = []
loss_valid_array = []
callback = function ()
    global iter += 1
    if iter % (length(train_loader)) == 0
        epoch = Int(iter / length(train_loader))
        actual_loss = loss_node(Signals_rep, t)
        forecast_loss = loss_valid(Signals_test, t)
        println("Epoch = $epoch || Loss: $actual_loss || Loss Forecast: $forecast_loss")
        push!(loss, actual_loss)
        push!(loss_valid_array, forecast_loss)
        if epoch % 20 == 0
            plot_predictions = scatter(t, Signals_rep', label = false, xlabel = "t", ylabel = "S(t)", title = "Predicción de señales", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
            plot!(t, Predict_Singals(U0, extra_parameters, extra_parameters2, t), label = false , xlabel = "t", ylabel = "S(t)", title = "Predicción de señales", lw = 2, color = :red, markershape = :circle)
            scatter!(t, Signals_test', label = false, lw = 2, color = :orange, markershape = :star6)
            scatter!(t, Predict_Singals(U0_test, extra_parameters_test, extra_parameters2_test, t), label = false, lw = 2, color = :blue, markershape = :utriangle)
            display(plot_predictions)

        end
    end
    return false
end

###############################################################################################################

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, epochs), opt, cb = callback)
# Guardamos los parámetros
df_parameters = DataFrame(reshape(p, length(p), 1), :auto)

CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/ArquitecturaEspecifica/Parameters/$(actual_id)_ParametersV2.csv", df_parameters)

# Guardamos las funciónes de loss
Loss_Matrix = zeros((length(loss), 2))

for i in 1:length(loss)
    Loss_Matrix[i,1] = loss[i]
    Loss_Matrix[i,2] = loss_valid_array[i]
end

df_losses = DataFrame(Loss_Matrix, :auto)
rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

if isfile("../3-GeneracionDeSeñales/ExploracionV2/ArquitecturaEspecifica/Losses/$(actual_id)_lossesV2.csv")
   df_losses = CSV.read("../3-GeneracionDeSeñales/ExploracionV2/ArquitecturaEspecifica/Losses/$(actual_id)_lossesV2.csv", DataFrame)
   loss = df_losses[:,1]
   loss_valid_array = df_losses[:,2]
else
    println("No se encontraron los loss de la red neuronal")
end

CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/ArquitecturaEspecifica/Losses/$(actual_id)_lossesV2.csv", df_losses)

start_index = 1

plots_loss = plot(loss[start_index:end], label = "Loss de entrenamiento", xlabel = "Época", ylabel = "Loss", title = "Loss de entrenamiento", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
plot!(loss_valid_array[start_index:end], label = "Loss de predicción", xlabel = "Época", ylabel = "Loss", title = "Loss de entrenamiento", lw = 2)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/ArquitecturaEspecifica/Imagenes/3LSignals_LossV2.png")

# # ##############################################################################################

# # # Vamos a hacer un plot de las señales de entrenamiento y sus predicciones
plot_predictions = plot(t, Signals_rep', label = false)
plot(toriginal, Predict_Singals(U0, extra_parameters, extra_parameters2, toriginal), label = false, lw = 2, color = :red)
scatter!(tvalid, Signals_valid', label = false, lw = 2, color = :blue, markershape = :star6, markersize = 8)
xlims!(0, 0.1)

plot_test = scatter(t, Signals_test', label = false)
scatter!(t, Predict_Singals(U0_test, extra_parameters_test, extra_parameters2_test, t), label = false, lw = 2, color = :red, markershape = :circle)
xlims!(0, 0.1) 