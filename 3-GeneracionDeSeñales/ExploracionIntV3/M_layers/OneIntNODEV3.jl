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
using BSplineKit
using Plots
# include("/home/juan.morales/ExploracionCompleta/UtilsRepresentative.jl")
include("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionCompleta/UtilsRepresentative.jl")

#################################################################################################################################################

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

# Tenemos dos tipos de sampleos uno desde 0 a 0.1 que tiene 0.0001 de paso y otro de 0.1 a 1 que tiene 0.001 de paso
t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))
    
# El tiempo total es concatenar estos dos arreglos
t = vcat(t_short, t_long)

# Guardamos este tiempo total para poder hacer las predicciones
toriginal = vcat(t_short, t_long)

# Path desde donde se leen los datos
# path_read = "/home/juan.morales/datos_PCA"
path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

# Parametros que varían para cada señal
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
# Desviaciones estándar
sigmas = 0.01:0.01:1

#################################################################################################################################################
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

#################################################################################################################################################

t_short = Float32.(collect(range(0, 0.1, length = 1000)))
t_long = Float32.(collect(range(0.1, 1, length = 100)))

# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto = 25 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 5


# Esto da 60 tiempos 40 puntos desde 0 a 0.1 y 20 puntos desde 0.1 a 1
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]


t = vcat(t_short, t_long)

# Tomamos 1 sigmas y 12 tamaños de compartimientos para cada sigma como entrenamiento
# Probamos en un conjunto pequeño de test para los mismos sigmas
# Incluyo además otro conjunto de validación para los mismos sigmas
sampled_sigmas =  [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
lcm_range = 1:20:551
lcm_valid = 5:75:551
lcm_test = 13:75:551

lcm_valid = [l for l in collect(lcm_valid) if l ∉ collect(lcm_range)]
lcm_test = [l for l in collect(lcm_test) if l ∉ collect(lcm_range)]


println("Sigmas: ", sampled_sigmas)
println("Lcms: ", collect(lcms)[lcm_range])
println("Lcms validacion: ", collect(lcms)[lcm_valid])
println("Lcms test: ", collect(lcms)[lcm_test])

# Sigmas: [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
# Lcms: [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9]
# Lcms validacion: [0.54, 1.29, 2.04, 2.79, 3.54, 4.29, 5.04, 5.79]
# Lcms test: [0.62, 1.37, 2.12, 2.87, 3.62, 4.37, 5.12, 5.87]

# Obtenemos las señales representativas para un conjunto de sigmas y lcms
Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)
Signals_valid, Signals_valid_derivadas, column_lcm_valid, column_sigmas_valid = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_valid, muestreo_corto, muestreo_largo, t)
Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmas_test = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_test, muestreo_corto, muestreo_largo, t)

#################################################################################################################################################
# Grafiquemos las señales

plot_signals = plot(t, Signals_rep[1,:], label = "Señales Entrenamiento", xlabel = "t", ylabel = "S(t)", title = "Varias Señales", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, color =:red)
for i in 2:size(Signals_rep)[1]
    plot!(plot_signals, t, Signals_rep[i,:], label = false, lw = 2, color = :red)
end

plot!(t, Signals_valid[1,:], label = "Señales Validación", lw = 2, color = :blue)
for i in 2:size(Signals_valid)[1]
    plot!(plot_signals, t, Signals_valid[i,:], label = false, lw = 2, color = :blue)
end

plot!(t, Signals_test[1,:], label = "Señales Test", lw = 2, color = :black)
for i in 2:size(Signals_test)[1]
    plot!(plot_signals, t, Signals_test[i,:], label = false, lw = 2, color = :black)
end

display(plot_signals)
xlims!(0, 0.1)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SignalsNEWZoom.png")

#################################################################################################################################################

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
Signals_test_puntos = Signals_test[:,indexes_usados]

# Obtenemos las derivadas de las señales de validación
Signals_rep_derivadas_puntos = get_signals_deriv(t_usados, Signals_rep_puntos)
Signals_valid_derivadas_puntos = get_signals_deriv(t_usados, Signals_valid_puntos)
Signals_test_derivadas_puntos = get_signals_deriv(t_usados, Signals_test_puntos)

#################################################################################################################################################
# Graficamos los puntos usados de todas las señales
plot_puntos_usados = scatter!(t_usados, Signals_rep_puntos[1,:], label = "Puntos usados", xlabel = "t", ylabel = "S(t)", title = "Varias Señales", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, color = :red)
for i in 2:size(Signals_rep_puntos)[1]
    scatter!(plot_puntos_usados, t_usados, Signals_rep_puntos[i,:], label = false, lw = 2, color = :red)
end

scatter!(t_usados, Signals_valid_puntos[1,:], label = "Puntos usados Validación", lw = 2, color = :blue)

for i in 2:size(Signals_valid_puntos)[1]
    scatter!(plot_puntos_usados, t_usados, Signals_valid_puntos[i,:], label = false, lw = 2, color = :blue)
end

scatter!(t_usados, Signals_test_puntos[1,:], label = "Puntos usados Test", lw = 2, color = :black)

for i in 2:size(Signals_test_puntos)[1]
    scatter!(plot_puntos_usados, t_usados, Signals_test_puntos[i,:], label = false, lw = 2, color = :black)
end

display(plot_puntos_usados)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SeñalesEntrenamiento.png")
xlims!(0, 0.1)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SeñalesEntrenamientoZoom.png")

# Graficamos las derivadas calculadas con los puntos usados

plot_derivadas_puntos = scatter(t_usados, Signals_rep_derivadas_puntos[:,1], label = "Derivadas usadas", xlabel = "t", ylabel = "S'(t)", title = "Derivadas de las señales de entrenamiento", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, color = :red)

for i in 2:size(Signals_rep_puntos)[1]
    scatter!(plot_derivadas_puntos, t_usados, Signals_rep_derivadas_puntos[:,i], label = false, lw = 2, color = :red)
end

scatter!(t_usados, Signals_valid_derivadas_puntos[:,1], label = "Derivadas usadas Validación", lw = 2, color = :blue)

for i in 2:size(Signals_valid_puntos)[1]
    scatter!(plot_derivadas_puntos, t_usados, Signals_valid_derivadas_puntos[:,i], label = false, lw = 2, color = :blue)
end

scatter!(t_usados, Signals_test_derivadas_puntos[:,1], label = "Derivadas usadas Test", lw = 2, color = :black)

for i in 2:size(Signals_test_puntos)[1]
    scatter!(plot_derivadas_puntos, t_usados, Signals_test_derivadas_puntos[:,i], label = false, lw = 2, color = :black)
end

display(plot_derivadas_puntos)
xlims!(0, 0.1)
# savefig("../ExploracionIntV3/ArquitecturaEspecifica/Imagenes/DerivadasUsadasV2.png")

#################################################################################################################################################

# Obtenemos las interpolaciones de las señales y de las derivadas
itp_signals = get_interpolated(t_usados, Signals_rep_puntos')
itp_derivadas = get_interpolated(t_usados, Signals_rep_derivadas_puntos)

itp_signals_valid = get_interpolated(t_usados, Signals_valid_puntos')
itp_derivadas_valid = get_interpolated(t_usados, Signals_valid_derivadas_puntos)

itp_signals_test = get_interpolated(t_usados, Signals_test_puntos')
itp_derivadas_test = get_interpolated(t_usados, Signals_test_derivadas_puntos)

extra_parameters = itp_signals
extra_parameters2 = itp_derivadas

extra_parameters_valid = itp_signals_valid
extra_parameters2_valid = itp_derivadas_valid

extra_parameters_test = itp_signals_test
extra_parameters2_test = itp_derivadas_test

#################################################################################################################################################
# Veamos las interpolaciones que acabamos de hacer

plot_interploaciones = plot(t_usados, itp_signals[1].(t_usados), label = "Interpolación Señales", xlabel = "t", ylabel = "S(t)", title = "Interpolaciones de las señales", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, color = :red)
for i in 2:size(Signals_rep_puntos)[1]
    plot!(plot_interploaciones, t_usados, itp_signals[i].(t_usados), label = false, lw = 2, color = :red)
end

plot!(t_usados, itp_signals_valid[1].(t_usados), label = "Interpolación Señales Validación", lw = 2, color = :blue)

for i in 2:size(Signals_valid_puntos)[1]
    plot!(plot_interploaciones, t_usados, itp_signals_valid[i].(t_usados), label = false, lw = 2, color = :blue)
end

plot(t_usados, itp_signals_test[1].(t_usados), label = "Interpolación Señales Test", lw = 2, color = :black)

for i in 2:size(Signals_test_puntos)[1]
    plot!(plot_interploaciones, t_usados, itp_signals_test[i].(t_usados), label = false, lw = 2, color = :black)
end

display(plot_interploaciones)
xlims!(0, 0.1)

#################################################################################################################################################

# Todas las señales tienen la misma condición inicial U0 = 1
U0 = ones32(size(Signals_rep)[1])
U0_valid = ones32(size(Signals_valid)[1])
U0_test = ones32(size(Signals_test)[1])

# id actual de la red
actual_id = 40

#Definimos el batch size
batch_size = 30

# Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
train_loader = Flux.Data.DataLoader((Signals_rep, t), batchsize = batch_size)

# Función de activación
activation = relu

# Definimos el rate de dropout en la primera capa
dropout_rate = 0.0

# 40,"[3, 32, 32, 16, 1]",relu,AdamW,30,0.0,0.1,0.0008262385337103576,0.0007237688535536802,0.0

nn = Chain(Dense(3, 32),
            Dense(32, 32, activation),
            Dense(32, 16, activation),
            Dense(16, 1))

# Tomamos un learning rate de 0.001
η = 0.1

# Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
epochs = 100

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial
p
# Leemos los parámetros de la red ya entrenada si es que existen
if isfile("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Parameters/$(actual_id)_Parameters.csv")
    theta = CSV.read("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Parameters/$(actual_id)_Parameters.csv", DataFrame)
    p = Float32.(theta[:,1])
else
    println("No se encontraron los parámetros de la red neuronal")
end

p

# println("Parámetros de la red neuronal", sort(abs.(p)))

##############################################################################################

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

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

# Función que calcula el loss de la red neuronal para un batch de datos y sus respectivos tiempos

# Penalization term
function penalization_term(time_batch,y)
    """
    Función de penalización para tratar de mantener la señal monotonamente decrecente
    En caso que la señal no sea monotonamente decreciente, la penalización es la suma de las diferencias positivas entre los tiempos
    """
    pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
end


function loss_node(batch, time_batch, lamb = 0.0)
    y = Predict_Singals(U0, extra_parameters, extra_parameters2, time_batch)
    return Flux.mse(y, batch') + lamb * sum(p.^2)
     #+ lamb * (penalization_term(time_batch, y))
end

function loss_valid(batch, time_batch, lamb = 0.0)
    y = Predict_Singals(U0_valid, extra_parameters_valid, extra_parameters2_valid, time_batch)
    return Flux.mse(y, batch') + lamb * sum(p.^2)
    #+ lamb * (penalization_term(time_batch, y))
end

function loss_test(batch, time_batch, lamb = 0.0)
    y = Predict_Singals(U0_test, extra_parameters_test, extra_parameters2_test, time_batch)
    return Flux.mse(y, batch') + lamb * sum(p.^2)
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
        forecast_loss = loss_valid(Signals_valid, t)
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

# Loss
println("Loss Entrenamiento: ", loss_node(Signals_rep, t))
println("Loss Validación: ", loss_valid(Signals_valid, t))
println("Loss Test: ", loss_test(Signals_test, t))

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, epochs), opt, cb = callback)
# Guardamos los parámetros
df_parameters = DataFrame(reshape(p, length(p), 1), :auto)

CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/M_layers/ParametrosEspecificos/$(actual_id)_Parameters.csv", df_parameters)

# Guardamos las funciónes de loss
Loss_Matrix = zeros((length(loss), 2))

for i in 1:length(loss)
    Loss_Matrix[i,1] = loss[i]
    Loss_Matrix[i,2] = loss_valid_array[i]
end

df_losses = DataFrame(Loss_Matrix, :auto)
rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

if isfile("../ExploracionIntV3/ArquitecturaEspecifica/Losses/$(actual_id)_losses.csv")
   df_losses = CSV.read("../ExploracionIntV3/ArquitecturaEspecifica/Losses/$(actual_id)_losses.csv", DataFrame)
   loss = df_losses[:,1]
   loss_valid_array = df_losses[:,2]
else
    println("No se encontraron los loss de la red neuronal")
end

CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/ArquitecturaEspecifica/Losses/$(actual_id)_lossesV2.csv", df_losses)

start_index = 20

plots_loss = plot(loss[start_index:end], label = "Loss de entrenamiento", xlabel = "Época", ylabel = "Loss", title = "Loss de entrenamiento", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
plot!(loss_valid_array[start_index:end], label = "Loss de predicción", xlabel = "Época", ylabel = "Loss", title = "Loss de entrenamiento", lw = 2)
savefig("../ExploracionIntV3/ArquitecturaEspecifica/Imagenes/Signals_LossV2.png")

###############################################################################################

# Predicción señales entrenamiento
plot_predictions = plot(t, Signals_rep[1,:], label = "Señales de entrenamiento", xlabel = "t", ylabel = "S(t)", title = "Predicción de señales entrenamiento", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, color = :blue)

for i in 2:5:size(Signals_rep)[1]
    plot!(plot_predictions, t, Signals_rep[i,:], label = false, lw = 2, color = :blue)
end


plot!(toriginal, Predict_Singals(U0[1], [extra_parameters[1]], [extra_parameters2[1]], toriginal), label = "Predicción", lw = 2, color = :red, alpha = 0.5)
plot!(toriginal, Predict_Singals(U0[1:5:end], extra_parameters[1:5:end], extra_parameters2[1:5:end], toriginal), label = false, lw = 2, color = :red, alpha = 0.5)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SignalsEntrenamientoPrediccionInicial.png")
xlims!(0, 0.1)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SignalsEntrenamientoPrediccionInicialZoom.png")


# Predicción señales validación

plot_predictions_valid = plot(t, Signals_valid[1,:], label = "Señales de validación", xlabel = "t", ylabel = "S(t)", title = "Predicción de señales validación", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, color = :blue)

for i in 2:size(Signals_valid)[1]
    plot!(plot_predictions_valid, t, Signals_valid[i,:], label = false, lw = 2, color = :blue)
end

plot!(toriginal, Predict_Singals(U0_valid[1], [extra_parameters_valid[1]], [extra_parameters2_valid[1]], toriginal), label = "Predicción", lw = 2, color = :red, alpha = 0.5)
plot!(toriginal, Predict_Singals(U0_valid, extra_parameters_valid, extra_parameters2_valid, toriginal), label = false, lw = 2, color = :red, alpha = 0.5)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SignalsPrediccionInicialValid.png")
xlims!(0, 0.1)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SignalsPrediccionInicialValidZoom.png")

# Predicción señales de Test
plot_predictions_test = plot(t, Signals_test[1,:], label = "Señales de test", xlabel = "t", ylabel = "S(t)", title = "Predicción de señales test", lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, color = :black)

for i in 2:size(Signals_test)[1]
    plot!(plot_predictions_test, t, Signals_test[i,:], label = false, lw = 2, color = :black)
end

plot!(toriginal, Predict_Singals(U0_test[1], [extra_parameters_test[1]], [extra_parameters2_test[1]], toriginal), label = "Predicción", lw = 2, color = :red, alpha = 0.5)
plot!(toriginal, Predict_Singals(U0_test, extra_parameters_test, extra_parameters2_test, toriginal), label = false, lw = 2, color = :red, alpha = 0.5)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SignalsPrediccionTest.png")
xlims!(0, 0.1)
savefig("../3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Imagenes/SignalsPrediccionTestZoom.png")