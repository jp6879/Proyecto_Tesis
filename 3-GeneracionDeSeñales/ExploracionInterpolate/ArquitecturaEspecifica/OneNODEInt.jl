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

include("/home/juan.morales/ExploracionCompleta/UtilsRepresentative.jl")
# include("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionCompleta/UtilsRepresentative.jl")

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
path_read = "/home/juan.morales/datos_PCA"
# path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

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


function recta(x1, y1, x0 = 0, y0 = 1)
    m = (y1 - y0) / (x1 - x0)
    b = y0
    function evalue_recta(x)
        return m .* x .+ b
    end
    return evalue_recta
end

function calucla_recta(x, x1, y1, x0 = 0, y0 = 1)
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return Float32.(m .* x' .+ b)
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
sampled_sigmas =  [0.01, 0.4, 0.8, 1.0]
lcm_range = 1:50:550

println("Sigmas: ", sampled_sigmas)
println("Lcms: ", collect(lcms)[lcm_range])

# Obtenemos las señales representativas para un conjunto de sigmas y lcms
Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

# Numero de puntos para la validacion
n_valid = 10
# Paso para tomar los tiempos de entrenamiento y validación
step = floor(Int, length(t) / n_valid) + 1

tvalid = t[1:step:end]
ttrain = [t for t in t if t ∉ tvalid]

# En la validación y en el train tenemos que tener el primer y último tiempo
ttrain = vcat(0, ttrain)
tvalid = vcat(tvalid, ttrain[end])

indexes_train = [i for i in 1:length(t) if t[i] in ttrain]
indexes_valid = [i for i in 1:length(t) if t[i] in tvalid]

Signals_valid = Signals_rep[:,indexes_valid]
Signals_train = Signals_rep[:,indexes_train]

# Calculamos primero la recta con estos dos puntos
recta_param_train = calucla_recta(t, t[end], Signals_rep[:,end], t[1], Signals_rep[:,1])'

recta_funcs = []

for i in 1:size(Signals_rep)[1]
    push!(recta_funcs, recta(t[end], Signals_rep[i,end], t[1], Signals_rep[i,1]))
end

Signals_derivadas_train = Signals_rep_derivadas[indexes_train,:]
Signals_derivadas_valid = zeros(size(Signals_valid))

for i in 1:size(Signals_valid)[1]
    Signals_derivadas_valid[i,:] = derivate_signals(tvalid,Signals_valid[i,:])
end

Signals_derivadas_valid = Float32.(Matrix(Signals_derivadas_valid'))

# Normalizamos las derivadas
for i in 1:size(Signals_valid)[1]
    Signals_derivadas_valid[:,i] = Signals_derivadas_valid[:,i] ./ maximum(abs.(Signals_derivadas_valid[:,i]))
end

itp_derivadas = []

# Interpolamos las derivadas
for i in 1:size(Signals_rep)[1]
    push!(itp_derivadas, CubicSpline(tvalid, Signals_derivadas_valid[:,i], extrapl = [2,], extrapr=[2,]))
end

extra_parameters = itp_derivadas
extra_parameters_valid = itp_derivadas
extra_parameters2 = recta_funcs

# Todas las señales tienen la misma condición inicial U0 = 1
U0 = ones32(size(Signals_rep)[1])

# id actual de la red
actual_id = 36

#Definimos el batch size
batch_size = 15

# Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
train_loader = Flux.Data.DataLoader((Signals_rep, t), batchsize = batch_size, shuffle = true)

# Función de activación
activation = tanh_fast

# 27,"[3, 32, 64, 16, 8, 1]",tanh_fast,AdamW,15,0.015395822743798204,0.013012183844752567

nn = Chain(Dense(3, 32, activation),
            Dense(32, 64, activation),
            Dense(64, 16, activation),
            Dense(16, 8, activation),
            Dense(8, 1)
            )

# Tomamos un learning rate de 0.001
η = 5e-3

# Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
epochs = 2000

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

##############################################################################################

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

# Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
function predict_NeuralODE(u0, parametros, parametros2, time_batch)
    # dSdt = NN(S, parametros_extra) 
    function dSdt(u, p, t; parametros_extra = parametros, parametros_extra2 = parametros2)
        parametros_actuales = parametros_extra[t] # Selecciona los parametros extra en el tiempo t
        parametros_actuales_2 = parametros_extra2(t)
        entrada_red = vcat(u, parametros_actuales, parametros_actuales_2) # Concatena los el valor de S(t) con los parametros extra en el tiempo t
        return re(p)(entrada_red) # Regresa la salida de la red neuronal re creada con los parámetros p
    end

    prob = ODEProblem(dSdt, u0, tspan)

    return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-5, abstol = 1e-5)) # Regresa la solución de la ODE
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
    pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
end

function loss_node(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0, extra_parameters, extra_parameters2, time_batch)
    return Flux.mse(y, batch')# + lamb * (penalization_term(time_batch, y))
end

function loss_valid(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0, extra_parameters_valid, extra_parameters2, time_batch)
    return Flux.mse(y, batch')# + lamb * (penalization_term(time_batch, y))
end

# Función de callback para guardar el loss en cada época
global iter = 0
loss = []
loss_valid_array = []
callback = function ()
    global iter += 1
    if iter % (length(train_loader)) == 0
        epoch = Int(iter / length(train_loader))
        actual_loss = loss_node(Signals_train, ttrain)
        forecast_loss = loss_valid(Signals_valid, tvalid)
        println("Epoch = $epoch || Loss: $actual_loss || Loss Forecast: $forecast_loss")
        push!(loss, actual_loss)
        push!(loss_valid_array, forecast_loss)
    end
    return false
end

###############################################################################################################

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, epochs), opt, cb = callback)

# Guardamos los parámetros
df_parameters = DataFrame(reshape(p, length(p), 1), :auto)

CSV.write("/home/juan.morales/ExploracionInterpolate/ArquitecturaEspecifica/Parameters/$(actual_id)_ParametersV2.csv", df_parameters)

# Guardamos las funciónes de loss
Loss_Matrix = zeros((length(loss), 2))

for i in 1:length(loss)
    Loss_Matrix[i,1] = loss[i]
    Loss_Matrix[i,2] = loss_valid_array[i]
end

df_losses = DataFrame(Loss_Matrix, :auto)
rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

CSV.write("/home/juan.morales/ExploracionInterpolate/ArquitecturaEspecifica/Losses/$(actual_id)_lossesV2.csv", df_losses)