# Programa que entrena una NODE con una arquitectura específica para la predicción de las señales de Hah a partir de datos previos de la señal y sus derivadas

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

# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto = 50 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 10
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]
t = vcat(t_short, t_long)

# Parametros que se varian
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.01:1

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\1-GeneracionDeDatos\\Datos_Final\\datos_PCA"

##############################################################################################
# Función para leer las señales
function GetSignals(path_read)
    dataSignals = CSV.read(path_read * "\\dataSignals.csv", DataFrame)
    dataSignals = Matrix(dataSignals)
    return dataSignals
end

# Vamos a tomar únicamente un rango de señales para hacer la prueba
function Get_Signals_Test(rango, path_read, muestreo_corto, muestreo_largo)
    dataSignals = Float32.(GetSignals(path_read))
    Signals_test = Float32.(Matrix(dataSignals[:,rango]'))
    Signals_test = Float32.(Matrix(dataSignals[:,rango]'))
    Signals_test_short = Signals_test[:,1:muestreo_corto:1000]
    Signals_test_long = Signals_test[:,1001:muestreo_largo:end]
    Signals_test = hcat(Signals_test_short, Signals_test_long)
    return Signals_test
end

##############################################################################################
# Función que idientifca las señales que se van a utilizar para el entrenamiento de la NODE con lcm y σ
function Get_Signals_Test_Parameters(rango,lcms,σs)
    dim1 = dimlcm = length(lcms)
    dim2 = dimσ = length(σs)

    column_lcm = zeros(dim1*dim2)
    column_σs = zeros(dim1*dim2)
    aux_lcm = collect(lcms)
    aux_σs = collect(σs)

    for i in 1:dim1
        for j in 1:dim2
            column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
            column_σs[(i - 1)*dim2 + j] = aux_σs[j]
        end
    end

    column_lcm_test = column_lcm[rango]
    column_σs_test = column_σs[rango]
    
    return column_lcm_test, column_σs_test
end

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

# Función que devuelve señales de prueba, sus derivadas y los parámetros con los que se generaron
function Get_Signals_Data_Training(path_read, rango, lcms, σs, muestreo_corto, muestreo_largo)
    Signals_test = Get_Signals_Test(rango, path_read, muestreo_corto, muestreo_largo)
    column_lcm_test, column_σs_test = Get_Signals_Test_Parameters(rango,lcms,σs)
    Signals_test_derivadas = zeros(size(Signals_test))
    for i in 1:size(Signals_test)[1]
        Signals_test_derivadas[i,:] = derivate_signals(t,Signals_test[i,:])
    end
    Signals_test_derivadas = Float32.(Matrix(Signals_test_derivadas'))
    for i in 1:size(Signals_test)[1]
        Signals_test_derivadas[:,i] = Signals_test_derivadas[:,i] ./ maximum(abs.(Signals_test_derivadas[:,i]))
    end
    return Signals_test, Signals_test_derivadas, column_lcm_test, column_σs_test
end

##############################################################################################
# Vamos a tomar un cierto rango de señales
rango = 1:100
rango = 101:201

Signals_test, Signals_test_derivadas, column_lcm_test, column_σs_test = Get_Signals_Data_Training(path_read, rango, lcms, σs, muestreo_corto, muestreo_largo)

plot(t, Signals_test[1,:], label = "$(column_lcm_test[1]) $(column_σs_test[1])", xlabel = "Tiempo", ylabel = "Señal", title = "Señales de prueba", lw = 2, )
plot!(t, Signals_test[2,:], label = "$(column_lcm_test[2]) $(column_σs_test[2])", lw = 2)
plot!(t, Signals_test[3,:], label = "$(column_lcm_test[3]) $(column_σs_test[3])", lw = 2)

plot!(t, Signals_test[end,:], label = "$(column_lcm_test[end]) $(column_σs_test[end])", lw = 2)
plot!(t, Signals_test[end-1,:], label = "$(column_lcm_test[end-1]) $(column_σs_test[end-1])", lw = 2)
plot!(t, Signals_test[end-2,:], label = "$(column_lcm_test[end-2]) $(column_σs_test[end-2])", lw = 2)

##############################################################################################
# Definimos el batch_size
batch_size = 5

t = vcat(t_short, t_long)
# Seteamos un tiempo de predicción y un tiempo de entrenamiento
tforecast = t[16:end]
t

Signals_test_train = Signals_test[:,1:15]
Signals_test_valid = Signals_test[:,16:end]

Signals_test_derivadas_train = Signals_test_derivadas[1:15,:]
Signals_test_derivadas_valid = Signals_test_derivadas[16:end,:]

train_loader = Flux.Data.DataLoader((Signals_test_train, t), batchsize = batch_size)





nn = Chain(Dense(2,32, tanh),
            Dense(32,64, tanh),
            Dense(64,16, tanh),
            Dense(16,1))

 # Tomamos un learning rate de 0.001
 η = 5e-3

 # Vamos a tomar 2000 épocas para entrenar todas las arquitecturas
 epochs = 250

 # Todas las señales tienen la misma condición inicial
 U0 = ones32(size(Signals_test_train)[1])

 # Definimos los parámetros extra 
extra_parameters = Signals_test_derivadas_train

# Tiempo sobre el cual resolver
tspan = (0f0, 1f0)

# Parametrizar indices para poder acceder a los parametros extras de la red neuronal
f(x,p) = round(Int, x * (length(p) - 1)) + 1

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

# Leemos parámetros de la arquitectura 29
θ = CSV.read("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion\\Parameters\\Parameters_arquitectura_29.csv", DataFrame)

p = Float32.(θ[:,1])

##############################################################################################

# Optimizardor
opt = AdamW(η)

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
function loss_node(batch, time_batch)
    y = Predict_Singals(U0, extra_parameters, time_batch)
    return mean((y .- batch') .^ 2)
end

# Función de callback para guardar el loss en cada época
global iter = 0
global epoch = 0
loss = []
callback = function ()
    global iter += 1
    if iter % (length(train_loader)) == 0
        global epoch += 1
        actual_loss = loss_node(Signals_test_train, t)
        println("Epoch = $epoch || Loss: $actual_loss")
        push!(loss, actual_loss)
    end
    return false
end

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p,U0), ncycle(train_loader, epochs), opt, cb = callback)