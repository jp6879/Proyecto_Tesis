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

# Funcion que tomas algunas señales para utilizarlas en el entrenamiento de la NODE, recibe el numero de señales que quermos tomar y de donde las queremos tomar
# En este caso se toman espaciadamente las señales según el número de señales que se quieren tomar
# Recordando que los datos comienzan en σ = 0.1 y lcm = 0.5 y va en aumento de lcm hasta 6 y luego pasa al siguiente σ
function Get_Signals_Test(numSignals, path_read, muestreo_corto, muestreo_largo)
    # Leemos las señales del archivo, las transformamos en Float32 para el manejo de las NODE
    dataSignals = Float32.(GetSignals(path_read))
    # Tomamos las señales que vamos a utilizar separando según el número de señales que queremos tomar
    Signals_test = Float32.(Matrix(dataSignals[:,1:Int(length(dataSignals[1,:])/numSignals):end]'))
    # Tomamos un subconjunto de tiempos para agilizar el entrenamiento, tiempos cortos es decir el muestreo hasta 0.1 s
    Signals_test_short = Signals_test[:,1:muestreo_corto:1000]
    # Tiempos largos es decir el muestreo después de 0.1 s
    Signals_test_long = Signals_test[:,1001:muestreo_largo:end]
    # Concatenamos las señales con el nuevo muestreo para agilizar el entrenamiento
    Signals_test = hcat(Signals_test_short, Signals_test_long)
    return Signals_test
end

##############################################################################################
# Función que idientifca las señales utilizadas, es decir los parámetros lcm y σ
function Get_Signals_Test_Parameters(numSignals,lcms,σs)
    dim1 = dimlcm = length(lcms)
    dim2 = dimσ = length(σs)

    # Inicializamos los arreglos que vamos a utilizar
    column_lcm = zeros(dim1*dim2)
    column_σs = zeros(dim1*dim2)
    # Convertimos los iterables de lcm y sigma con los cuales generamos las señales en arreglos
    aux_lcm = collect(lcms)
    aux_σs = collect(σs)

    # Asignamos los valores de lcm y sigma ordenados tal cual están las señales, es decir los datos comienzan en σ = 0.1 y lcm = 0.5 y va en aumento de lcm hasta 6 y luego pasa al siguiente σ
    for i in 1:dim1
        for j in 1:dim2
            column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
            column_σs[(i - 1)*dim2 + j] = aux_σs[j]
        end
    end
    # Con esto ya tenemos los arrelgos de los parámetros que se utilizaron para generar las señales
    column_lcm_test = column_lcm[1:Int(length(column_lcm)/numSignals):end]
    column_σs_test = column_σs[1:Int(length(column_σs)/numSignals):end]
    
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
function Get_Signals_Data_Training(path_read, numSignals, lcms, σs, muestreo_corto, muestreo_largo)
    # Obtenemos las señales que vamos a utilizar
    Signals_test = Get_Signals_Test(numSignals, path_read, muestreo_corto, muestreo_largo)

    # Obtenemos los parámetros con los que se generaron las señales
    column_lcm_test, column_σs_test = Get_Signals_Test_Parameters(numSignals,lcms,σs)

    # Calculamos las derivadas de las señales
    Signals_test_derivadas = zeros(size(Signals_test))
    for i in 1:size(Signals_test)[1]
        Signals_test_derivadas[i,:] = derivate_signals(t,Signals_test[i,:])
    end
    Signals_test_derivadas = Float32.(Matrix(Signals_test_derivadas'))
    # Normalizamos las derivadas de las señales
    for i in 1:size(Signals_test)[1]
        Signals_test_derivadas[:,i] = Signals_test_derivadas[:,i] ./ maximum(abs.(Signals_test_derivadas[:,i]))
    end
    return Signals_test, Signals_test_derivadas, column_lcm_test, column_σs_test
end

##############################################################################################
# Vamos a tomar 5 señales para entrenar la NODE
numSignals = 5
Signals_test, Signals_test_derivadas, column_lcm_test, column_σs_test = Get_Signals_Data_Training(path_read, numSignals, lcms, σs, muestreo_corto, muestreo_largo)

# Definimos el batch_size
batch_size = 3

t = vcat(t_short, t_long)
# Indice de los tiempos desde el cual vamos a predecir
idxforecast = 16

# Tomamos los tiempos de entrenamiento y de predicción
tforecast = t[idxforecast:end]
ttrain = t[1:idxforecast-1]

Signals_test_train = Signals_test[:,1:idxforecast-1]
Signals_test_valid = Signals_test[:,idxforecast:end]

Signals_test_derivadas_train = Signals_test_derivadas[1:idxforecast-1,:]
Signals_test_derivadas_valid = Signals_test_derivadas[idxforecast:end,:]

train_loader = Flux.Data.DataLoader((Signals_test_train, ttrain), batchsize = batch_size)

nn = Chain(Dense(2,32, tanh_fast),
            Dense(32,64, tanh_fast),
            Dense(64,16, tanh_fast),
            Dense(16,1))

 # Tomamos un learning rate de 0.001
 η = 5e-3

 # Vamos a tomar 2000 épocas para entrenar todas las arquitecturas
 epochs = 5

 # Todas las señales tienen la misma condición inicial
 U0 = ones32(size(Signals_test_train)[1])

 # Definimos los parámetros extra 
extra_parameters = Signals_test_derivadas_train

# Tiempo sobre el cual resolver
tspan = (0f0, 1f0)

# Parametrizar indices para poder acceder a los parametros extras de la red neuronal
f(x,p) = round(Int, x * (length(p) - 1)) + 1

# Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial
p, re = Flux.destructure(nn) 

# Leemos parámetros de la arquitectura 29/30
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
    # Iteramos sobre las condiciones iniciales que son las mismas para todas las señles
    # y además sobre las derivadas punto a punto de cada señal
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
    return Flux.mse(y, batch')
end

# Función de callback para guardar el loss en cada época
global iter = 0
global epoch = 0
loss = []
callback = function ()
    global iter += 1
    if iter % (length(train_loader)) == 0
        global epoch += 1
        actual_loss = loss_node(Signals_test_train, ttrain)
        println("Epoch = $epoch || Loss: $actual_loss")
        push!(loss, actual_loss)
    end
    return false
end

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p,U0), ncycle(train_loader, epochs), opt, cb = callback)

##############################################################################################
# Vamos a hacer un plot de las señales de entrenamiento y sus predicciones
colors = [:blue, :blue, :blue, :blue, :blue]
colors_prediction = [:red, :red, :red, :red, :red]

linspace = range(-0.05, 1.1, length = 1000)
constant = ones(length(linspace))*t[15]

Plots_Train = []

# Plot 2x2
for i in 1:length(colors)
    if i == 1        
        actual_plot = plot(vcat(t,tforecast), Signals_test[i,:], label = "Señal Real", xlabel = L"$t$ (s)", ylabel = L"S(t)", title = L"Predicción de señal $l_{cm} = $" * " $(column_lcm_test[i]), " * L"σ = " * "$(column_σs_test[i])", lw = 2, color = colors[i], tickfontsize=12, labelfontsize=11, legendfontsize=1, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, legend_position = :bottomleft)
        plot!(actual_plot, t, Predict_Singals(U0[i], Signals_test_derivadas_train[:,i], t), color = :red, label = "Predicción datos entrenamiento", lw = 2, tickfontsize=12, labelfontsize=11, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
        plot!(actual_plot, tforecast, Predict_Singals(U0[i], Signals_test_derivadas_train[:,i], tforecast), color = colors_prediction[i], label = "Predicción", ls = :dash, lw = 2, tickfontsize=12, labelfontsize=11, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
        plot!(actual_plot, constant, linspace, label = false, lw = 2, ls = :dash, color = :gray)
        push!(Plots_Train, actual_plot)
    else
        actual_plot = plot(vcat(t,tforecast), Signals_test[i,:], label = "Señal Real", xlabel = L"$t$ (s)", ylabel = L"S(t)", title = L"Predicción de señal $l_{cm} = $" * " $(column_lcm_test[i]), " * L"σ = " * "$(column_σs_test[i])", lw = 2, color = colors[i], tickfontsize=12, labelfontsize=11, legendfontsize=1, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, legend_position = :best)
        plot!(actual_plot, t, Predict_Singals(U0[i], Signals_test_derivadas_train[:,i], t), color = :red, label = "Predicción datos entrenamiento", lw = 2, tickfontsize=12, labelfontsize=11, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
        plot!(actual_plot, tforecast, Predict_Singals(U0[i], Signals_test_derivadas_train[:,i], tforecast), color = colors_prediction[i], label = "Predicción", ls = :dash, lw = 2, tickfontsize=12, labelfontsize=11, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
        plot!(actual_plot, constant, linspace, label = false, lw = 2, ls = :dash, color = :gray)
        push!(Plots_Train, actual_plot)
    end
    # display(plot(t, Signals_test[i,:], label = "Señal $i", xlabel = L"t", ylabel = L"S(t)", title = "Predicción de señal lcm = $(column_lcm_test[i]), σ = $(column_σs_test[i])", lw = 2, color = colors[i], markershape = :circle, tickfontsize=12, labelfontsize=15, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10))
    # display(plot!(t, Predict_Singals(U0[i], Signals_test_derivadas[:,i], t), color = colors_prediction[i], label = "Predicción $i", markershape = :star6, ls = :dash, lw = 2))
    # savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion\\Figuras_arquitectura_30\\Arquitectura_30_Signal_$i.pdf")
end

combined_plot = plot(Plots_Train[1], Plots_Train[2], Plots_Train[3], Plots_Train[4], layout=(2, 2), size = (1280,800), margin = 5mm)
savefig(combined_plot,"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion\\Figuras_arquitectura_30\\Train_4.pdf")
savefig(Plots_Train[5], "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion\\Figuras_arquitectura_30\\Train_5.pdf")

# plot(t, Signals_test[1,:], label = "Señal 1", xlabel = L"t", ylabel = L"S(t)", title = "Predicción de señales", lw = 2, color = :blue, markershape = :circle)
# plot!(t, Predict_Singals(U0[1], Signals_test_derivadas[:,1], t), color = :blue, label = "Predicción 1", markershape = :star6, ls = :dash)

loss_node(Signals_test, t)

Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, epochs), opt, cb = callback)

# Recuperemos el loss de la NODE
df_loss = CSV.read("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion\\Loss_list\\Losses_arquitecturasNODE.csv", DataFrame)

loss_list = df_loss[!,"id 29"]

vcat(loss_list, loss)

plot(loss_list,label = "Datos de entrenamiento", xlabel = "Épocas", ylabel = "Loss", lw = 2,tickfontsize=12, labelfontsize=15, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, rightmargin = 5mm)
savefig("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion\\Figuras_arquitectura_30\\Loss_Train.pdf")


# Fin del programa

##############################################################################################
# Funciones que voy a utilizar ocomo validación

# # Funcion que tomas algunas señales para utilizarlas en el entrenamiento de la NODE, recibe el numero de señales que quermos tomar y de donde las queremos tomar
# function Get_Signals_Test_Valid(numSignals, path_read, muestreo_corto, muestreo_largo)
#     dataSignals = Float32.(GetSignals(path_read))
#     Signals_test = Float32.(Matrix(dataSignals[:,10:Int(length(dataSignals[1,:])/numSignals):end]'))
#     Signals_test_short = Signals_test[:,1:muestreo_corto:1000]
#     Signals_test_long = Signals_test[:,1001:muestreo_largo:end]
#     Signals_test = hcat(Signals_test_short, Signals_test_long)
#     return Signals_test
# end

##############################################################################################
# Función que idientifca las señales que se van a utilizar para el entrenamiento de la NODE con lcm y σ
# function Get_Signals_Test_Valid_Parameters(numSignals,lcms,σs)
#     dim1 = dimlcm = length(lcms)
#     dim2 = dimσ = length(σs)

#     column_lcm = zeros(dim1*dim2)
#     column_σs = zeros(dim1*dim2)
#     aux_lcm = collect(lcms)
#     aux_σs = collect(σs)

#     for i in 1:dim1
#         for j in 1:dim2
#             column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
#             column_σs[(i - 1)*dim2 + j] = aux_σs[j]
#         end
#     end

#     column_lcm_test = column_lcm[10:Int(length(column_lcm)/numSignals):end]
#     column_σs_test = column_σs[10:Int(length(column_σs)/numSignals):end]
    
#     return column_lcm_test, column_σs_test
# end

# # Función que devuelve señales de validacion, sus derivadas y los parámetros con los que se generaron
# function Get_Signals_Data_Valid(path_read, numSignals, lcms, σs, muestreo_corto, muestreo_largo)
#     Signals_test = Get_Signals_Test_Valid(numSignals, path_read, muestreo_corto, muestreo_largo)
#     column_lcm_test, column_σs_test = Get_Signals_Test_Valid_Parameters(numSignals,lcms,σs)
#     Signals_test_derivadas = zeros(size(Signals_test))
#     for i in 1:size(Signals_test)[1]
#         Signals_test_derivadas[i,:] = derivate_signals(t,Signals_test[i,:])
#     end
#     Signals_test_derivadas = Float32.(Matrix(Signals_test_derivadas'))
#     for i in 1:size(Signals_test)[1]
#         Signals_test_derivadas[:,i] = Signals_test_derivadas[:,i] ./ maximum(abs.(Signals_test_derivadas[:,i]))
#     end
#     return Signals_test, Signals_test_derivadas, column_lcm_test, column_σs_test
# end


# Signals_valid, Signals_valid_derivadas, column_lcm_valid, column_σs_valid = Get_Signals_Data_Valid(path_read, numSignals, lcms, σs, muestreo_corto, muestreo_largo)

# Plots_Valid = []

# # Create a 2x2 plot
# for i in 1:length(colors)
#     actual_plot = plot(t, Signals_valid[i,:], label = "Señal validación $i", xlabel = L"t", ylabel = L"S(t)", title = L"Predicción de señal $l_{cm} = $" * " $(column_lcm_valid[i]), " * L"σ = " * "$(column_σs_valid[i])", lw = 2, color = colors[i], markershape = :circle, tickfontsize=12, labelfontsize=15, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
#     plot!(actual_plot, t, Predict_Singals(U0[i], Signals_valid_derivadas[:,i], t), color = colors_prediction[i], label = "Predicción $i", markershape = :star6, ls = :dash, lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
#     push!(Plots_Valid, actual_plot)
# end

# combined_plot = plot(Plots_Valid[1], Plots_Valid[2], Plots_Valid[3], Plots_Valid[4], layout=(2, 2), size = (1280,800), margin = 5mm)
# savefig(combined_plot,"C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\TrackVersionsProgramsGit\\2-GeneraciónSeñalesNODE\\Mini-Batch-Test\\Exploracion\\Figuras_30\\Valid_4.pdf")
# savefig(Plots_Valid[5], "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\TrackVersionsProgramsGit\\2-GeneraciónSeñalesNODE\\Mini-Batch-Test\\Exploracion\\Figuras_30\\Valid_5.pdf")


# loss_node(Signals_valid, t)

##############################################################################################


