## Programa para ver las señales con las cuales se exploraron los parámetros en Paralel_training_NODE_batching
using Plots
using DataFrames
using CSV

###################################################################################

# Función para leer las señales

function GetSignals(path_read)
    dataSignals = CSV.read(path_read * "/dataSignals.csv", DataFrame)
    dataSignals = Matrix(dataSignals)
    return dataSignals
end
###################################################################################

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


# Parámetros fijos
# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de tiempos
N = 5000
time_sample_lenght = 1000

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 45

# Rango de tamaños medios de correlación en μm y de desviaciones estándar
lcms = 0.5:0.01:6
sigmas = 0.01:0.01:1

# Vista de las señales

t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))

# Concatenate t_short and t_long
t = vcat(t_short, t_long)

# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto = 100 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 1
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]

t = vcat(t_short, t_long)

path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

# Voy a tomar 100 señales elegidas tomando lcm de 5 en 5 hasta 5.45 μm y manteniendo σ = 1.0
# rango = 100:500:50000
rango = 100:200:10000
Signals_test, Signals_test_derivadas, column_lcm_test, column_sigmass_test = Get_Signals_Data_Training(path_read, rango, lcms, sigmas, muestreo_corto, muestreo_largo)

idx_forecast = 25

tforecast = t[idx_forecast:end]
t = t[1:idx_forecast-1]

Signals_test_train = Signals_test[:,1:idx_forecast-1]
Signals_test_valid = Signals_test[:,idx_forecast:end]

pl = scatter(t, Signals_test_train[1,:], xlabel = "t (s)", ylabel = "S(t)", color = :cyan, lw = 3, label = "σ = $(column_sigmass_test[1]) lcm = $(column_lcm_test[1])", tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
scatter!(tforecast, Signals_test_valid[1,:], color = :red, lw = 3, label = false)
for i in 2:50
            
    # if i == 50
    #     scatter!(t, Signals_test_train[i,:], color = :green, lw = 3, label = "σ = $(column_sigmass_test[i]) lcm = $(column_lcm_test[i])")
    #     scatter!(tforecast, Signals_test_valid[i,:], color = :red, lw = 3, label = false)
    if i == 50
        scatter!(t, Signals_test_train[i,:], color = :yellow, lw = 3, label = "σ = $(column_sigmass_test[i]) lcm = $(column_lcm_test[i])")
        scatter!(tforecast, Signals_test_valid[i,:], color = :red, lw = 3, label = false)
    else
        scatter!(t, Signals_test_train[i,:], color = :blue, lw = 3, label = false)
        scatter!(tforecast, Signals_test_valid[i,:], color = :red, lw = 3, label = false)
    end
end

display(pl)

savefig("Señales_Exploracion2.png")
savefig("Señales_Exploracion2.pdf")


pl = scatter(t,Signals_test_train[1,:], xlabel = "t (s)", ylabel = "S(t)", color = :blue, lw = 3)
scatter!(tforecast,Signals_test_valid[1,:], color = :red, lw = 3)