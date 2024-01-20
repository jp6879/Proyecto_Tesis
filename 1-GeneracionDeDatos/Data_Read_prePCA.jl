# Programa que utiliza el código de la función GenData para generar los datos de las señales y las distribuciones de tamaños de compartimientos
# y luego los guarda en un archivo CSV con un formato específico para luego ser leídos realizar el pre procesamiento de Principal Component Analysis (PCA)

# Importacion de paquetes necesarios
include("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData_PCA_Final\\Gen_Read_Data-Hahn.jl")
using CSV
using DataFrames
using Plots
using QuadGK

##########################################################################################
# En este momento decidimos cuales son los parámetros que dejamos fijos al momento de generar los datos
# Se dejan fijos el numero de compartimientos N, el rango de tamaños de compartimientos l0 y lf, el tiempo final de simulación lf
# Además se muestrea el tiempo en dos partes, una corta y una larga para tener una mejor resolución en la parte inicial de la señal
# Siendo esta imporatnte para diferenciar las señales de diferentes distribuciones de tamaños de compartimientos en el caso de que estas sean muy similares (lcm y σ grandes)

# Parámetros fijos
N = 5000
time_sample_lenght_long = 1000
time_sample_lenght_short = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 45

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N) # Esto nos da un muestreo de 0,008998 μm en lc
t_short = collect(range(0, 0.1, length = time_sample_lenght_short)) # Muestreo corto de 0.1 ms
t_long = collect(range(0.1, 1, length = time_sample_lenght_long)) # Muestreo largo de 10 ms

# Concatenamos los tiempos para tener un muestreo completo 
t = vcat(t_short, t_long)

# Parametros que se varian, estos se corresponden a la mediana y la desviación estándar de la distribución de tamaños de compartimientos lcms en μm y σs adimensionales

lcms = 0.5:0.01:6
σs = 0.01:0.01:1

##########################################################################################
# Generación de datos en CSV para cada combinación de parámetros en el path especificado, este va a ser el mismo que use para leer los datos
# Estos son los datos de cada señal y distribución de tamaños de compartimientos especificando cada combinación de lcm y σ
path = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_crudos"

# Dejar comentado si ya se generaron los datos
# GenCSVData(N, time_sample_lenght, l0, lf, t, lcms, σs, path)

##########################################################################################
# Como tenemos los datos en CSV usamos la función de lectura de datos para leerlos y que nos devuelva todas las señales y distribuciones de tamaños de compartimientos
Probabilitys, Signals = ReadCSVData(N, time_sample_lenght, l0, lf, t, lcms, σs, path)

# Para dejar guardado todas las señales y distribuciones de tamaños de compartimientos en un archivo CSV para luego ser leídos y pre procesados con PCA
# hay que re organizar los datos

length_σs =  length(σs)
length_lcms = length(lcms)
length_t = length(t)
max_length = maximum(length.([t, lc]))

function reshape_data(old_matrix, old_shape, new_shape)

    old_matrix = old_matrix

    dim1, dim2, dim3 = old_shape

    new_matrix = zeros(Float64, dim3, dim1*dim2)

    for i in 1:dim1
        for j in 1:dim2
            for k in 1:dim3
                new_matrix[k,(i - 1) * dim2 + j] = old_matrix[i,j,k]
            end
        end
    end

    return new_matrix

end

# Nuevo tamaño de los datos
new_size = (length_σs * length_lcms, max_length)

# Ahora si tenemos los datos de entrada y salida es decir las señales y las distribuciones de probabilidad en un formato que podemos guardar en un archivo csv
# estos datos son todas las señales generadas en las columnas de una matriz
# y para el otro archivo las distribuciones de tamaños de compartimientos en las columnas de una matriz
dataSignals = reshape_data(Signals, size(Signals), new_size)
dataProbd = reshape_data(Probabilitys, size(Probabilitys), new_size)

# En un momento para tener un DataFrame llenamos los datos de la señal con 0s los sacamos de cada columna.
dataSignals = dataSignals[1:length_t, :]


# Guardamos estos datos en los archivos CSV
# Elegir el path donde se van a guardar los datos
path_save = "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\1-GeneracionDeDatos\\Datos_Final\\datos_PCA"

df_dataSignals = DataFrame(dataSignals, :auto)
df_dataProbd = DataFrame(dataProbd, :auto)

CSV.write(path_save * "\\dataSignals.csv", df_dataSignals)
CSV.write(path_save * "\\dataProbd.csv", df_dataProbd)

# Fin del programa