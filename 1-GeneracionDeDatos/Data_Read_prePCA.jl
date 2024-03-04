# Programa que utiliza el código de la función GenData para generar los datos de las señales y las distribuciones de tamaños de compartimientos
# y luego los guarda en un archivo CSV con un formato específico para luego ser leídos realizar el pre procesamiento de Principal Component Analysis (PCA)

# Importacion de paquetes necesarios
include("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData_PCA_Final\\Gen_Read_Data-Hahn.jl")
using CSV
using DataFrames
using Plots
using QuadGK

##########################################################################################
# Traemos los parametros fijos desde otro programa
include("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\1-GeneracionDeDatos\\Parametros.jl")

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
    """Función que re organiza los datos de entrada y salida en un formato que podemos guardar en un archivo csv
    
    Parámetros:
        old_matrix: matriz de datos
        old_shape: tamaño de la matriz de datos
        new_shape: nuevo tamaño de la matriz de datos
    
    Retorna:
        new_matrix: matriz de datos re organizada
    """
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

# Los datos de las señales se generan en un archivo CSV donde tiene en la primera fila
# el indice de las señales como x1, x2 ... xn
# y en cada columna los datos de cada una
# tiempos quedan guardados en el programa Parametros.jl

CSV.write(path_save * "\\dataSignals.csv", df_dataSignals)
CSV.write(path_save * "\\dataProbd.csv", df_dataProbd)

# Fin del programa