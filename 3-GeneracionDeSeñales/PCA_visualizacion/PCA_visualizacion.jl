using Plots
using LaTeXStrings
using MultivariateStats
using DataFrames
using CSV
using Statistics
using Measures
using StatsPlots
include("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/Exploracion Paralelizada/Representativos/UtilsRepresentative.jl")
#------------------------------------------------------------------------------------------
# Parámetros fijos

# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 5000
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 50

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.01:1

#------------------------------------------------------------------------------------------

function GetSignals(path_read)
    dataSignals = CSV.read(path_read * "\\dataSignals.csv", DataFrame)
    dataSignals = Matrix(dataSignals)
    return dataSignals
end

# Función que lee los datos de las distribuciones de probabilidad
function GetProbd(path_read)
    dataProbd = CSV.read(path_read * "\\dataProbd.csv", DataFrame)
    dataProbd = Matrix(dataProbd)
    return dataProbd
end

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\1-GeneracionDeDatos\\Datos_Final\\datos_PCA"

dataSignals = GetSignals(path_read)
# dataProbd = GetProbd(path_read)

#------------------------------------------------------------------------------------------
# Funcion que centra los datos

function CenterData(Non_C_Matrix)
	data_matrix = Non_C_Matrix
	col_means = mean(data_matrix, dims = 1)
	centered_data = data_matrix .- col_means
	return centered_data
end

#------------------------------------------------------------------------------------------

# PCA para ciertos datos

function PCA_Data(dataIN)

    # Primero centramos los datos
    dataIN_C = CenterData(dataIN)

    # Esto ya hace PCA sobre la matriz dada donde cada observación es una columna de la matriz
    pca_model = fit(PCA, dataIN_C)

    # Esta instancia de PCA tiene distintas funciones como las siguientes

    #projIN = projection(pca_model) # Proyección de los datos sobre los componentes principales

    # Vector con las contribuciones de cada componente (es decir los autovalores)
    pcsIN = principalvars(pca_model)

    # Obtenemos la variaza en porcentaje para cada componente principal
    explained_varianceIN = pcsIN / sum(pcsIN) * 100

    # Grafiquemos esto para ver que tan importante es cada componente principal
    display(Plots.bar(explained_varianceIN, title="Varianza en porcentaje datos entrada",label = false, xlabel="Componente principal", ylabel="Varianza (%)"))

    reduced_dataIN = MultivariateStats.transform(pca_model, dataIN_C)

    return reduced_dataIN, pca_model

end

#------------------------------------------------------------------------------------------
# Identificación de los datos reducidos con señales y distribuciones de probabilidad originales

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

#------------------------------------------------------------------------------------------
# Modelos reducidos de las señales y distribuciones de probabilidad

reduced_data_Signals, pca_model_signals = PCA_Data(dataSignals)
# reduced_data_Probd, pca_model_probd = PCA_Data(dataProbd)

df_PCA_Signals = DataFrame(
		pc1 = reduced_data_Signals[1, :],
	    pc2 = reduced_data_Signals[2, :],
	    σs = column_σs,
	    lcm = column_lcm,
	)

plot_lcms_S = @df df_PCA_Signals StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :σs,
    marker = (0.2,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para S(t) distingue σ",
)

plot_lcms_S = @df df_PCA_Signals StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.2,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para S(t) distingue lcm",
)


#------------------------------------------------------------------------------------------
# Vamos a seleccionar algunos σ, de estos vamos a samplear algunos valores de lcm en una tira de valores y así hasta tener 80 señales
# que representen aproximadamente bien el espacio de parámetros
# sampled_sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1]
sampled_sigmas = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
sampled_lcm = collect(lcms[1:25:250])
# collect(lcms)
# lcm_range = 1:1:275
# println(column_lcm[lcm_range])
println(sampled_lcm)


# lcm_range = 126:25:250
# lcm_range = 1:25:250
# rangos = []
# lcms_explorados = []

find_rows_sigma = []
find_rows_lcm = []
selected_rows = []

for σ in sampled_sigmas
    push!(find_rows_sigma, findall(x -> x == σ, df_PCA_Signals.σs))
end

for lcm in sampled_lcm
    push!(find_rows_lcm, findall(x -> x == lcm, df_PCA_Signals.lcm))
end
selected_rows = intersect(vcat(find_rows_sigma...), vcat(find_rows_lcm...))

# for σ in sampled_sigmas
#     find_rows = findall(x -> x == σ, df_PCA_Signals.σs)
#     find_lcms = df_PCA_Signals.lcm[find_rows]
#     println(length(find_rows))
#     println(find_rows)
#     push!(rangos, find_rows)
#     push!(lcms_explorados, find_lcms)
# end
# lcms_explorados
# rangos

# println(length(rangos[1][:]))

# rangos = vcat(rangos...)

xs = df_PCA_Signals.pc1[selected_rows]
ys = df_PCA_Signals.pc2[selected_rows]

plot_lcms_S = @df df_PCA_Signals StatsPlots.scatter(
    :pc1,
    :pc2,
    marker = (0.2,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para S(t)",
)

scatter!(plot_lcms_S, xs, ys, label = false, color = "red", markersize=5,legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=8, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
savefig("PCA_SignalsRepresent_01.png")
#------------------------------------------------------------------------------------------

t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))
    
# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 4

# Esto da 100 tiempos 50 puntos desde 0 a 0.1 y 25 puntos desde 0.1 a 1
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]

t = vcat(t_short, t_long)

# Obtenemos las señales representativas para un conjunto de sigmas y lcms
Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

#------------------------------------------------------------------------------------------

pl = scatter(t, Signals_rep[1,:], label = "lcm = $(column_lcm_rep[1]) sigma = $(column_sigmas_rep[1])")
for i in 2:size(Signals_rep)[1]
    if i == size(Signals_rep)[1]
        scatter!(pl, t, Signals_rep[i,:], label = "lcm = $(column_lcm_rep[i]) sigma = $(column_sigmas_rep[i])")
    else
        scatter!(pl, t, Signals_rep[i,:], label = false)
    end
end

pl