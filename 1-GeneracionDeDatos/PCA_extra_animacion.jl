using Plots
using LaTeXStrings
using MultivariateStats
using DataFrames
using CSV
using Statistics
using Measures
using StatsPlots
#------------------------------------------------------------------------------------------
# Parámetros fijos

# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 1500
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

df_PCA_Probd = DataFrame(
    pc1 = reduced_data_Probd[1, :],
    pc2 = reduced_data_Probd[2, :],
    σs = column_σs,
    lcm = column_lcm,
)

#------------------------------------------------------------------------------------------
# Vamos a seleccionar algunos σ, de estos vamos a samplear algunos valores de lcm en una tira de valores y así hasta tener 80 señales
# que representen aproximadamente bien el espacio de parámetros
sampled_sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1]

rangos = []

for σ in sampled_sigmas
    find_rows = findall(x -> x == σ, df_PCA_Signals.σs)[1:25:250]
    push!(rangos, find_rows)
end

println(length(rangos[1][:]))

rangos = vcat(rangos...)


xs = df_PCA_Signals.pc1[rangos]
ys = df_PCA_Signals.pc2[rangos]

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
savefig("PCA_RepSignals.png")
#------------------------------------------------------------------------------------------
# Tratemos de buscar un conjunto de señales
find_rows_sigma1 = findall(x -> x == 1, df_PCA_Signals.σs)
xs1 = df_PCA_Signals.pc1[find_rows_sigma1][1:2:100]
ys1 = df_PCA_Signals.pc2[find_rows_sigma1][1:2:100]

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

scatter!(plot_lcms_S, xs1, ys1, label = "σ = 1", color = "red", markersize=5,legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=8, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)

savefig("PCA_SignalsSigma1_V2.pdf")
savefig("PCA_SignalsSigma1_V2.png")

#------------------------------------------------------------------------------------------

# Graficos con PCA, señales y distribuciones de probabilidad

find_σ = 0.4
find_lcm = 1

find_σ2 = 0.1
find_lcm2 = 0.5

find_σ3 = 1
find_lcm3 = 4

# find in a DataFrame the two columns that matchs with the two values
find_row = findall(x -> x == find_σ, df_PCA_Signals.σs)
find_column = findall(x -> x == find_lcm, df_PCA_Signals.lcm)

find_row2 = findall(x -> x == find_σ2, df_PCA_Signals.σs)
find_column2 = findall(x -> x == find_lcm2, df_PCA_Signals.lcm)

find_row3 = findall(x -> x == find_σ3, df_PCA_Signals.σs)
find_column3 = findall(x -> x == find_lcm3, df_PCA_Signals.lcm)

# find the index of the row and column
find_index = intersect(find_row, find_column)[1]
find_index2 = intersect(find_row2, find_column2)[1]
find_index3 = intersect(find_row3, find_column3)[1]

x = df_PCA_Signals.pc1[find_index]
y = df_PCA_Signals.pc2[find_index]

u = df_PCA_Signals.pc1[find_index2]
v = df_PCA_Signals.pc2[find_index2]

w = df_PCA_Signals.pc1[find_index3]
z = df_PCA_Signals.pc2[find_index3]

plot_lcms_S = @df df_PCA_Signals StatsPlots.scatter(
    :pc1,
    :pc2,
    marker = (0.4,5),
    xaxis = (title = "PC 1"),
    yaxis = (title = "PC 2"),
    xlabel = "PC 1",
    ylabel = "PC 2",
    labels = false,  # Use the modified labels
)

ref_point = Plots.scatter!(plot_lcms_S,(x, y), label = L"l_{cm} = "*" $(df_PCA_Signals.lcm[find_index]) " * L"σ = " * " $(df_PCA_Signals.σs[find_index])" , color = "red", markersize=5, legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=8, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
ref_point2 = Plots.scatter!(plot_lcms_S,(u, v), label = L"l_{cm} = "*" $(df_PCA_Signals.lcm[find_index2]) " * L"σ = " * " $(df_PCA_Signals.σs[find_index2])" , color = "orange", markersize=5, legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=8, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
ref_point3 = Plots.scatter!(plot_lcms_S,(w, z), label = L"l_{cm} = "*" $(df_PCA_Signals.lcm[find_index3]) " * L"σ = " * " $(df_PCA_Signals.σs[find_index3])" , color = "purple", markersize=5, legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=8, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie1\\PCA_extra.pdf")

signal_plot = Plots.plot(t, dataSignals[:,find_index], label =  L"$l_{cm}$ = "*" $(df_PCA_Signals.lcm[find_index]) " * L"σ = " * " $(df_PCA_Signals.σs[find_index])", color = "red", xlabel = L"$t$ (s)", ylabel = L"S(t)", legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm);
signal_plot

savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie1\\Signal_extra.pdf")

signal_plot2 = Plots.plot(t, dataSignals[:,find_index2], label =  L"$l_{cm}$ = "*" $(df_PCA_Signals.lcm[find_index2]) " * L"σ = " * " $(df_PCA_Signals.σs[find_index2])", color = "orange", xlabel = L"$t$ (s)", ylabel = L"S(t)", legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm);
signal_plot2

savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie1\\Signal_extra2.pdf")

signal_plot3 = Plots.plot(t, dataSignals[:,find_index3], label =  L"$l_{cm}$ = "*" $(df_PCA_Signals.lcm[find_index3]) " * L"σ = " * " $(df_PCA_Signals.σs[find_index3])", color = "purple", xlabel = L"$t$ (s)", ylabel = L"S(t)", legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm);
signal_plot3

savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie1\\Signal_extra3.pdf")

Probdist_plot = Plots.plot(lc, dataProbd[:,find_index], label = L"$l_{cm}$ = "*" $(df_PCA_Probd.lcm[find_index]) " * L"σ = " * " $(df_PCA_Probd.σs[find_index])", color = "red", xlabel = L"$l_c$ $($$μ$m$)$", ylabel = L"P(l_c)", legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm);
Probdist_plot
xlims!(0, 20)
savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie1\\Probdist_extra.pdf")
plll = Plots.plot(plot_lcms_S, signal_plot, Probdist_plot, layout = (1,3))

xp = df_PCAProbd.pc1[find_index]
	yp = df_PCAProbd.pc2[find_index]

plot_lcms_P = @df df_PCAProbd StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para P(lc)",
)

ref_point_P = Plots.scatter!(plot_lcms_P,(xp, yp), label = L"l_{cm} = "*" $(df_PCAProbd.lcm[find_index]) " * L"σ = " * " $(df_PCAProbd.σs[find_index])" , color = "red", markersize=5)
Probdist_plot = Plots.plot(lc, dataOUT[:,find_index], label = false, color = "red", xlabel = "lc (μm)", ylabel = "P(lc)", title = L"P(lc) $l_{cm}$ = "*" $(df_PCAProbd.lcm[find_index]) " * L"σ = " * " $(df_PCAProbd.σs[find_index])");

plll2 = Plots.plot(plot_lcms_P, Probdist_plot, layout = (2,1), size = (800,1000))
