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

# Arquitecturas que vamos a utilizar
architectures = [
    [[3, 32, 32, 16, 1], relu], # Tres capas ocultas simple
    [[3, 128, 64, 16, 1], relu], # Tres capas ocultas con mas neuronas
    [[3, 16, 32, 64, 16, 1], swish], # Cuatro capas ocultas simple
    [[3, 32, 64, 32, 16, 1], swish], # Cuatro capas ocultas con mas neuronas
    ]

# Optimizadores que vamos a utilizar
optimizers = [opt for opt in [AdamW]]

# Numero de mini-batchs que vamos a utilizar 
batchs_size = [10, 20, 30]

# Dropout rates
dropout_rates = [0.0, 0.2, 0.5]

# Lambdas regularizacion L2
lambdas = [0.0, 0.01, 0.1]

# Learning rates
learning_rates = [0.001, 0.1]

# Vector de configuraciones que vamos a utilizar
configuraciones = []

for arch in architectures, opt in optimizers, batch_size in batchs_size, lambd in lambdas, eta in learning_rates, dropout_rate in dropout_rates
    push!(configuraciones, (arch, opt, batch_size, lambd, eta, dropout_rate))
end

df_resultados = CSV.read("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Resultados/Resultados_ML.csv", DataFrame)

IDs = df_resultados.ID

dp_rates = [dp[end] for dp in configuraciones[IDs]]

df_resultados[!,"Dropout_Rate"] = dp_rates

df_resultados = sort(df_resultados, :ID)

df_resultados

CSV.write("C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionIntV3/M_layers/Resultados/Resultados_ML.csv", df_resultados)