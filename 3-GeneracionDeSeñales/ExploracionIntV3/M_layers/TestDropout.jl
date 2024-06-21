using Flux
using DataFrames
using CSV
using DifferentialEquations
using SciMLSensitivity
using ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux
using Interpolations
using OrdinaryDiffEq
using IterTools: ncycle
using BSplineKit

function create_model(layers::Vector{Int}, activation::Function, dropout_rate::Float32)
    activations = [activation for i in 1:length(layers) - 2]
    startlayer = Dense(layers[1], layers[2])
    hiddenlayers = hcat([[Dense(layers[i], layers[i+1], activations[i]), Dropout(dropout_rate)] for i in 2:length(layers) - 2]...)
    return Chain(startlayer, hiddenlayers..., Dense(layers[end-1], layers[end]))
end

dropout_rate = Float32(0.1)
layers = [3, 32, 64, 32, 16, 1]
activation = relu

nn = create_model(layers, activation, dropout_rate)

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

for arch in architectures
    for opt in optimizers
        for batch_size in batchs_size
            for lambd in lambdas
                for eta in learning_rates
                    for dropout_rate in dropout_rates
                        push!(configuraciones, (arch, opt, batch_size, lambd, eta, dropout_rate))
                    end
                end
            end
        end
    end
end

length(configuraciones)