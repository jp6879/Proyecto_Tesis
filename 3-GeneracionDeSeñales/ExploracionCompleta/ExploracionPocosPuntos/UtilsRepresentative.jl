# Funcion para leer las señales desde el path_read
function GetSignals(path_read)
    dataSignals = CSV.read(path_read * "/dataSignals.csv", DataFrame)
    dataSignals = Matrix(dataSignals)
    return dataSignals
end

# Funcion que toma las señales representativas para un conjunto de sigmas y lcms además las muestrea según el muestreo corto y largo
function Get_Rep_Signals(indexes, sampled_sigmas, lcm_range, path_read, muestreo_corto, muestreo_largo)
    dataSignals = Float32.(GetSignals(path_read))
    Signals_rep = Float32.(Matrix(dataSignals[:,indexes]'))
    Signals_rep_short = Signals_rep[:,1:muestreo_corto:1000]
    Signals_rep_long = Signals_rep[:,1001:muestreo_largo:end]
    Signals_rep = hcat(Signals_rep_short, Signals_rep_long)
    return Signals_rep
end

###################################################################################
# Función que idientifca las señales que se van a utilizar para el entrenamiento de la NODE con lcm y σ
function Get_Signals_Test_Parameters(lcms,sigmas)
    dim1 = dimlcm = length(lcms)
    dim2 = dimsigma = length(sigmas)

    column_lcm = zeros(dim1*dim2)
    column_sigmas = zeros(dim1*dim2)
    aux_lcm = collect(lcms)
    aux_sigmas = collect(sigmas)

    for i in 1:dim1
        for j in 1:dim2
            column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
            column_sigmas[(i - 1)*dim2 + j] = aux_sigmas[j]
        end
    end
    
    return column_lcm, column_sigmas
end

###################################################################################

# Función que devuelve señales de prueba, sus derivadas y los parámetros con los que se generaron
function Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)
    # Obtenemos primero los parámetros con los que se generaron las señales
    column_lcm, column_sigmas = Get_Signals_Test_Parameters(lcms,sigmas)
    df_SignalsParams = DataFrame(
        sigmas = column_sigmas,
	    lcm = column_lcm,
	)
    indexes = []
    # A partir de ciertos sigmas dados por sampled_sigmas y un rango de lcm dado por lcm_range, obtenemos las señales representativas
    for sigma in sampled_sigmas
        find_rows = findall(x -> x == sigma, df_SignalsParams.sigmas)[lcm_range]
        push!(indexes, find_rows)
    end
    # Indices de estas señales
    indexes = vcat(indexes...)
    # Obtenemos las señales representativas ya muestreadas
    Signals_rep = Get_Rep_Signals(indexes, sampled_sigmas, lcm_range, path_read, muestreo_corto, muestreo_largo)
    # Derivadas de las señales
    Signals_rep_derivadas = zeros(size(Signals_rep))
    for i in 1:size(Signals_rep)[1]
        Signals_rep_derivadas[i,:] = derivate_signals(t,Signals_rep[i,:])
    end
    Signals_rep_derivadas = Float32.(Matrix(Signals_rep_derivadas'))
    # Normalizamos las derivadas
    for i in 1:size(Signals_rep)[1]
        Signals_rep_derivadas[:,i] = Signals_rep_derivadas[:,i] ./ maximum(abs.(Signals_rep_derivadas[:,i]))
    end
    column_lcm_rep = column_lcm[indexes]
    column_sigmas_rep = column_sigmas[indexes]
    return Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep
end