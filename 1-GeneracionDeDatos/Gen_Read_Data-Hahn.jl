# Programa de generacion de datos de señales de RMN Hahn

# Importaciones de paquetes
using CSV
using DataFrames
using Plots

##########################################################################################

# Constantes útiles para el cálculo de las seañes 

γ = 2.675e8  # Factor girómagetico del esín nuclear del proton (s⁻¹T⁻¹) de https://physics.nist.gov/cgi-bin/cuu/Value?gammap
G = 8.73e-7  # Gradiente externo (T/μm) de Validating NOGSE’s size distribution predictions in yeast cells Paper 1
D0 = 1e3 # Coeficiente de difusión (μm²/s) del Ejercico
# Las unidades utilzadas estan en μm, s y T

##########################################################################################
# Primero para generar las señales tenemos que tener una distribución de tamaños de compartimientos lc
# Generación de distribution log-normal, con parámetros lcm y σ, en este caso lcm es el tamaño medio del compartimiento y σ es la desviación estándar del tamaño del compartimiento
# Hagamos una función P(lc) de la distribucion log-normal

function P(lc,lcm,σ)
    return ( exp( -(log(lc) - log(lcm))^2 / (2σ^2) ) ) / (lc*σ*sqrt(2π))
end

##########################################################################################
# Si el tamaño de los compartimientos es único entonces la señal detectada es simplemente la magnetización de Hahn M(t)
# Función M_l Magnetización de Hahn, para un tiempo t y un tamaño lc
function Ml_Hahn(t, lc)
    τc = lc^2 / (2 * D0)
    term1 = -γ^2 * G^2 * D0 * τc^2
    term2 = t - τc * (3 + exp(-t / τc) - 4 * exp(-t / (2 * τc)))
    return exp(term1 * term2)
end

##########################################################################################
# La señal detectada S(t) es la suma de las señales de Hahn ponderadas por la distribución de tamaños de compartimientos P(lc)
# Función S_hanh(t) recibe el tamaño medio de compartimiento lcm, la desviación estándar σ, la cantidad de compartimientos N, 
# el tamaño mínimo de compartimiento l0, el tamaño máximo de compartimiento lf y el tiempo t

function S_han(lcm, σ, N, l0, lf, t)
    lc = range(l0, lf, length = N) # Generamos los tamaños de compartimientos lc desde l0 hasta lf

    P_lc = P.(lc,lcm, σ) # Consideramos media lcm y ancho σ
    
    M_lc = Ml_Hahn.(t, lc) # Calculamos M_lc(t) para cada tamaño de compartimiento
    
    S = sum(M_lc .* P_lc)
    return S
end

##########################################################################################
# Finalmente la función que genera los datos de las señales noramlizadas y las distribuciones de tamaños de compartimientos
# N: cantidad de compartimientos lc
# lcm: tamaño medio de compartimiento
# σ: desviación estándar de compartimiento
# l0: tamaño mínimo de compartimiento
# lf: tamaño máximo de compartimiento
# time_sim: tiempo máximo de simulación
# time_sample_lenght: cantidad de puntos de tiempo

function GenData(N, lcm, σ, l0, lf, t)
    # Generamos los tamaños de compartimientos
    lc = range(l0, lf, length = N)

    # Generamos las distribuciones
    P_l = P.(lc, lcm, σ)

    # Calculamos la señal
    S0 = S_han(lcm, σ, N, l0, lf, 0)
    S = S_han.(lcm, σ, N, l0, lf, t) ./ S0
    return S, P_l
end

##########################################################################################
# Ahora generamos los datos en CSV para cada combinación de parámetros en el path especificado, este va a ser el mismo que use para leer los datos
# Generación de datos en CSV
# N numero de compartimientos
# time_sample_lenght cantidad de puntos de tiempo
# l0 tamaño mínimo de compartimiento
# lf tamaño máximo de compartimiento
# tf tiempo máximo de simulación
# lcms vector de tamaños medios de compartimientos
# σs vector de desviaciones estándar

function GenCSVData(N, time_sample_lenght, l0, lf, t, lcms, σs, path)

    function fill_missing(value, column, max_lenght)
        if length(column) < max_lenght
            return vcat(column,fill(value, max_lenght - length(column)))
        else
            return column
        end
    end

    l = range(l0, lf, length = N)
    max_lenght = maximum(length.([l,t]))

    for lcm in lcms
        for σ in σs
            S, P_l = GenData(N, lcm, σ, l0, lf, t)
            S = fill_missing(0, S, max_lenght)
            df = DataFrame(S = S, P_l = P_l)
            CSV.write(path * "/$(lcm)_$(σ)l_$N.csv", df)
        end
    end
end

# Estos datos en CSV contieen 2 columnas la primera es la señal y la segunda la distribución de tamaños de compartimientos
# Como los datos pueden tener un muestreo diferente se rellenan con ceros para que tengan la misma longitud
# Por esto es importante tener en cuenta el muestreo del tiempo y el de los tamaños de compartimientos

##########################################################################################

# Lectura de los datos que se generaron
# mismos parámetros que GenCSVData
# Esta función acumula todos los archivos generados de las señales y las distribuciones de tamaños de compartimientos especificando cada lcm y σ

function ReadCSVData(N, time_sample_lenght, l0, lf, t, lcms, σs, path)
    lc = range(l0, lf, length = N)
    length_t = length(t)
    length_lc = length(lc)
    max_lenght = maximum(length.([t, lc]))

    Probabilitys = zeros(length(lcms), length(σs), max_lenght)
    Signals = zeros(length(lcms), length(σs), max_lenght)

    for lcm in lcms
        for σ in σs
            df = CSV.read(path * "/$(lcm)_$(σ)l_$N.csv", DataFrame)
            Probabilitys[findall(x -> x == lcm, lcms), findall(x -> x == σ, σs), :] = df.P_l
            Signals[findall(x -> x == lcm, lcms), findall(x -> x == σ, σs), :] = df.S

        end
    end

    return Probabilitys, Signals

end

# Fin del programa