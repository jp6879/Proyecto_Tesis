### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ e0d43e36-9410-4d5e-89d5-7ca289b27102
using PlutoUI

# ╔═╡ 054b047d-6eb9-473c-9bba-21adc18c45b9
md"# NOdes para realizar una predicción continua de señales de Hahn

## Predicción para una única señal

Comencemos por lo básico, primero que todo vamos a crear un modelo para predecír una única señal y ver como se comporta. 

Comenzamos intentando utilizar para entrenar una señal con solo 60 puntos, de los cuales vamos a suponer que 10 son puntos medidos para predecir la señal

Visualizemos primero los dos conjuntos que vamos a utilizar, como también las derivadas que se pasan como extra al modelo. En partícular vamos a utilizar la señal con $l_{cm} = 0.5$ $\mu$m y $\sigma = 1$.
"

# ╔═╡ ab94bb01-d059-4d31-9dd0-eef92adb72bd
PlutoUI.Resource("https://imgur.com/XSHXjdP.png")

# ╔═╡ a778665d-503d-4e3c-97cd-e7a81014d3f9
PlutoUI.Resource("https://imgur.com/J8FPbf0.png")

# ╔═╡ e5491fa7-0243-4101-9939-d13511e211bd
md"Veamos ahora las derivadas, porque esto influye bastante en la señal predicha por la red neuronal"

# ╔═╡ 5536bcd2-aa9c-4cd8-8123-94994edfea59
PlutoUI.Resource("https://imgur.com/IVJ2pqt.png")

# ╔═╡ 24c41cc8-3db4-47c9-ac9b-6af8844d88c1
md"Vamos a ver de que esto es probablemente lo que causa problemas en las predicciones"

# ╔═╡ 11f9189d-89ed-40e4-85a6-44191a1158d8
# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 45

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lcs = range(l0, lf, length = N)

t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))
    
# Concatenamos t_short and t_long
t = vcat(t_short, t_long)
toriginal = vcat(t_short, t_long)

# Leeemos los datos
# path_read = "/home/juan.morales/datos_PCA"
path_read = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/1-GeneracionDeDatos/Datos_Final/datos_PCA"

# Parametros que se varian
# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
sigmas = 0.01:0.01:1

######################################################################################

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

######################################################################################
    
# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto =  25 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 5

# Esto da 60 tiempos, distribuidosen 40 puntos desde 0 a 0.1 y 20 puntos desde 0.1 a 1
t_short = t_short[1:muestreo_corto:end]
t_long = t_long[1:muestreo_largo:end]

t = vcat(t_short, t_long)

# Tomamos 1 sigmas y 1 tamaño de compartimientos para cada sigma o sea 60 señales
sampled_sigmas =  [1.0]
lcm_range = 1:250:250

println("Sigmas: ", sampled_sigmas)
println("Lcms: ", collect(lcms)[lcm_range])

# Obtenemos las señales representativas para un conjunto de sigmas y lcms
Signals_rep, Signals_rep_derivadas, column_lcm_rep, column_sigmas_rep = Get_Signals_Data_Training(path_read, lcms, sigmas, sampled_sigmas, lcm_range, muestreo_corto, muestreo_largo, t)

# Numero de puntos para la validacion
n_valid = 11 # Terminan siendo 10

# Paso para tomar los tiempos de entrenamiento y validación
step = floor(Int, length(t) / n_valid) + 1

tvalid = t[1:step:end]
ttrain = [t for t in t if t ∉ tvalid]
# Concatenate in the begining
ttrain = vcat(0, ttrain)

indexes_train = [i for i in 1:length(t) if t[i] in ttrain]
indexes_valid = [i for i in 1:length(t) if t[i] in tvalid]

Signals_valid = Signals_rep[:,indexes_valid]
Signals_train = Signals_rep[:,indexes_train]

Signals_derivadas_train = Signals_rep_derivadas[indexes_train,:]
Signals_derivadas_valid = zeros(size(Signals_valid))

for i in 1:size(Signals_valid)[1]
    Signals_derivadas_valid[i,:] = derivate_signals(tvalid,Signals_valid[i,:])
end

Signals_derivadas_valid = Float32.(Matrix(Signals_derivadas_valid'))

# Normalizamos las derivadas
for i in 1:size(Signals_valid)[1]
    Signals_derivadas_valid[:,i] = Signals_derivadas_valid[:,i] ./ maximum(abs.(Signals_derivadas_valid[:,i]))
end

extra_parameters = Signals_derivadas_train
extra_parameters_valid = Signals_derivadas_vali

# ╔═╡ a6c7b7ed-467d-4374-b0bc-5a0c8f5b3256
md"Vamos a usar una red que funcionó anteriormente para pocas señales, la cual tiene 3 capas ocultas con 32, 64 y 16 neuronas con una función de activación tanh.

Además entrenamos este modelo con un batchsize de 5 puntos, un learning rate de 0.001 durante 2000 épocas con el método AdamW, es decir Adam con weight decay.

Seguimos utilizando la función de penalización para garantizar una señal decreciente con un $\lambda = 0.1$ en la función de Loss
"

# ╔═╡ bae54963-a828-45e7-b4f7-31e89d000442
# Las condiciones iniciales son todas 1 para cada señal
U0 = ones32(size(Signals_rep)[1])

#Definimos el batch size
batch_size = 5

# Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
train_loader = Flux.Data.DataLoader((Signals_train, ttrain), batchsize = batch_size)

# Función de activación
activation = tanh_fast

nn = Chain(Dense(2, 32, activation),
            Dense(32, 64, activation),
            Dense(64, 16, activation),
            Dense(16, 1)
            )

# Tomamos un learning rate de 0.001
η = 1e-3

# Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
epochs = 2000

# Parametrizar indices para poder acceder a los parametros extras de la red neuronal
f(x,p) = round(Int, x * (length(p) - 1)) + 1

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

##############################################################################################

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

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

    return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-5, abstol = 1e-5)) # Regresa la solución de la ODE
end

# Función que predice las señales para un conjunto de condiciones iniciales y parámetros extra
function Predict_Singals(U0, parametros_extra, time_batch)
    Predicted_Signals = zeros(size(time_batch))
    for i in 1:length(U0)
        u0 = Float32[U0[i]]
        predicted_signal = predict_NeuralODE(u0, parametros_extra[:, i], time_batch)[1, :]
        Predicted_Signals = hcat(Predicted_Signals, predicted_signal)
    end    
    Predicted_Signals[:,2:end]
end

# Función que calcula el loss de la red neuronal para un batch de datos y sus respectivos tiempos

# Término de pensalización

function penalization_term(time_batch,y)
    """
    Función de penalización para tratar de mantener la señal monotonamente decrecente
    En caso que la señal no sea monotonamente decreciente, la penalización es la suma de las diferencias positivas entre los tiempos
    """
	pen = sum(sum.(max.(y[2:end,:] .- y[1:end-1,:], 0)))
end


function loss_node(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0, extra_parameters, time_batch)
    return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y))
end


# Hay dos funciones de loss porque le tengo que distinguir las derivadas del entrenamiento con las de validación.

function loss_valid(batch, time_batch, lamb = 0.1)
    y = Predict_Singals(U0, extra_parameters_valid, time_batch)
    return Flux.mse(y, batch') + lamb * (penalization_term(time_batch, y))
end

# Función de callback para guardar el loss en cada época
global iter = 0
loss = []
loss_valid_array = []
callback = function ()
    global iter += 1
    if iter % (length(train_loader)) == 0
        epoch = Int(iter / length(train_loader))
        actual_loss = loss_node(Signals_train, ttrain)
        println("Epoch = $epoch || Loss: $actual_loss")
    end
    return false
end

###############################################################################################################

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p), ncycle(train_loader, epochs), opt, cb = callback)

# ╔═╡ e5430075-31d9-45a4-afd1-21874efd13b8
md"Si vemos la función de loss del entrenamiento nos damos cuenta de que ya llegamos a un mínimo"

# ╔═╡ 207da352-0a66-4aae-8f85-342d1eebb5e8
PlutoUI.Resource("https://imgur.com/wvXdB9H.png")

# ╔═╡ 53330f4b-efbd-4a03-89bd-84eeeb873fdf
md"Si vemos la predicción con los puntos de entrenamiento, es decir pasandole a la red los púntos de entreanmiento con los de las derivadas de estos obtenemos una señal predicha razonable."

# ╔═╡ cb21a843-dc19-4555-9647-7bd0acf7cf59
PlutoUI.Resource("https://imgur.com/iCRUhU8.png")

# ╔═╡ 8d5251c0-ee65-4b85-8b88-e1c6f720a10a
md"Veamos ahora como se ve la señal predicha con los diez puntos de validación, y con la derivada calculada con estos puntos. Esto es todo lo que necesita la red para encontrar la dinámica. Además le pasamos los tiempos en los que queremos que nos dé los puntos predichos, vamos a pedirle los 60 tiempos originales."

# ╔═╡ e346b709-40b8-449a-8038-c07d87f3d396
PlutoUI.Resource("https://imgur.com/7phyotX.png")

# ╔═╡ d4e5a41c-a3b3-44d8-99d5-15fd39fb7200
md"Vemos que la predicción no es buena, tengamos en cuenta que solo neceistamos 10 puntos para obtener toda esta curva que serían los datos 'medidos'. Veamos que pasa si le damos mas puntos y la derivada de estos."

# ╔═╡ 58ef90bf-891e-410b-b2d0-5426c322eb27
PlutoUI.Resource("https://imgur.com/Dd9Lp1D.png")

# ╔═╡ 8b371f35-2033-4f54-8d3d-68e7a7f0ba8c
PlutoUI.Resource("https://imgur.com/Bjpxd17.png")

# ╔═╡ d4c20590-f0ff-4fbb-bf7d-7634090117cb
md"Ahora con 20 puntos 'medidos' la derivada es esta la cual todavía le falta información del pico pero se parece un poco mas a la derviada con la que se entrenó el modelo."

# ╔═╡ aa32ff80-d549-4b0b-b4b5-3729f11f81a6
PlutoUI.Resource("https://imgur.com/wacHpXL.png")

# ╔═╡ 26566557-e5e1-4e99-a9e1-edb9f8166cf7
md"La señal se pega un poco mas a la señal original y se parece mas a la predicha.

Podrían ser necesarios más paráemtros extras para distinguir las señales, uno interesante sería utilizar una recta que una los puntos inicial y final, esta es fácil de calcular para mediciones y coincide perfectamente con parámetros extra del entrenamiento. Probablemente esto haya sido uno de los problemas que estuvo teniendo la red al entrenar los modelos.

Algo simple que probé es agregar una recta con el valor inicial y final de los datos, en general esto serviría para casos donde la señal no llega a decaer por completo como para $\sigma = 1$ a estos tiempos.
"

# ╔═╡ a6099ec1-7be4-48b5-8a75-f8bb559ba234
PlutoUI.Resource("https://imgur.com/FYpPc5n.png")

# ╔═╡ b76b706b-5996-4766-b9da-344a667aea07
md"Como la recta necesita solo dos puntos podemos usar los mismos para el entrenamiento y la predicción. Hacer esto debería dar a la red mas información. Veamos los resultados que obtuvimos para la misma señal con un entrenamiento mucho mas corto."

# ╔═╡ 58ec0c8f-3dcf-49c2-972a-ee98a3e1bb6a
PlutoUI.Resource("https://imgur.com/7TzWZhE.png")

# ╔═╡ 60af69bb-7ec1-4afe-83cc-d7cfcd166f08
PlutoUI.Resource("https://imgur.com/XdeRb69.png")

# ╔═╡ f4fb8923-92f5-44ef-ba8e-b25af2974b34
md"Mucho mejor que antes la predicción que realiza, igualmente esto no es muy impresionante la idea sería usarlo para mas señales. En este caso usamos solo 10 puntos de la señal para predecir esta señal. Los cambios en el código fueron que ahora usamos 3 valores de entrada a la señal y el cálculo de la recta, que además depende únicamente de un solo punto porque sabemos siempre que empiezan en (0,1)."

# ╔═╡ 71f87e48-c3f8-479d-9db5-3a7b9c4c1928
function calucla_recta(x, x1, y1, x0 = 0, y0 = 1)
    m = (y1 - y0) / (x1 - x0)
    return Float32.(m .* x' .+ b)
end

# Función de activación
activation = tanh_fast

nn = Chain(Dense(3, 32, activation),
            Dense(32, 16, activation),
            Dense(16, 8, activation),
            Dense(8, 1)
            )

# Tomamos un learning rate de 0.001
η = 5e-3

# Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
epochs = 800

# Parametrizar indices para poder acceder a los parametros extras de la red neuronal
f(x,p) = round(Int, x * (length(p) - 1)) + 1

p, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

##############################################################################################

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

# Función que resuelve la ODE con los parametros extra y las condiciones iniciales que instanciemos y nos regresa la solución en un arreglo
function predict_NeuralODE(u0, parametros, parametros2, time_batch)
    # dSdt = NN(S, parametros_extra) 
    function dSdt(u, p, t; parametros_extra = parametros, parametros_extra2 = parametros2)
        indx = f(t, parametros)
        parametros_actuales = parametros[indx] # Selecciona los parametros extra en el tiempo t
        parametros_actuales_2 = parametros2[indx]
        entrada_red = vcat(u, parametros_actuales, parametros_actuales_2) # Concatena los el valor de S(t) con los parametros extra en el tiempo t
        return re(p)(entrada_red) # Regresa la salida de la red neuronal re creada con los parámetros p
    end

    prob = ODEProblem(dSdt, u0, tspan)

    return Array(solve(prob, Tsit5(), dtmin=1e-9 , u0 = u0, p = p, saveat = time_batch, reltol = 1e-5, abstol = 1e-5)) # Regresa la solución de la ODE
end

# Función que predice las señales para un conjunto de condiciones iniciales y parámetros extra
function Predict_Singals(U0, parametros_extra, parametros_extra2, time_batch)
    Predicted_Signals = zeros(size(time_batch))
    for i in 1:length(U0)
        u0 = Float32[U0[i]]
        predicted_signal = predict_NeuralODE(u0, parametros_extra[:, i], parametros_extra2[:, i], time_batch)[1, :]
        Predicted_Signals = hcat(Predicted_Signals, predicted_signal)
    end    
    Predicted_Signals[:,2:end]
end

# ╔═╡ 2b27f6d5-b80d-40aa-abf6-a1d6462887ed


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "6e7bcec4be6e95d1f85627422d78f10c0391f199"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═e0d43e36-9410-4d5e-89d5-7ca289b27102
# ╟─054b047d-6eb9-473c-9bba-21adc18c45b9
# ╟─ab94bb01-d059-4d31-9dd0-eef92adb72bd
# ╟─a778665d-503d-4e3c-97cd-e7a81014d3f9
# ╟─e5491fa7-0243-4101-9939-d13511e211bd
# ╟─5536bcd2-aa9c-4cd8-8123-94994edfea59
# ╟─24c41cc8-3db4-47c9-ac9b-6af8844d88c1
# ╠═11f9189d-89ed-40e4-85a6-44191a1158d8
# ╟─a6c7b7ed-467d-4374-b0bc-5a0c8f5b3256
# ╠═bae54963-a828-45e7-b4f7-31e89d000442
# ╟─e5430075-31d9-45a4-afd1-21874efd13b8
# ╟─207da352-0a66-4aae-8f85-342d1eebb5e8
# ╟─53330f4b-efbd-4a03-89bd-84eeeb873fdf
# ╟─cb21a843-dc19-4555-9647-7bd0acf7cf59
# ╟─8d5251c0-ee65-4b85-8b88-e1c6f720a10a
# ╟─e346b709-40b8-449a-8038-c07d87f3d396
# ╟─d4e5a41c-a3b3-44d8-99d5-15fd39fb7200
# ╟─58ef90bf-891e-410b-b2d0-5426c322eb27
# ╟─8b371f35-2033-4f54-8d3d-68e7a7f0ba8c
# ╟─d4c20590-f0ff-4fbb-bf7d-7634090117cb
# ╟─aa32ff80-d549-4b0b-b4b5-3729f11f81a6
# ╟─26566557-e5e1-4e99-a9e1-edb9f8166cf7
# ╟─a6099ec1-7be4-48b5-8a75-f8bb559ba234
# ╟─b76b706b-5996-4766-b9da-344a667aea07
# ╟─58ec0c8f-3dcf-49c2-972a-ee98a3e1bb6a
# ╟─60af69bb-7ec1-4afe-83cc-d7cfcd166f08
# ╟─f4fb8923-92f5-44ef-ba8e-b25af2974b34
# ╠═71f87e48-c3f8-479d-9db5-3a7b9c4c1928
# ╠═2b27f6d5-b80d-40aa-abf6-a1d6462887ed
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
