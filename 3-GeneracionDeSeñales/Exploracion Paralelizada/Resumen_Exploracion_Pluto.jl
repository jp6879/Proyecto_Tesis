### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 4825797f-1135-42b6-b0ea-e8bb023fdbe9
using PlutoUI

# ╔═╡ 9ecc5f2e-f3c4-4575-ade5-90203c8269b8
md"Para continuar con la exploración de hiperparámetros de las NODEs con el objetivo de predecir señales de Hhan a partir de datos, se utilizó el clúster para llevar a cabo este proceso de manera paralelizada.

En este caso, se seleccionó un conjunto de 100 señales que presentaban un nivel de ruido $\sigma = 1$ y una longitud de onda central $l_{cm}$ que variaba entre 0.5 y 5.50 $\mu m$. Estas señales están representadas en el gráfico de PCA como se muestra a continuación:
"

# ╔═╡ 808c7b06-ac83-4713-8a26-d94fd0bdaf90
PlutoUI.Resource("https://imgur.com/VV25cib.png")

# ╔═╡ 226682b6-f0b7-4b83-a6f6-a78e089d9096
md"Sin embargo, a diferencia de lo que ocurre con un conjunto reducido de señales, estas redes encuentran más desafíos al aprender el conjunto de 100 señales disponible.

Se entrenaron 22 modelos de NODEs utilizando diversas arquitecturas, funciones de activación y tamaños de batch. Todos estos modelos fueron entrenados durante 1500 épocas, utilizando una función de loss MSE, y un ritmo de aprendizaje de $\eta = 5 \times 10^{-3}$.

Las señales utilizadas para el entrenamiento y la predicción se muestran en la siguiente figura. Los puntos en azul, cian (inicio) y amarillo (final) representan los datos utilizados para el entrenamiento, mientras que los puntos en rojo representan los datos que se desean predecir con la NODE.

Se utilizaron 22 puntos de datos para el entrenamiento, mientras que se buscan predecir los 23 restantes. Puesto que en este caso el muestreo es de 45 puntos
"

# ╔═╡ 9f6f182a-74a4-4c28-a4be-471c39025391
PlutoUI.Resource("https://imgur.com/lFYNvY1.png")

# ╔═╡ bd9f329f-8d0f-40ca-85b4-0988c379114f
md"
$$\begin{aligned}
& \begin{array}{cccc}
\hline \hline
\text{ID} & \text{Arq.} & \text{Act.} & \text{Opt.} & \text{BS} & \text{LossTrain} & \text{LossPredict} \\
\hline
1 & [2, 8, 1] & relu & AdamW & 5 & 0.02770 & 0.0140 \\
2 & [2, 8, 1] & relu & AdamW & 10 & 0.02823 & 0.01380 \\
3 & [2, 8, 1] & relu & RMSProp & 5 & 0.02769 & 0.01391 \\
4 & [2, 8, 1] & relu & RMSProp & 10 & 0.02816 & 0.01375 \\
5 & [2, 8, 1] & tanh & AdamW & 5 & 0.02796 & 0.01387 \\
6 & [2, 8, 1] & tanh & AdamW & 10 & 0.03266 & 0.01374 \\
7 & [2, 8, 1] & tanh & RMSProp & 5 & 0.02778 & 0.01375 \\
8 & [2, 8, 1] & tanh & RMSProp & 10 & 0.02782 & 0.01343 \\
9 & [2, 8, 1] & swish & AdamW & 5 & 0.02776 & 0.01394 \\
10 & [2, 8, 1] & swish & AdamW & 10 & 0.02845 & 0.01360 \\
11 & [2, 8, 1] & swish & RMSProp & 5 & 0.02778 & 0.01395 \\
12 & [2, 8, 1] & swish & RMSProp & 10 & 0.02788 & 0.01360 \\
14 & [2, 16, 16, 1] & relu & AdamW & 10 & 0.02768 & 0.01360 \\
15 & [2, 16, 16, 1] & relu & RMSProp & 5 & 0.02685 & 13.84 \\
16 & [2, 16, 16, 1] & relu & RMSProp & 10 & 0.02772 & 0.01337 \\
17 & [2, 16, 16, 1] & tanh & AdamW & 5 & 0.02760 & 0.01391 \\
22 & [2, 16, 16, 1] & swish & AdamW & 10 & 0.02784 & 0.01393 \\
50 & [2, 32, 64, 64, 32, 16, 1] & relu & AdamW & 10 & 0.02514 & 0.02405 \\
28 & [2, 32, 64, 16, 1] & relu & RMSProp & 10 & 0.02122 & 1153.10 \\
30 & [2, 32, 64, 16, 1] & tanh & AdamW & 10 & 0.02331 & 0.00937 \\
34 & [2, 32, 64, 16, 1] & swish & AdamW & 10 & 0.02757 & 0.01372 \\
52 & [2, 32, 64, 64, 32, 16, 1] & relu & RMSProp & 10 & 0.01884 & 0.02785 \\
\hline
\end{array}
\end{aligned}$$

"

# ╔═╡ e1134168-437c-43b3-9afe-facccd580873
md"La arquitectura que muestra un mejor desempeño es la identificada con el ID 30. Esta arquitectura consta de 3 capas ocultas con 32, 64 y 16 neuronas respectivamente, utilizando una función de activación tanh. Además, fue entrenada con mini lotes de 10 puntos.

A pesar de que esta arquitectura ha mostrado un rendimiento prometedor en términos de su desempeño durante el entrenamiento, los resultados de las predicciones no cumplen con lo esperado. Es evidente que hay discrepancias entre las predicciones generadas por el modelo y los valores reales de los datos."

# ╔═╡ f7ad76dd-dcd5-47c1-b04c-df33a7f6c119
PlutoUI.Resource("https://imgur.com/tCFc84F.png")

# ╔═╡ ef3b1564-fa92-4ced-ba70-436c87a4e33d
PlutoUI.Resource("https://imgur.com/jQYjJla.png")

# ╔═╡ aca33433-be0b-42f3-8132-7f048989a06e
PlutoUI.Resource("https://imgur.com/feLdpQc.png")

# ╔═╡ 94fb618c-794b-483c-a7ac-adfc888fb77a
PlutoUI.Resource("https://imgur.com/5eV1ZRb.png")

# ╔═╡ 7e9be462-709a-4601-9b55-b12fe0517c28
md"Las predicciones de las señales tienden a acumularse en una zona donde en los datos originales hay una gran densidad de señales puesto a que para $l_{cm}$ mas grandes vimos que se parecen mas, y la NODE está aprendiendo a llevarlas hacia allí.

Si observamos el loss en función de las épocas vemos que el aprendizaje de entrenamiento mejora pero el de predicción tiene comportamientos de subida algunas, quizas por una mala elección del learning rate.
"

# ╔═╡ 9f053d4f-fbfc-4663-8d54-3c38c34150c4
PlutoUI.Resource("https://imgur.com/FNsAnkp.png")

# ╔═╡ 62640253-a1ac-442e-b78e-446a15c335ed
md"Para intentar verificar si el problema proviene de la elección de los datos de entrenamiento, esta vez utilizamos 50 señales, nuevmente con $\sigma = 1$, pero esta vez con un $l_{cm}$ entre 1 y 1.5 $\mu m$. En el grafico de PCA se corresponden con las siguientes señales
"

# ╔═╡ 01ca37d6-f276-4d43-aa25-5f45b301ca17
PlutoUI.Resource("https://imgur.com/GZFopXV.png")

# ╔═╡ 5df2f655-4d45-4bfa-89cf-b56642859b85
md"Las señales se muestran en la siguiente figura donde esta vez tomamos una mayor cantidad de puntos de las señales, utilizamos 110 puntos en total tomamos 24 para el entrenamiento y el resto los dejamos como predicción."

# ╔═╡ 2233898d-28a5-452c-9c26-d0f7ca33c0b2
PlutoUI.Resource("https://imgur.com/rFszspg.png")

# ╔═╡ 92a5a294-d80a-43ce-a71b-5912bc971fbc
md"Se entrenaron 15 arquitecturas comenzando desde aquellas con 3 capas ocultas durante 1500 épocas. Uitlizando una función loss MSE con un learning rate menor al caso anterior de $\eta = 1 \times 10^{-3}$. Esta vez los resultados del loss MSE son mejores que el anterior caso con el entrenamiento para 100 señales."

# ╔═╡ a7266221-27be-4775-833d-51bcd0c5cf07
md"
$$\begin{aligned}
& \begin{array}{ccccccc}
\hline \hline
\text{ID} & \text{Arq.} & \text{Act.} & \text{Opt.} & \text{Batch} & \text{LossTrain} &\text{LossPred.} \\
\hline
1 & [2, 32, 64, 16, 1] & relu & AdamW & 5 & 0.00635 & 0.04347 \\
2 & [2, 32, 64, 16, 1] & relu & AdamW & 15 & 0.00767 & 0.02064 \\
3 & [2, 32, 64, 16, 1] & tanh & AdamW & 5 & 0.00201 & 0.00664 \\
4 & [2, 32, 64, 16, 1] & tanh & AdamW & 15 & 0.01204 & 0.01433 \\
5 & [2, 32, 64, 16, 1] & swish & AdamW & 5 & 0.00877 & 29546.01 \\
6 & [2, 32, 64, 16, 1] & swish & AdamW & 15 & 0.00863 & 0.21363 \\
7 & [2, 50, 64, 16, 1] & relu & AdamW & 5 & 0.00760 & 0.01538 \\
8 & [2, 50, 64, 16, 1] & relu & AdamW & 15 & 0.00820 & 0.02016 \\
10 & [2, 50, 64, 16, 1] & tanh & AdamW & 15 & 0.01119 & 0.01270 \\
11 & [2, 50, 64, 16, 1] & swish & AdamW & 5 & 0.00823 & 0.05710 \\
12 & [2, 50, 64, 16, 1] & swish & AdamW & 15 & 0.00925 & 0.00759 \\
13 & [2, 64, 128, 64, 32, 16, 1] & relu & AdamW & 5 & 0.00097 & 0.00596 \\
14 & [2, 64, 128, 64, 32, 16, 1] & relu & AdamW & 15 & 0.00422 & 0.01394 \\
17 & [2, 64, 128, 64, 32, 16, 1] & swish & AdamW & 5 & 0.00785 & 0.01675 \\
18 & [2, 64, 128, 64, 32, 16, 1] & swish & AdamW & 15 & 0.00796 & 0.01866 \\
\hline
\end{array}
\end{aligned}$$
"

# ╔═╡ dc7017ad-a2bd-437f-8188-5bc7f5abca5e
md"Si observamos la tabla la red que obtuvo una mejor performance es la red correspondiente al ID 13 la cual contiene 5 capas ocultas con 64, 128, 64, 32 y 16 capas ocultas con función de activación ReLU, y entrenadas con mini-batchs de 5 puntos.

Si observamos los resultados de esta arquitectura podemos ver que las predicciones tampoco son las mejores, pero por ejemplo la señal predecida con $\sigma = 1$ y $l_{cm} = 1.48$ $\mu m$ es similar a la señal simulada."

# ╔═╡ 51a802a4-6d2c-4f5e-9be7-89896060534e
PlutoUI.Resource("https://imgur.com/zNhrMf7.png")

# ╔═╡ a51a93ac-855c-4f6c-9f72-5bb8250b6cfc
PlutoUI.Resource("https://imgur.com/sjFbmV6.png")

# ╔═╡ 52adf242-1cc9-4434-82fe-eb27cb8bde26
PlutoUI.Resource("https://imgur.com/DuDzeqr.png")

# ╔═╡ d4c0199c-eb23-443f-a98c-f3f492aa5d11
PlutoUI.Resource("https://imgur.com/FYKir2n.png")

# ╔═╡ 7b45fce7-3e52-4ae9-a03f-d20e38888a25
md"Cuando observamos el loss a lo largo de las épocas se observan pequeños saltos en el Loss calculado para las predicciones pero mucho menos debido a que disminuimos el valor del learning rate."

# ╔═╡ c34dd2e8-3841-4845-94f8-e6731841501d
PlutoUI.Resource("https://imgur.com/LRcfrI0.png")

# ╔═╡ 5a267d71-635d-46cf-8135-61d9cbf920a6
md"Algunas mejoras realizadas a este modelo que se pueden realizar es añadir mas puntos al aprendizaje y agregar a la función de costo una penalización para cuando la señal predecida no sea monotonamente decreciente.

Esta segunda mejora fue implementada añadiendo a la función costo el término

$$\begin{equation}

\lambda \sum_{j=1}^{\mu} \sum_{i=1}^{N-1} max\left(S^{(j)}(t_{i+1}) - S^{(j)}(t_i),\: 0\right),

\end{equation}$$

donde $\lambda$ es otro hiperparámetro del modelo el cual se tomó como $\lambda = 1$, $\mu$ es la cantidad de señales tomadas y $N$ es el número de puntos que se predicen de cada señal. Con este término la función Loss aumenta cuando $S(t_{i+1}) > S(t_i)$, es decir si la señal predecida es creciente.

Teniendo esto en cuenta se utilizaron los parámetros de la red 13 y se procedió a entrenar la misma durante 1500 épocas mas.

Se obtuvieron unos valores de Loss finales de:

Loss Entrenamiento: 0.0006221

y

Loss Predicción: 0.003171

Existe mucha diferencia entre el entrenamiento y la predicción por lo tanto puede haber un overfitting. Los resultados tampoco son los esperados, anque haya un cambio en el loss se siguen pareciendo demasiado a los obtenidos anteriormente."

# ╔═╡ e734c56f-c988-4087-b01b-76feec757a94
PlutoUI.Resource("https://imgur.com/SYPxFHw.png")

# ╔═╡ abc7feed-ab4b-426f-9e91-d0172eca149b
PlutoUI.Resource("https://imgur.com/Wu5IMSb.png")

# ╔═╡ efbfc41a-c889-4dd5-8e14-a2cacbb27c0a
PlutoUI.Resource("https://imgur.com/Jl6rlbE.png")

# ╔═╡ d8d6f127-d301-4629-a6c8-d70f8a680cf8
PlutoUI.Resource("https://imgur.com/kBFPfqD.png")

# ╔═╡ d0fbf95c-d8c2-4aee-bc36-968107442b8a
md"Un test con Darts, una biblioteca de Python para pronósticos y detección de anomalías en series temporales, diseñada para ser user friendly. Contiene una variedad de modelos. Y pueden ser utilizados de la misma manera, utilizando las funciones fit() y predict(), similares a scikit-learn."

# ╔═╡ 5914d0b2-ca96-449b-b0e4-acc8e9b5c2f1
md"# TFT
Un Temporal Fusion Transformer (TFT) es un modelo de aprendizaje profundo diseñado para la predicción de series temporales. Combina la capacidad de los transformers con la capacidad de manejar datos temporales de manera efectiva. Utiliza capas de transformers y un mecanismo de atención para seleccionar partes relevantes de los datos.

Para intentar utilizar sistemas mas complejos se utilizó la biblioteca Darts para experimentar un poco su capacidad antes de intentar utilizar una en específico

En este caso utilizamos un Temporal Fusion Transformer que es una arquitectura mas general que los transformers utilizados para predecir series temporales.

Tomamos exactamente la misma serie de datos, esta vez dando 50 puntos de entrenamiento para predecir los 60 restantes. El TFT utiliza los mismos hiperparámetros que el paper que los presenta y se encuentra como ejemplo del paquete https://arxiv.org/abs/1912.09363


Fue entrenado durante 3000 épocas lo que llevó únicamente 4 minutos y los resultados de una arquitectura sin optimizar y teniendo en cuenta que se utilizó un ejemplo de codigo no son tan malos, se podría incursionar mas en estas redes.
"

# ╔═╡ 2f9ec79e-10cb-4281-97f1-8adcde013e34
PlutoUI.Resource("https://imgur.com/8PVAsQh.png")

# ╔═╡ d31c3f50-210e-4fe2-94dc-a9a8727f18ef
PlutoUI.Resource("https://imgur.com/MQO73Ap.png")

# ╔═╡ c295287a-8bb7-4d67-a448-53e1fa67cae0
PlutoUI.Resource("https://imgur.com/fVdSLOm.png")

# ╔═╡ 3db51b62-550b-4765-a6b1-b324f60f5e2d
PlutoUI.Resource("https://imgur.com/8PhGvED.png")

# ╔═╡ fe1064a9-f781-488a-b098-b1a65ee40e82
md"# Time-series Dense Encoder (TiDE)
Los time series dense encoder son redes con un modelo similar al de los transformers, es decir utilizando la estructura encoder - decoder, pero sin el mecanismo de atención que caracteriza a los trasnsformers. Estas en cambio utilizan una base de redes neuronales convencionales, lo cual las hace mas eficientes a la hora del entrenamiento. Este modelo no da buenos resultados entrenando multiples señales, pero sí se puede usar para series temporales únicas hasta en 1 minuto por lo que se podría pensar en una estructura en paralelo que entrene distintas señales.
"

# ╔═╡ 918839ae-874c-4f4d-8df2-5e29351a9314
PlutoUI.Resource("https://imgur.com/9OO9ERJ.png")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "acf8a7d70217bc877a8d448fd5475fd91c6dd480"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

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
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9e8fed0505b0c15b4c1295fd59ea47b411c019cf"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.2"

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
# ╠═4825797f-1135-42b6-b0ea-e8bb023fdbe9
# ╟─9ecc5f2e-f3c4-4575-ade5-90203c8269b8
# ╠═808c7b06-ac83-4713-8a26-d94fd0bdaf90
# ╟─226682b6-f0b7-4b83-a6f6-a78e089d9096
# ╠═9f6f182a-74a4-4c28-a4be-471c39025391
# ╟─bd9f329f-8d0f-40ca-85b4-0988c379114f
# ╠═e1134168-437c-43b3-9afe-facccd580873
# ╠═f7ad76dd-dcd5-47c1-b04c-df33a7f6c119
# ╠═ef3b1564-fa92-4ced-ba70-436c87a4e33d
# ╠═aca33433-be0b-42f3-8132-7f048989a06e
# ╠═94fb618c-794b-483c-a7ac-adfc888fb77a
# ╟─7e9be462-709a-4601-9b55-b12fe0517c28
# ╠═9f053d4f-fbfc-4663-8d54-3c38c34150c4
# ╟─62640253-a1ac-442e-b78e-446a15c335ed
# ╠═01ca37d6-f276-4d43-aa25-5f45b301ca17
# ╟─5df2f655-4d45-4bfa-89cf-b56642859b85
# ╠═2233898d-28a5-452c-9c26-d0f7ca33c0b2
# ╟─92a5a294-d80a-43ce-a71b-5912bc971fbc
# ╟─a7266221-27be-4775-833d-51bcd0c5cf07
# ╟─dc7017ad-a2bd-437f-8188-5bc7f5abca5e
# ╠═51a802a4-6d2c-4f5e-9be7-89896060534e
# ╠═a51a93ac-855c-4f6c-9f72-5bb8250b6cfc
# ╠═52adf242-1cc9-4434-82fe-eb27cb8bde26
# ╠═d4c0199c-eb23-443f-a98c-f3f492aa5d11
# ╟─7b45fce7-3e52-4ae9-a03f-d20e38888a25
# ╠═c34dd2e8-3841-4845-94f8-e6731841501d
# ╟─5a267d71-635d-46cf-8135-61d9cbf920a6
# ╠═e734c56f-c988-4087-b01b-76feec757a94
# ╠═abc7feed-ab4b-426f-9e91-d0172eca149b
# ╠═efbfc41a-c889-4dd5-8e14-a2cacbb27c0a
# ╠═d8d6f127-d301-4629-a6c8-d70f8a680cf8
# ╟─d0fbf95c-d8c2-4aee-bc36-968107442b8a
# ╟─5914d0b2-ca96-449b-b0e4-acc8e9b5c2f1
# ╠═2f9ec79e-10cb-4281-97f1-8adcde013e34
# ╠═d31c3f50-210e-4fe2-94dc-a9a8727f18ef
# ╠═c295287a-8bb7-4d67-a448-53e1fa67cae0
# ╠═3db51b62-550b-4765-a6b1-b324f60f5e2d
# ╟─fe1064a9-f781-488a-b098-b1a65ee40e82
# ╠═918839ae-874c-4f4d-8df2-5e29351a9314
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
