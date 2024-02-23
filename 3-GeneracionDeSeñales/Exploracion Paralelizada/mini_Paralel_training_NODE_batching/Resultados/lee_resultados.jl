using CSV
using DataFrames


paths = ["C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\1_[2, 32, 64, 16, 1]_relu_AdamW_5.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\4_[2, 32, 64, 16, 1]_tanh_AdamW_10.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\3_[2, 32, 64, 16, 1]_tanh_AdamW_5.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\2_[2, 32, 64, 16, 1]_relu_AdamW_10.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\5_[2, 32, 64, 16, 1]_swish_AdamW_5.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\12_[2, 50, 64, 16, 1]_swish_AdamW_10.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\6_[2, 32, 64, 16, 1]_swish_AdamW_10.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\7_[2, 50, 64, 16, 1]_relu_AdamW_5.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\8_[2, 50, 64, 16, 1]_relu_AdamW_10.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\10_[2, 50, 64, 16, 1]_tanh_AdamW_10.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\11_[2, 50, 64, 16, 1]_swish_AdamW_5.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\13_[2, 32, 64, 64, 32, 16, 1]_relu_AdamW_5.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\14_[2, 32, 64, 64, 32, 16, 1]_relu_AdamW_10.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\16_[2, 32, 64, 64, 32, 16, 1]_tanh_AdamW_10.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\17_[2, 32, 64, 64, 32, 16, 1]_swish_AdamW_5.csv"
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\18_[2, 32, 64, 64, 32, 16, 1]_swish_AdamW_10.csv"]

rows = []


for path in paths
    df = CSV.read(path, DataFrame)
    push!(rows, df[1, :])
end

rows

df = DataFrame(rows)

CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching\\Resultados\\Resultados_parcial.csv", df)
