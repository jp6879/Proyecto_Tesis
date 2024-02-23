using Plots
using CSV
using DataFrames

paths = ["C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\1_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\2_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\3_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\4_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\5_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\6_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\7_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\8_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\9_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\10_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\11_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\12_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\13_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\14_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\16_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\18_losses.csv",
"C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\17_losses.csv"]


df = CSV.read("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\mini_Paralel_training_NODE_batching_2\\Losses\\1_losses.csv", DataFrame)


plot_entrenamiento = plot(df[!, :Loss_Entrenamiento], label = "1")
i = 0
for path in paths
    i+=1
    df = CSV.read(path, DataFrame)
    plot(df[!, :Loss_Entrenamiento][50:end], label = "Entrenamiento $i")
    display(plot!(df[!, :Loss_Predicción][50:end], label = "Predicción $i"))
end
