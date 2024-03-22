using CSV
using DataFrames


paths = [
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\1_[2, 32, 64, 16, 1]_relu_AdamW_5.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\2_[2, 32, 64, 16, 1]_relu_AdamW_15.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\3_[2, 32, 64, 16, 1]_tanh_AdamW_5.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\4_[2, 32, 64, 16, 1]_tanh_AdamW_15.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\5_[2, 32, 64, 16, 1]_swish_AdamW_5.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\6_[2, 32, 64, 16, 1]_swish_AdamW_15.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\7_[2, 50, 64, 16, 1]_relu_AdamW_5.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\8_[2, 50, 64, 16, 1]_relu_AdamW_15.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\10_[2, 50, 64, 16, 1]_tanh_AdamW_15.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\11_[2, 50, 64, 16, 1]_swish_AdamW_5.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\12_[2, 50, 64, 16, 1]_swish_AdamW_15.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\13_[2, 64, 128, 64, 32, 16, 1]_relu_AdamW_5.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\14_[2, 64, 128, 64, 32, 16, 1]_relu_AdamW_15.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\17_[2, 64, 128, 64, 32, 16, 1]_swish_AdamW_5.csv",
    "C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\18_[2, 64, 128, 64, 32, 16, 1]_swish_AdamW_15.csv"
    ]
rows = []


for path in paths
    df = CSV.read(path, DataFrame)
    push!(rows, df[1, :])
end

rows

df = DataFrame(rows)

CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\Tesis_V1\\Proyecto_Tesis\\3-GeneracionDeSeñales\\Exploracion Paralelizada\\Final_trainingNODE\\Resultados\\Resultados_TNODE.csv", df)

minimo_loss_predict = minimum(df[!,"Loss_Final_Predicción"])
id_min = df[df.Loss_Final_Predicción .== minimo_loss_predict, :].ID
println("El minimo valor de Loss_Final_Predicción es: $minimo_loss_predict en la arquitectura con ID: $id_min")

minimo_loss_train = minimum(df[!,"Loss_Final_Entrenamiento"])
id_min_t = df[df.Loss_Final_Entrenamiento .== minimo_loss_train, :].ID
println("El minimo valor de Loss_Final_Entrenamiento es: $minimo_loss_train en la arquitectura con ID: $id_min_t")