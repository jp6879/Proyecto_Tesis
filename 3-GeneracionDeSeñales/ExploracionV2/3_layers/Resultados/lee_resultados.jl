using CSV
using DataFrames

folder_path = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionV2/3_layers/Resultados"  # Replace with the actual path to your folder
csv_files = readdir(folder_path, join=true) |> filter(file -> endswith(file, ".csv"))

for name in csv_files
    println(name)
end

rows = []

for path in csv_files
    df = CSV.read(path, DataFrame)
    push!(rows, df[1, :])
end

rows

df = DataFrame(rows)

CSV.write(folder_path * "/Resultados_3L.csv", df)

minimo_loss_predict = minimum(df[!,"Loss_Final_Predicción"])
id_min = df[df.Loss_Final_Predicción .== minimo_loss_predict, :].ID
println("El minimo valor de Loss_Final_Predicción es: $minimo_loss_predict en la arquitectura con ID: $id_min")

minimo_loss_train = minimum(df[!,"Loss_Final_Entrenamiento"])
id_min_t = df[df.Loss_Final_Entrenamiento .== minimo_loss_train, :].ID
println("El minimo valor de Loss_Final_Entrenamiento es: $minimo_loss_train en la arquitectura con ID: $id_min_t")