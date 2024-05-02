### Chequeemos las funciónes de Loss de los modelos
using CSV
using DataFrames
using Plots

folder_path = "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionCompleta/ExploracionPocosPuntos/3_layers/Losses"  # Replace with the actual path to your folder
csv_files = readdir(folder_path, join=true) |> filter(file -> endswith(file, ".csv"))

file = csv_files[1]

# Epoca desde donde empezar a graficar
start_from_epoch = 1

for (i, file) in enumerate(csv_files)
    df = CSV.read(file, DataFrame)
    epochs = Array(range(1, length(df[!,"Loss_Entrenamiento"])))
    plot = Plots.plot(epochs[start_from_epoch:end], df[start_from_epoch:end,"Loss_Entrenamiento"], label="Loss Entrenamiento", title="Modelo $i", xlabel = "Epochs", ylabel = "Loss", lw = 2)
    plot = Plots.plot!(epochs[start_from_epoch:end], df[start_from_epoch:end,"Loss_Predicción"], label="Loss Validación", lw = 2)
    savefig(plot, "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionCompleta/ExploracionPocosPuntos/3_layers/Losses/Modelo_$i.png")
    # plot = PlotlyJS.plot(df[!,"Loss_Entrenamiento"], label="Loss_Entrenamiento", title="Modelo $i", layout=Layout(yaxis_title="Loss", xaxis_title="Epochs"))
    # plot = PlotlyJS.plot(df[!,"Loss_Predicción"], label="Loss_Validación")
    # PlotlyJS.savefig(plot, "C:/Users/Propietario/Desktop/ib/Tesis_V1/Proyecto_Tesis/3-GeneracionDeSeñales/ExploracionCompleta/Mini/4_layers/Losses/Modelo_$i.html")
end

