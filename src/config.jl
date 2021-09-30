using Plots
using Random
using Colors

# seed
# Random.seed!(3)

random_tenosr_dist = "uniform"
#random_tenosr_dist = "gauss"

# experimental setup
rep_max = 10

# zero cutoff KL divergence
eps_thred = 1.0e-12

# save file name option
n_digits = 5

# Real datasets
datasets = [
            "AttFace",      #(112,92,400)
           #"4DLFD"    #(9,9,512,512,3)
           ]
sort!(datasets)

dataset_label_dict = Dict(
                   "AttFace" => "AttFace",
                   #"4DLFD" => "4DLFD",
                )
dataset_label_dict = sort(dataset_label_dict)


# approximation methods
methods_flags = Dict(
             "LT1R" => false,
             "LTR" => true,
             "NTD_KL" => true,
             "NTD_LS" => true,
             "lraSNTD" => true,
            )
non_negative_methods = ["LT1R", "LTR", "NTD_KL", "NTD_LS", "lraSNTD"]
methods = [k for k = keys(methods_flags) if methods_flags[k] == true]

# move to "TLrRR" to top
sort!(methods)
methods = insert!( setdiff(methods, ["LTR"]), 1, "TLR" )
std_method = "LTR"

# #################################
# Experimental design for real data
# #################################

plans = Dict( dataset for dataset = zip(datasets, [ [] for i = 1:length(datasets) ] ) )
# AttFace (112,92,400)
plans["AttFace"] =
[[1,1,1], [3,3,3], [5,5,5], [10,10,10], [15,15,15], [20,20,20], [30,30,30], [40,40,40], [50,50,50], [60,60,60], [70,70,70], [80,80,80]]

#"4DLFD"    #(9,9,512,512,3)
plans["4DLFD"] =
[[1,1,1,1,1], [2,2,2,2,1],[3,3,4,4,1],[3,3,5,5,1],[3,3,6,6,1],[3,3,7,7,1],[3,3,8,8,1],[3,3,16,16,1],[3,3,20,20,1],
 [3,3,40,40,1],[3,3,60,60,1],[3,3,80,80,1]]


# ##########################
# config for other methods
# ##########################

NTD_init = "SVD"

# ##########################
# config for Plots
# ##########################

# plt options for AI data
# fonts list : Helvetica, palatino, courier, times, newcentryschlbk, advantgrade
img_size = (500, 500)
fnt1 = font(20, "times")
fnt2 = font(12, "times")
fnt3 = font(15, "times")

linewidth = 2.5
markerstrokewidth = 1.0
markersize = 12
markerwidth = 2

legend = :none
#legend = :bottomleft
legendfont = fnt1

lw_y0 = 3

# plot options for real data
img_size_wide = (1050, 300)

# color setting
colors = palette(:tab20)

colors_dict = Dict(
                   "LT1R" => :red,
                   "TLrRR" => :red,
                   "NTD_KL" => :black,
                   "NTD_LS" => :blue,
                   "lraSNTD" => :brown,
                )

label_dict = Dict(
                   "LT1R" => "TL1R",
                   "LTR" => "LTD(proposed)",
                   "NTD_KL" => "NTD_KL",
                   "NTD_LS" => "NTD_LS",
                   "lraSNTD" => "lraSNTD",
                )

markershapes_dict = Dict(
                   "LT1R" => :circle,
                   "TLrRR" => :circle,
                   "NTD_KL" => :rect,
                   "NTD_LS" => :dtriangle,
                   "lraSNTD" => :utriangle,
                )

linetype_dict = Dict(
                   "LT1R" => :dashdot,
                   "LTR" => :dashdot,
                   "NTD_KL" => :dashdot,
                   "NTD_LS" => :dot,
                   "lraSNTD" => :dashdot,
                )
