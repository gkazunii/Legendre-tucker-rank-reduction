using Glob
using FileIO
using CSV
using DataFrames
using Statistics
using Colors
using Plots
using Plots.PlotMeasures
using LaTeXStrings

#using PackageCompiler
#create_sysimage(:Plots, sysimage_path="sys_plots.so")

pyplot()

include("../config.jl")
include("../utils/utils.jl")

function get_data(datasetname)
    result_path = "../../result/real/$datasetname/"
    horizon_list = glob("$result_path*")

    horizontal_ax = []
    for horizon in horizon_list
        tmp = split(horizon, "/")[end]
        rank_str = split(tmp,"_")[2:end]
        rank = parse.(Int, rank_str)
        #append!(horizontal_ax, rank[1])
        append!(horizontal_ax, prod(rank))
    end

    KLerror = Dict( method for method = zip(methods, [ [] for i=1:length(methods)] ) )
    LSerror = Dict( method for method = zip(methods, [ [] for i=1:length(methods)] ) )
    runtime = Dict( method for method = zip(methods, [ [] for i=1:length(methods)] ) )

    KLerror_std = Dict( method for method = zip(methods, [ [] for i=1:length(methods)] ) )
    LSerror_std = Dict( method for method = zip(methods, [ [] for i=1:length(methods)] ) )
    runtime_std = Dict( method for method = zip(methods, [ [] for i=1:length(methods)] ) )

    for horizon in horizon_list
        for method in methods
            target_path = horizon*"/"*method
            datalists = glob("$target_path*")
            tmp_for_runtime = []
            tmp_for_LSerror = []
            tmp_for_KLerror = []
            for datalist in datalists
                data = load(datalist)
                append!(tmp_for_runtime, data["runtime"])
                append!(tmp_for_LSerror, data["LSerror"])
                append!(tmp_for_KLerror, data["KLerror"])
            end
            append!(LSerror[method], mean( tmp_for_LSerror ))
            append!(KLerror[method], mean( tmp_for_KLerror ))
            append!(runtime[method], mean( tmp_for_runtime ))
            append!(LSerror_std[method], std( tmp_for_LSerror ))
            append!(KLerror_std[method], std( tmp_for_KLerror ))
            append!(runtime_std[method], std( tmp_for_runtime ))
        end
    end
    return horizontal_ax, KLerror, LSerror, runtime, KLerror_std, LSerror_std, runtime_std
end

function plot_error(horizontal_ax, error, error_std, datasetname ;cost="LS", rel=false, xlog=false, std=true)
    ylabel = "$cost error"
    if rel == true
        rel_error = Dict( method for method = zip(methods, [ [] for i=1:length(methods)] ) )
        for method in methods
            rel_error[method] = ( error[method] - error["TLrRR"] ) ./ error[method]
        end
        ylabel = "Relative $cost error"
        error = rel_error
    end

    plt_time = plot()
    if xlog
        plt_time = plot!(plt_time, xaxis=:log)
    end

    title     = "$datasetname"
    xlabel    = "Number of parameters of core tensor"

    #ylim_min = minimum(minimum.(values(error)))
    #ylim_max = maximum(maximum.(values(error)))
    for method in methods
        if std == false
            error_std[method] .= 0.0
        end
        #if method == "TLrRR" && rel == true
        #    continue
        #end
        plot!(plt_time, horizontal_ax[2:end], error[method][2:end],
            title     = title,
            xlabel    = xlabel,
            ylabel    = ylabel,
            grid      = "on",
            legend    = legend,
            legendfont = legendfont,
            size       = img_size,

            #ylim = (ylim_min, ylim_max),

            label       = label_dict[method],
            linestyle   = linetype_dict[method],
            markershapes= markershapes_dict[method],
            linewidth   = linewidth,

            yguidefont  = fnt3,
            xguidefont  = fnt3,
            xtickfont   = fnt3,
            ytickfont   = fnt3,
            #yerror = error_std[method],
            xticks = ([100,1000,10000], [100, 1000, 10000]),
            #yticks = ([25000,35000,45000, 55000], [25000, 35000, 45000, 55000]),

            markersize  = markersize,
            markercolor = :transparent,
            markerstrokewidth = markerstrokewidth,
            markerstrokecolor = colors_dict[method],
            linecolor = colors_dict[method],
            )
    end
    save_path = "../../png/real/$cost"*"error_$datasetname.pdf"
    savefig(plt_time, save_path)
    println("$save_path has been saved")
end

function plot_time(horizontal_ax, runtime, runtime_std, datasetname; xlog, std=true)
    title     = "$datasetname"
    xlabel    = "Number of parameters of core tensor"

    plt_time = plot()
    if xlog
        plt_time = plot!(plt_time, xaxis=:log)
    end

    ylim_min = minimum(minimum.(values(runtime)))
    ylim_max = maximum(maximum.(values(runtime)))
    for method in methods
        if std == false
            runtime_std[method] .= 0.0
        end
        plot!(plt_time, horizontal_ax[2:end], runtime[method][2:end],
            yaxis     = :log,
            title     = title,
            xlabel    = xlabel,
            ylabel    = "Running time(sec.)",
            grid      = "on",
            legend    = legend,
            legendfont = legendfont,
            size       = img_size,
            linewidth  = linewidth,

            ylim = (ylim_min, ylim_max),
            label       = label_dict[method],
            linestyle   = linetype_dict[method],
            markershapes= markershapes_dict[method],

            yguidefont  = fnt3,
            xguidefont  = fnt3,
            xtickfont   = fnt1,
            ytickfont   = fnt1,
            yerror = runtime_std[method],
            xticks = ([100,1000,10000], [100, 1000, 10000]),

            markersize  = markersize,
            markercolor = :transparent,
            markerstrokewidth = markerstrokewidth,
            markerstrokecolor = colors_dict[method],
            linecolor = colors_dict[method],
             )
    end
    save_path = "../../png/real/time_$datasetname.pdf"
    savefig(plt_time, save_path)
    println("$save_path has been saved")
end

function plots_fix_size(datasetname)
    horizontal_ax, KLerror, LSerror, runtime, KLerror_std, LSerror_std, runtime_std = get_data(datasetname)
    println("Data has been collected")
    plot_time(horizontal_ax, runtime, runtime_std, datasetname, xlog=true)
    plot_error(horizontal_ax, LSerror, LSerror_std, datasetname, cost="LS", rel=false, xlog=true, std=false)
    plot_error(horizontal_ax, KLerror, KLerror_std, datasetname, cost="KL", rel=false, xlog=true, std=false)
end

#datasetname = "AttFace"
#plots_fix_size(datasetname)
datasetname = "4DLFD"
plots_fix_size(datasetname)
#datasetname = "TTSB"
#plots_fix_size(datasetname)
#datasetname = "RGBD"
#plots_fix_size(datasetname)
#datasetname = "GTEA"
#plots_fix_size(datasetname)
#datasetname = "fMRI"
#plots_fix_size(datasetname)
