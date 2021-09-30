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

function get_data(fix_size ; fix)
    if fix == "size"
        size_str =  get_tensorinfo(fix_size)
        result_path = "../../result/synthetic/size$size_str/"
    elseif fix == "rank"
        rank_str =  get_tensorinfo(fix_size)
        result_path = "../../result/synthetic/reqrank$rank_str/"
    else
        error("invalid fix $fix")
    end
    horizon_list = glob("$result_path*")

    horizontal_ax = []
    for horizon in horizon_list
        tmp = split(horizon, "/")[end]
        rank_str = split(tmp,"_")[2:end]
        rank = parse.(Int, rank_str)
        append!(horizontal_ax, rank[1])
        #append!(horizontal_ax, prod(rank))
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

function plot_error(horizontal_ax, error, error_std, fixed_target ;fix, cost="LS", rel=false, xlog=false)
    ylabel = "$cost error"
    if rel == true
        rel_error = Dict( method for method = zip(methods, [ [] for i=1:length(methods)] ) )
        for method in methods
            rel_error[method] = ( error[method] - error["TLrRR"] ) ./ error[method]
            error_std[method] .= 0.0
        end
        ylabel = "Relative $cost error"
        error = rel_error
    end

    plt_time = plot()
    if xlog
        plt_time = plot!(plt_time, xaxis=:log)
    end

    if fix == "size"
        tensor_size = fixed_target
        title     = "Random $tensor_size tensor"
        xlabel    = "Tensor rank"
    elseif fix == "rank"
        tensor_rank = fixed_target
        title     = "Fixed $tensor_rank rank tensor"
        xlabel    = "Tensor size"
    else
        error("invalid fix $fix")
    end

    #ylim_min = minimum(minimum.(values(error)))
    #ylim_max = maximum(maximum.(values(error)))
    ylim_min = -0.05
    ylim_max = +0.05
    for method in methods
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
            xtickfont   = fnt1,
            ytickfont   = fnt1,
            #yerror = error_std[method],
            #xticks = [2,4,6,8,10,12],
            xticks = ([20,100,300,1000], [20, 100, 300, 1000]),

            markersize  = markersize,
            markercolor = :transparent,
            markerstrokewidth = markerstrokewidth,
            markerstrokecolor = colors_dict[method],
            linecolor = colors_dict[method],
            )
    end
    fixval = get_tensorinfo(fixed_target)
    save_path = "../../png/synthetic/$cost"*"error_$fix$fixval.pdf"
    savefig(plt_time, save_path)
    println("$save_path has been saved")
end

function plot_time(horizontal_ax, runtime, runtime_std, fixed_target; fix, xlog)
    if fix == "size"
        tensor_size = fixed_target
        title     = "Random $tensor_size tensor"
        xlabel    = "Tensor rank"
    elseif fix == "rank"
        tensor_rank = fixed_target
        title     = "Fixed $tensor_rank rank tensor"
        xlabel    = "Tensor size"
    else
        error("invalid fix $fix")
    end

    plt_time = plot()
    if xlog
        plt_time = plot!(plt_time, xaxis=:log)
    end

    ylim_min = minimum(minimum.(values(runtime)))
    ylim_max = maximum(maximum.(values(runtime)))
    for method in methods
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
            #xticks = ([20,100,300,1000], [20, 100, 300, 1000]),
            xticks = [2,4,6,8,10,12],

            markersize  = markersize,
            markercolor = :transparent,
            markerstrokewidth = markerstrokewidth,
            markerstrokecolor = colors_dict[method],
            linecolor = colors_dict[method],
             )
    end
    fixval = get_tensorinfo(fixed_target)
    save_path = "../../png/synthetic/time_$fix$fixval.pdf"
    savefig(plt_time, save_path)
    println("$save_path has been saved")
end

function plots_fix_size(tensor_size)
    horizontal_ax, KLerror, LSerror, runtime, KLerror_std, LSerror_std, runtime_std = get_data(tensor_size, fix="size")
    println("Data has been collected")
    plot_time(horizontal_ax, runtime, runtime_std, tensor_size,  fix="size", xlog=false)
    plot_error(horizontal_ax, LSerror, LSerror_std, tensor_size, fix="size", cost="LS", rel=false, xlog=false)
    plot_error(horizontal_ax, KLerror, KLerror_std, tensor_size, fix="size", cost="KL", rel=false, xlog=false)
end

function plots_fix_rank(reqrank)
    horizontal_ax, KLerror, LSerror, runtime, KLerror_std, LSerror_std, runtime_std = get_data(reqrank, fix="rank")
    println("Data has been collected")
    plot_time(horizontal_ax,  runtime, runtime_std, reqrank, fix="rank", xlog=true)
    plot_error(horizontal_ax, LSerror, LSerror_std, reqrank, fix="rank", cost="LS", rel=false, xlog=true)
    plot_error(horizontal_ax, KLerror, KLerror_std, reqrank, fix="rank", cost="KL", rel=false, xlog=true)
end

#plots_fix_size([25,25,25,25,25])
#plots_fix_size([30,30,30,30,30])
#plots_fix_size([200,200,200])
#plots_fix_rank([3,3,3,3,3])
plots_fix_rank([10,10,10])
