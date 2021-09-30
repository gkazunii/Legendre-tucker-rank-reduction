using FileIO

function load_tensor(datasetname)
    if datasetname == "AttFace"
        T = load("../data/tensor_att_face.jld2", "tensor_att_face")
    elseif datasetname == "4DLFD"
        T = convert(Array{Float64}, load("../data/tensor_light.jld2", "tensor_light"))
    else
        error("dataset name error")
    end

    return T

end
