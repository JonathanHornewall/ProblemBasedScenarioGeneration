struct DataSet{X,W,T,H,Q}
    x_data::X
    xi_W_data::W
    xi_T_data::T
    xi_h_data::H
    xi_q_data::Q
end

function Base.length(data_set::DataSet)
    return _dataset_component_length(data_set.x_data)
end

function Base.getindex(data_set::DataSet, i::Integer)
    return (
        x=_dataset_observation(data_set.x_data, i),
        xi_W=_dataset_observation(data_set.xi_W_data, i),
        xi_T=_dataset_observation(data_set.xi_T_data, i),
        xi_h=_dataset_observation(data_set.xi_h_data, i),
        xi_q=_dataset_observation(data_set.xi_q_data, i),
    )
end

function Base.iterate(data_set::DataSet, state::Int=1)
    state > length(data_set) && return nothing
    return (data_set[state], state + 1)
end

function _dataset_component_length(data)
    data === nothing && return 0
    data isa AbstractMatrix && return size(data, 2)
    data isa AbstractVector && return length(data)
    return length(data)
end

function _dataset_observation(data, i::Integer)
    data === nothing && return nothing
    if data isa AbstractMatrix
        return data[:, i]
    elseif data isa AbstractVector
        return data[i]
    else
        return getindex(data, i)
    end
end
