function test(z_policy, data_set::DataSet)
    values = [z_policy(row.x, row) for row in data_set]
    return (values=values, mean=mean(values))
end
