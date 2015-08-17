$$\sum_{x=1}^{10} x^2$$

{% highlight julia %}
function FFNNet(layers::Vector{FFNNLayer}, inputsize::Int)
    # Create a vector of weight matrices
    weights = Array(Matrix{Float64}, length(layers))

    ɛ(a,b) = sqrt(6)/(sqrt(a + b))

    # The first weight matrix is from the input (including bias)
    #  to the first layer (excluding bias unit)
    eps = ɛ(size(layers[1]), inputsize)
    weights[1] = rand(size(layers[1]), inputsize + 1)*2*eps - eps

    # Matrices from layer i-1 (including bias) to layer i
    for i in 2:length(layers)
        eps = ɛ(size(layers[i]), size(layers[i-1]))
        weights[i] = rand(size(layers[i]), size(layers[i-1]) + 1)
    end

    return FFNNet(layers, weights, inputsize)
end
{% endhighlight %}
