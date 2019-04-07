using Flux, BSON, Images, DelimitedFiles, Base.Iterators

labels = readdlm("./char_dict.txt",String)[:,2]
BSON.@load "./trained_model.bson" predict

function recognition(imgs,batch_size=128)
    x = (cat(map(x->imresize(1 .- float(Gray.(load(x))),64,64),img)..., dims=4) for img in partition(imgs,batch_size))
    y = labels[vcat([Flux.onecold(predict(i)) for i in x])...]
    for i=1:length(imgs)
        println(imgs[i], '\t', y[i])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    recognition(ARGS)
end
