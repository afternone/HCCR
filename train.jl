using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, @epochs
using Images, FileIO, JLD2, DelimitedFiles
using Base.Iterators: partition, take
using BSON, Dates, Random
using CuArrays
using CUDAnative: tanh, log, exp

@load "./train.jld2" train_imgs train_labels
@load "./test.jld2" test_imgs test_labels
characters = readdlm("./char_dict.txt",String)[:,2]

batch_size = 128
test_set = ((gpu(cat((test_imgs[batch]/255f0)..., dims=4)), gpu(onehotbatch(test_labels[batch],1:3755))) for batch in partition(1:length(test_imgs), batch_size))

model = Chain(
  # input size 64 x 64 x 1 x batch_szie (WHCN order)
  Conv((3, 3), 1 => 64, leakyrelu, pad=1, stride=2),
  BatchNorm(64),
  Conv((3, 3), 64 => 128, leakyrelu, pad=1, stride=1),
  BatchNorm(128),
  Conv((3, 3), 128 => 128, leakyrelu, pad=1, stride=1),
  BatchNorm(128),
  x -> maxpool(x, (3, 3), stride=2, pad=1),
  Conv((3, 3), 128 => 256, leakyrelu, pad=1, stride=1),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, leakyrelu, pad=1, stride=1),
  BatchNorm(256, ϵ=0.001, momentum=0.66),
  x -> maxpool(x, (3, 3), stride=2, pad=1),
  Conv((3, 3), 256 => 384, leakyrelu, pad=1, stride=1),
  BatchNorm(384),
  Conv((3, 3), 384 => 384, leakyrelu, pad=1, stride=1),
  BatchNorm(384),
  Conv((3, 3), 384 => 384, leakyrelu, pad=1, stride=1),
  BatchNorm(384),
  x -> maxpool(x, (3, 3), stride=2, pad=1),
  Conv((3, 3), 384 => 512, leakyrelu, pad=1, stride=1),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, leakyrelu, pad=1, stride=1),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, leakyrelu, pad=1, stride=1),
  BatchNorm(512),
  x -> maxpool(x, (3, 3), stride=2, pad=1),
  x -> reshape(x, :, size(x, 4)),
  Dense(2*2*512, 1024, leakyrelu),
  BatchNorm(1024),
  Dense(1024, 256, leakyrelu),
  BatchNorm(256),
  Dense(256, 3755),
  softmax) |> gpu

loss(x, y) = crossentropy(model(x), y)
opt = ADAM()
# uncomment to apply inverse time decay
# Flux.Optimise.Optimiser(InvDecay(), opt)

# callback function to show accuracy and save model every 600s
evalcb = throttle(600) do
  # show accuracy
  predict = mapleaves(Flux.Tracker.data, model)
  Flux.testmode!(predict, true)
  accuracy(x, y) = mean(cpu(onecold(predict(x))) .== onecold(cpu(y)))
  acc = mean(accuracy(x...) for x in test_set)
  @show(acc)
  # export the model
  model_cpu = cpu(model)
  rightnow = (Dates.format(now(), "yyyy-mm-ddTHH-MM-SS"))
  BSON.@save "model-$(rightnow)-A$(round(UInt16,10000acc)).bson" model_cpu acc
end

@epochs 2 begin
  train_set = ((gpu(cat((train_imgs[batch]/255f0)..., dims=4)), gpu(onehotbatch(train_labels[batch],1:3755))) for batch in partition(randperm(length(train_imgs)), batch_size))
  # uncomment to random rotate images between -15° and 15°
  #train_set = ((gpu(cat(map(x->imrotate(x,rand(-15:15)*pi/180,axes(x),0), train_imgs[batch]/255f0)..., dims=4)), gpu(onehotbatch(train_labels[batch],1:3755))) for batch in partition(randperm(length(train_imgs)), batch_size))
  Flux.train!(loss, params(model), train_set, opt, cb=evalcb)
end
