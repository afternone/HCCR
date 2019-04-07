using Flux, Statistics, BSON, Dates, Random, Images, FileIO, JLD2
using Flux: onehotbatch, onecold, crossentropy, throttle, @epochs
using Base.Iterators: partition, take
using CuArrays
using CUDAnative: tanh, log, exp

@load "./train.jld2" train_imgs train_labels
@load "./test.jld2" test_imgs test_labels

batch_size = 128
test_set = ((gpu(cat((test_imgs[batch]/255f0)..., dims=4)), gpu(onehotbatch(test_labels[batch],1:3755))) for batch in partition(1:length(test_imgs), batch_size))

function conv_unit(chanel, nb_filters, mp=false)
  conv_bn = Chain(Conv((3, 3), chanel => nb_filters, leakyrelu, pad=1, stride=1),BatchNorm(nb_filters))
  mp ? Chain(conv_bn..., x -> maxpool(x, (3, 3), stride=2, pad=1)) : conv_bn
end

model = Chain(
  # input size 64 x 64 x 1 x batch_szie (WHCN order)
  Conv((3, 3), 1 => 64, leakyrelu, pad=1, stride=2),
  BatchNorm(64),
  conv_unit(64, 128)...,
  conv_unit(128, 128, true)...,
  conv_unit(128, 256)...,
  conv_unit(256, 256, true)...,
  conv_unit(256, 384)...,
  conv_unit(384, 384)...,
  conv_unit(384, 384, true)...,
  conv_unit(384, 512)...,
  conv_unit(512, 512)...,
  conv_unit(512, 512, true)...,
  x -> reshape(x, :, size(x, 4)),
  Dense(2*2*512, 1024, leakyrelu),
  BatchNorm(1024),
  Dense(1024, 256, leakyrelu),
  BatchNorm(256),
  Dense(256, 3755),
  softmax) |> gpu

loss(x, y) = crossentropy(model(x), y)
predict = mapleaves(Flux.Tracker.data, model)
accuracy(x, y) = mean(cpu(onecold(predict(x))) .== onecold(cpu(y)))
opt = ADAM()
# uncomment to apply inverse time decay
# Flux.Optimise.Optimiser(InvDecay(), opt)

# callback function to show accuracy and save model every 600s
evalcb = throttle(600) do
  Flux.testmode!(predict)
  acc = mean(accuracy(x...) for x in test_set)
  @show(acc)
  model_cpu = cpu(model)
  rightnow = (Dates.format(now(), "yyyy-mm-ddTHH-MM-SS"))
  BSON.@save "model-$(rightnow)-A$(round(UInt16,10000acc)).bson" model_cpu acc
end

@epochs 2 begin
  train_set = ((gpu(cat((train_imgs[batch]/255f0)..., dims=4)), gpu(onehotbatch(train_labels[batch],1:3755))) for batch in partition(randperm(length(train_imgs)), batch_size))
  Flux.train!(loss, params(model), train_set, opt, cb=evalcb)
end
