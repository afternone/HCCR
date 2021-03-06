# HCCR
A Flux implementation for offline Handwritten Chinese Character Recognition

## Requirements
+ Flux
+ Images
+ BSON
+ JLD2
+ CuArrays

## Get started
1. Clone the repository
`git clone https://github.com/afternone/HCCR.git`
2. Download the `trained_model.bson` from [here](https://pan.baidu.com/s/1YP3_KdrrdWQxacU8eyUnXg) (access code: 47kh) and put it in the root directory `HCCR`

## Usage
### Command Line mode (PowerShell)
Open the PowerShell and change to the directory `HCCR`
+ Single image
```
PS C:\HCCR> julia test.jl image/01.png
image/01.png    不
```
+ Multiple images
```
PS C:\HCCR> julia test.jl image/01.png image/02.png
image/01.png    不
image/02.png    忘
```
+ All images in a directory
```
PS C:\HCCR> julia test.jl (ls .\image\*.png).Fullname
C:\image\01.png       不
C:\image\02.png       忘
C:\image\03.png       初
C:\image\04.png       心
C:\image\05.png       牢
C:\image\06.png       记
C:\image\07.png       使
C:\image\08.png       命
```
### REPL mode
```julia
julia> include("test.jl")

julia> recognition(["image/01.png"])
image/01.png    不

julia> recognition(joinpath.("image",readdir("image")))
image\01.png    不
image\02.png    忘
image\03.png    初
image\04.png    心
image\05.png    牢
image\06.png    记
image\07.png    使
image\08.png    命
julia>
```
## Train your model
Download datasets `train.jld2` and `test.jld2` from [here](https://pan.baidu.com/s/1YP3_KdrrdWQxacU8eyUnXg) (access code: 47kh) and put them in `HCCR`.
Then train your model as follows:
```julia
julia train.jl
```
The accuracy of the `trained_model.bson` is 95.22%.
