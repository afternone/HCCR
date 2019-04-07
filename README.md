# HCCR
A Flux implementation for offline Handwritten Chinese Character Recognition

# Requirements
+ Flux
+ Images
+ BSON
+ JLD2
+ CuArrays

# Usage
## Clone the repository
`git clone https://github.com/afternone/HCCR.git`
## Command Line mode (PowerShell)
Open the PowerShell and change to the directory HCCR
+ Single image
```julia
PS C:\Users\han\Desktop\ML\HCCR> julia test.jl image/01.png
image/01.png    不
```
+ Multiple images
```julia
PS C:\Users\han\Desktop\ML\HCCR> julia test.jl image/01.png image/02.png
image/01.png    不
image/02.png    忘
```
+ All images in a directory
```julia
PS C:\Users\han\Desktop\ML\HCCR> julia test.jl (ls .\image\*.png).Fullname
C:\Users\han\Desktop\ML\HCCR\image\01.png       不
C:\Users\han\Desktop\ML\HCCR\image\02.png       忘
C:\Users\han\Desktop\ML\HCCR\image\03.png       初
C:\Users\han\Desktop\ML\HCCR\image\04.png       心
C:\Users\han\Desktop\ML\HCCR\image\05.png       牢
C:\Users\han\Desktop\ML\HCCR\image\06.png       记
C:\Users\han\Desktop\ML\HCCR\image\07.png       使
C:\Users\han\Desktop\ML\HCCR\image\08.png       命
```
##REPL mode
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
# Train your model
TO DO
