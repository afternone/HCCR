using StringEncodings

function get_train_imgs(train_path)
    header_size = 10
    imgs = Array{UInt8,2}[]
    labels = String[]
    for file in readdir(train_path)
        if endswith(file, ".gnt")
            open(joinpath(train_path, file), "r") do f
                while !eof(f)
                    header = read(f, header_size)
                    isempty(header) && break
                    sample_size = sum(Int(header[i])*2^(8(i-1)) for i=1:4)
                    label = decode(header[5:6], "GB2312")
                    push!(labels, label)
                    width = convert(Int, header[7]) + header[8]*2^8
                    height = convert(Int, header[9]) + header[10]*2^8
                    header_size+width*height != sample_size && break
                    img = reshape(read(f, width*height), width, height)'
                    push!(imgs, img)
                end
            end
        end
    end
    imgs, labels
end
