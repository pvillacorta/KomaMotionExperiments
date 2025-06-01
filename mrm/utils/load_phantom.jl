using Downloads, KomaMRICore.ProgressMeter

const pb = Progress(1; desc = "Downloading phantom...", barlen=40)

function show_progress(total, now)
    if total > 0
        if pb.n != total
            pb.n = total
        end
        update!(pb, now)
    end
end

function load_phantom(filename)
    path = "../phantoms/$(filename)"
    # Check if the file exists
    if !isfile(path)
        downloader = Downloads.Downloader()
        downloader.easy_hook = (easy, info) -> Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_TIME, 0)
        # Download the file from Zenodo
        url = "https://zenodo.org/records/15554360/files/$(filename)"
        @info "$(filename) not found in ../phantoms. \n Downloading it from Zenodo ($(url))"
        Downloads.download(url, path; progress=show_progress)
        @info "Phantom successfully downloaded to $(path)"
    else
        @info "$(filename) found in ../phantoms."
    end
    @info "Loading $(filename)... (This may take a while if the phantom is large)"
    obj = read_phantom(path)
    return obj
end