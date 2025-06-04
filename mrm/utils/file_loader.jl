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
    download_file(path)
    @info "Loading $(filename)... (This may take a while if the phantom is large)"
    obj = read_phantom(path)
    return obj
end

function download_file(path; url="https://zenodo.org/records/15591102/files/$(basename(path))")
    filename = basename(path)
    foldername = dirname(path)
    # Check if the file exists
    if !isfile(path)
        downloader = Downloads.Downloader()
        downloader.easy_hook = (easy, info) -> Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_TIME, 0)
        # Download the file from Zenodo
        @info "$(filename) not found in $(foldername). \n Downloading it from Zenodo ($(url))"
        Downloads.download(url, path; progress=show_progress)
        @info "$(filename) successfully downloaded to $(foldername)"
    else
        @info "$(filename) found in $(foldername)."
    end
end