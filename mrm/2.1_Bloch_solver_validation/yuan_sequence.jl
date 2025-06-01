function yuan_sequence()
    # Grad
    T = 2.6794e-3     # 2.6794 ms
    Gz0 = 1.0 * 1e-2  # 10 mT/m
    z0 = 7e-3         # 7 mm 
    # RF
    h = 8.9313e-7     # s
    A = 0.1750 * 1e-4 # T
    # Window
    L = 4500
    i = 0:floor(Int, 2L/3)
    W = 0.42 .- 0.5 .* cos.(2π .* i ./(2L/3 - 1)) .+ 0.08 .* cos.(4π .* i ./(2L/3 - 1))
    # Sinc
    ω0 = 2π * γ * z0 * Gz0 / 2
    H1 = A .* W .* sin.(ω0 .* (h .* i  .- T/2)) ./ (ω0 .* (h .* i .- T/2))
    # Slice-selective RF pulse
    seq =  Sequence([Grad(0,0); Grad(0,0); Grad(Gz0, T);;], [RF(H1, T);;])
    seq += Sequence([Grad(0,0); Grad(0,0); Grad(-Gz0, T/2);;])
    return seq
end