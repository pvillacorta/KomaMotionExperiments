function divide_spins_ranges(total_spins::Int, max_spins::Int)
	parts = [] 
	start_idx = 1 
	
	while total_spins > 0
		spins_in_part = min(max_spins, total_spins)
		end_idx = start_idx + spins_in_part - 1
		push!(parts, start_idx:end_idx)
		total_spins -= spins_in_part
		start_idx = end_idx + 1
	end

	return parts
end