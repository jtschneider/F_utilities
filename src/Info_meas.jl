function Eigenvalues_of_rho(M)
    N = convert(Int64, size(M, 1) / 2)

    evor = ones(Float64, 2^N)

    D, U = LinearAlgebra.eigen(Hermitian(M))

    for iiter = 1:2^N
        index = iiter - 1
        for jiter = 1:N
            evor[iiter] =
                evor[iiter] * round(
                    (
                        mod(index - 1, 2) * D[jiter] +
                        (1 - mod(index - 1, 2)) * (D[jiter+N])
                    ),
                    digits = 32,
                )
            index -= mod(index, 2)
            index = index / 2
        end
    end

    return sort(evor;lt=!isless)
end

function approx_eigenvalues_of_rho(M; mode_cutoff::Int = 20)

    N = size(M, 1)÷2

    D, U = LinearAlgebra.eigen(Hermitian(M))

    trueModeCutoff = min(mode_cutoff,N)

    v_k = sort(D)[ N+1 : N+trueModeCutoff]

    v_k_inverse = 1.0 .- v_k
    
    λs_even = zeros(Float64, 2^(trueModeCutoff))
    λs_odd = zeros(Float64, 2^(trueModeCutoff))

    
    for (index, int_number) in enumerate(0:2^(trueModeCutoff)-1)
        bits_selected = digits(Bool, int_number, base=2, pad = trueModeCutoff)
        anti_selected = .!bits_selected

        parity_pm = prod( (-1) .^ bits_selected )
        if parity_pm == 1
            λs_even[index] = prod([v_k[bits_selected]..., v_k_inverse[anti_selected]...])
        else
            λs_odd[index] = prod([v_k[bits_selected]..., v_k_inverse[anti_selected]...])
        end
    end

    return (sort(λs_even;lt=!isless), sort(λs_odd;lt=!isless))
end


function approx_eigenvalues_of_H(diagonal_H;
	mode_cutoff::Int = 10,
	array_lim::Union{Nothing,Int} = nothing,
    return_occupation::Bool=false,
	untared::Bool = false,
    )
	# this yields the energies of each of the N modes, unfortunately doubled as fermions and anti-fermions are counted:
	energy_per_mode  = diag(diagonal_H)
    N = length(energy_per_mode)÷2
	# note that modes_shifted == sort(modes) and
	modes_shifted = energy_per_mode[end:-1:length(energy_per_mode)÷2+1]
	# that mode are now occupied as 0,1 (either by a particle or a hole),
    # NB: that one hole goes with NEGATIVE energy, while a particle has positive weight.
    # One has therefore N modes with each either a weight of +/-1,
    # giving rise to 2^N different states.

	trueModeCutoff = min(mode_cutoff,N)
	
	occupation_energies_ℤ_even  = zeros(Float64, 2^(trueModeCutoff-1));
	occupation_energies_ℤ_odd   = zeros(Float64, 2^(trueModeCutoff-1));

	# all_occupations = map( n -> digits(Bool, n, base=2, pad = trueModeCutoff) )
	occupations_even = zeros(Bool, trueModeCutoff, 2^(trueModeCutoff-1))
	occupations_odd  = zeros(Bool, trueModeCutoff, 2^(trueModeCutoff-1))
    # map the occupation 0,1 to the energy weight -1,1
	map_01_pm(i::Bool) = (i == false) ? -1 : 1
	
    ind0 = 0
    ind1 = 0

	for (index, int_number) in enumerate(0:2^(trueModeCutoff)-1)
    # we select to slice the (sorted) array of energy per mode with a bit array that must have same length
    # however, we want to select all 2^(mode_cutoff)-1 different energy states
    # by iterating over the binary representation of all integers between 0 and 2^(mode_cutoff)-1
    # this is a good estimation for the first few energy levels and becomes increasingly inaccurate for
    # higher energy levels as it is a priori not clear that a single particle state |0....01> is not smaller
    # in energy than a many-body state |01101010...0>
		bits_selected = digits(Bool, int_number, base=2, pad = trueModeCutoff)
		# parity_pm = prod( (-1) .^ bits_selected )
		parity_pm = count_ones(int_number)
		weights = map_01_pm.(bits_selected)

		if iseven(parity_pm)
            ind0 += 1
			occupation_energies_ℤ_even[ind0]  = sum(weights .* modes_shifted[1:trueModeCutoff])
			occupations_even[:,ind0] = bits_selected
		else
            ind1 += 1
			occupation_energies_ℤ_odd[ind1]  = sum(weights .* modes_shifted[1:trueModeCutoff])
			occupations_odd[:,ind1] = bits_selected
		end
	end
	p_even = sortperm(occupation_energies_ℤ_even)
	p_odd  = sortperm(occupation_energies_ℤ_odd)

    array_lim_R = isnothing(array_lim) ? 2^(trueModeCutoff-1) : array_lim

	absolut_min = untared ? 0.0 : min(
		occupation_energies_ℤ_even[p_even[1]] ,
		occupation_energies_ℤ_odd[p_odd[1]]
	)
	
	tared_even = (occupation_energies_ℤ_even[p_even])[1:array_lim_R] .- absolut_min
	tared_odd  = (occupation_energies_ℤ_odd[p_odd])[1:array_lim_R] .- absolut_min

	if return_occupation
        sorted_occupation_even = (occupations_even[:,p_even])[:,1:array_lim_R]
	    sorted_occupation_odd = (occupations_odd[:,p_odd])[:,1:array_lim_R]
	    return (tared_even, tared_odd, sorted_occupation_even, sorted_occupation_odd)
    else
        return (tared_even, tared_odd)
    end
end

function VN_entropy_old(M)
    N = size(M, 1)

    D, U = LinearAlgebra.eigen(Hermitian((M + M') / 2.0))

    S = 0
    for iiter = 1:N
        nu = abs(round.(D[iiter]; digits = 32))
        if (nu != 0 && nu != 1)
            S -= log(nu) * nu
        end
    end

    return S
end

function VN_entropy(M; accuracy::Float64 = 1e-32)
    λs, U = LinearAlgebra.eigen(Hermitian(M))
    λs_filter = filter( p -> p >= accuracy,  λs)
    S = mapreduce(p -> -log(p)*p, +, λs_filter; init=0.0)
    return S
end



function Purity(M)
    N_f = convert(Int64, size(M, 1) / 2.0)
    M[1, 1] += eps()
    D, U = Diag_gamma(M)

    purity = 1

    for iiter = 1:N_f
        purity = real(purity * (2 * (D[iiter, iiter] - 1) * D[iiter, iiter] + 1))
    end

    return purity
end

function Contour(Γ)
    N = div(size(Γ, 1), 2)
    D, U = Diag_gamma((Γ + Γ') / 2.0)

    p = zeros(Float64, N, N)

    for i = 1:N
        for k = 1:N
            dand = real(U[i, k] * conj(U[i, k]))
            ndda = real(U[i+N, k+N] * conj(U[i+N, k+N]))
            dada = real(U[i, k+N] * conj(U[i, k+N]))
            ndnd = real(U[i+N, k] * conj(U[i+N, k]))
            p[i, k] = 0.5 * (dand + ndda + dada + ndnd)
        end
    end

    Ent_Cont = zeros(Float64, N)
    for i = 1:N
        for k = 1:N
            ν = real(D[k, k])
            if (ν < 0.0)
                print()
                ν = 0
            end
            if (ν > 1.0)
                ν = 1
            end
            if (ν != 0.0 && ν != 1.0)
                Ent_Cont[i] -= p[i, k] * (ν * log(ν) + (1 - ν) * log(1 - ν))
            end
        end
    end

    return Ent_Cont
end
