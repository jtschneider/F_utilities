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

function approx_eigenvalues_of_rho(M; mode_cutoff::Int = 14)

    N = size(M, 1)÷2

    evor = ones(Float64, 2^(mode_cutoff))

    D, U = LinearAlgebra.eigen(Hermitian(M))


    D_reduced = D[N-mode_cutoff+1:N+mode_cutoff-1 .+ 1]

    for iiter = 1:2^(mode_cutoff)
        index = iiter - 1
        for jiter = 1:mode_cutoff
            evor[iiter] *= round(
                    (
                        mod(index - 1, 2) * D_reduced[jiter] +
                        (1 - mod(index - 1, 2)) * (D_reduced[jiter+mode_cutoff])
                    ),
                    digits = 64,
                )
            index -= mod(index, 2)
            index = index / 2
        end
    end

    return sort(evor;lt=!isless)
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

function VN_entropy(M; kwargs...)
    λs = entanglement_spectrum(M; kwargs...)
    S = mapreduce(p -> -log(p)*p, +, λs)
    return S
end

function Renyi_entropy(M; order::Real, kwargs...)
    if order <= 0 
        throw(ArgumentError("Rényi order cannot be negative, it must be a positive real number!"))
    end

    if isone(order)
        return VN_entropy(M;kwargs...)
    else
        λs = entanglement_spectrum(M; kwargs...)
        power_sum = mapreduce(p -> p^(order), +, λs)
        return -log(power_sum)/(1-order)
    end
end

function entanglement_spectrum(M; accuracy::Int=32)
    D, U = LinearAlgebra.eigen(Hermitian((M + M' )/ 2.0))
    return round.(sort(abs.(D); lt=!isless); digits=accuracy)
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
