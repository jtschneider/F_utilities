function Diag_real_skew(M, rand_perturbation::Int64 = 0)
    N = div(size(M, 1), 2)

    # Random perturbation before forcing skew symmetrisation
    if (rand_perturbation != 0)
        if (rand_perturbation == 1)
            random_M = rand(2N, 2N) * eps()
            random_M = (random_M - random_M') / 2.0
            M += random_M
        end
        if (rand_perturbation == 4)
            r = eps() / 2
            M = copy(M)
            M[1, N+2] += r;  M[N+2, 1] -= r
            M[2, N+1] -= r;  M[N+1, 2] += r
        end
        if (rand_perturbation == 5)
            r = eps()
            M = copy(M)
            M[1, 2] += r;  M[2, 1] -= r
        end
    end

    # M = real((M-M')/2.); #Force skew-symmetry
    # #Random pertubation after the skew symmetrization
    # if (rand_perturbation != 0)
    #   if (rand_perturbation == 2)    #Perturb the diagonal elements (loose perfect skew-symmetry)
    #     M += diagm(rand(2*N)*eps())
    #   end
    #   if (rand_perturbation == 3)  #Perturb the whole matrix (loose perfect skew-symmetry)
    #     random_M = 1*rand(2*N,2*N)*eps();
    #     random_M = (random_M-random_M')/2.;
    #     M += random_M;
    #   end
    # end

    Schur_object = LinearAlgebra.schur(M)

    Schur_ort_i = Schur_object.vectors
    Schur_blocks_i = Schur_object.Schur

    # Reorder so that all 2x2 blocks have positive value in top right, accounting for 1x1 blocks.
    swap_perm = Vector{Int64}(undef, 0)
    sizehint!(swap_perm, 2N)
    iiter = 1
    while iiter < 2N
        if abs(Schur_blocks_i[iiter, iiter]) >= abs(Schur_blocks_i[iiter+1, iiter]) # alternative: s_b_i[i+1, i] == 0
            # We have a 1x1 block - move it to the front. These always come in pairs, but not always sequentially.
            pushfirst!(swap_perm, iiter)
            iiter += 1
        elseif (Schur_blocks_i[iiter+1, iiter] >= 0.0)
            # Flipped 2x2 block
            push!(swap_perm, iiter + 1, iiter)
            iiter += 2
        else
            # Unflipped 2x2 block
            push!(swap_perm, iiter, iiter + 1)
            iiter += 2
        end
    end
    if iiter == 2N  # catch the final 1x1 block if it exists
        pushfirst!(swap_perm, 2N)
    end

    # Sort the blocks, λ_1>=λ_2>=...>=λ_N with λ_1 the coefficient in the upper left block.
    # In the common case (no 1x1 blocks / zero eigenvalues) we can compose the two permutations
    # directly, avoiding two intermediate 2N×2N matrix allocations.
    has_1x1_blocks = (swap_perm[1] > swap_perm[2])  # pushfirst! reverses order for 1x1 pairs
    full_psort = zeros(Int64, 2N)
    if !has_1x1_blocks
        block_evals = [Schur_blocks_i[swap_perm[2k-1], swap_perm[2k]] for k in 1:N]
        psort = sortperm(block_evals, rev = true)
        full_psort[begin:2:end]   .= 2 .* psort .- 1
        full_psort[begin+1:2:end] .= 2 .* psort
        combined = swap_perm[full_psort]
        M_f = Schur_blocks_i[combined, combined]
        O_f = real.(Schur_ort_i[:, combined])
    else
        M_temp = Schur_blocks_i[swap_perm, swap_perm]
        O_temp = Schur_ort_i[:, swap_perm]
        psort = sortperm(diag(M_temp, 1)[begin:2:end], rev = true)
        full_psort[begin:2:end]   .= 2 .* psort .- 1
        full_psort[begin+1:2:end] .= 2 .* psort
        M_f = M_temp[full_psort, full_psort]
        O_f = real.(O_temp[:, full_psort])
    end

    return M_f, O_f
end


function Diag_h(M, rand_perturbation::Int64 = 0)
    N = size(M, 1) ÷ 2

    Ω = Build_Omega(N)
    M_skew = real(-im * Ω * M * Ω')
    M_skew = (M_skew - M_skew') / 2.0
    _, O = Diag_real_skew(M_skew, rand_perturbation)

    # F_xptxx is a permutation [1,3,…,2N-1, 2,4,…,2N]; apply as column selection (O(N²) vs O(N³))
    p = [1:2:2N; 2:2:2N]
    U_f = Ω' * O[:, p] * Ω

    return real.(U_f' * M * U_f), U_f
end

function Diag_gamma(Γ, rand_perturbation::Int64 = 0)
    Γ = Hermitian((Γ + Γ') / 2.0)
    γ, U = Diag_h(Γ - 0.5 * I, rand_perturbation)

    return U' * Γ * U, U#real(γ+0.5*eye(size(Γ,1))),U
end


function GS_gamma(D, U)
    N = div(size(D, 1), 2)
    T = eltype(U)
    d = real.(diag(D)[1:N])
    # Occupy particle mode if energy >= 0, hole mode if energy < 0
    particle = [dₖ < 0 ? zero(T) : one(T) for dₖ in d]
    Gamma = U * Diagonal(vcat(particle, one(T) .- particle)) * U'
    return Hermitian((Gamma + Gamma') / 2)
end





function Energy(Γ, (D, U))
    N_f = size(Γ, 1) ÷ 2
    Γ = Hermitian((Γ + Γ') / 2)
    Γ_d = real.(diag(U' * Γ * U))
    d   = real.(diag(D))
    return dot(Γ_d[1:N_f], d[N_f+1:end]) + dot(Γ_d[N_f+1:end], d[1:N_f])
end

function Evolve(M, (D, U), t)
    M = Hermitian((M + M') / 2)
    M_diag = U' * M * U
    phases = Diagonal(exp.(2im .* real.(diag(D)) .* t))
    M_evolv = U * (phases * M_diag * phases') * U'
    return Hermitian((M_evolv + M_evolv') / 2)
end
