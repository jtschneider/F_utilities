function Circulant(cv)
    cv = reverse(cv)
    return Toeplitz(vcat(cv, cv[1:(end-1)]))
end

function Build_Omega(N)
    #Build the matrix omega of dimension 2*N, that is for N sites.
    Ω = zeros(Complex{Float64}, 2 * N, 2 * N)
    ν = (1 / (sqrt(2)))
    Ω[1:N, 1:N] = Diagonal(ν * ones(N))
    Ω[1:N, (1:N).+N] = Diagonal(ν * ones(N))
    Ω[(1:N).+N, 1:N] = Diagonal(im * ν * ones(N))
    Ω[(1:N).+N, (1:N).+N] = Diagonal(-im * ν * ones(N))
    return Ω
end

function Build_FxxTxp(N)
    FxxTxp = zeros(Int64, 2 * N, 2 * N)
    for iiter = 1:N
        FxxTxp[2*iiter-1, iiter] = 1
        FxxTxp[2*iiter, iiter+N] = 1
    end
    return FxxTxp
end

function Build_FxpTxx(N)
    FxpTxx = zeros(Int64, 2 * N, 2 * N)
    for iiter = 1:N
        FxpTxx[iiter, 2*iiter-1] = 1
        FxpTxx[iiter+N, 2*iiter] = 1
    end
    return FxpTxx
end

#Generate a random Hamiltonian with just nearest neighbour interactions
function Random_NNhamiltonian(N)
    ud = rand(N - 1) .+ im * rand(N - 1)
    d = rand(N) .+ im * rand(N)
    bd = rand(N - 1) .+ im * rand(N - 1)
    A = Tridiagonal(bd, d, ud)
    A = (A + A') / 2.0
    B = Tridiagonal(bd, zeros(Complex{Float64}, N), -bd)
    H = zeros(Complex{Float64}, 2 * N, 2 * N)
    H[(1:N), (1:N)] = -conj(A)
    H[(1:N).+N, (1:N)] = -conj(B)
    H[(1:N), (1:N).+N] = B
    H[(1:N).+N, (1:N).+N] = A

    return H
end

function Build_hopping_hamiltonian(N; PBC = false)
    H = zeros(Float64, 2 * N, 2 * N)
    A = zeros(Float64, N, N)
    A[1:N, 1:N] =
        1 / 2 * Tridiagonal(ones(Int64, N - 1), zeros(Int64, N), ones(Int64, N - 1))
    if PBC
        A[1, N] = 1 / 2.0
        A[N, 1] = 1 / 2.0
    end
    H[(1:N), (1:N)] = -A
    H[(1:N).+N, (1:N).+N] = A

    return H
end

function Build_Fourier_matrix(N)
    ω = exp(-im * 2 * pi / N)
    W = ones(Complex{Float64}, N, N)
    U_ω = zeros(Complex{Float64}, 2 * N, 2 * N)
    for i = 1:(N-1)
        for j = 1:(N-1)
            W[i, j] = ω^(i * j)
        end
    end
    W = 1 / sqrt(N) * W
    U_ω[(1:N), (1:N)] = W
    U_ω[(1:N).+N, (1:N).+N] = conj.(W)

    return U_ω
end


function Build_A_TFI(N, J::AbstractVector, h::AbstractVector, PBC::Number)
    @assert length(J) == N - 1
    @assert length(h) == N
    return -0.5 * LinearAlgebra.diagm(-1 => J, 0 => 2 .* h, 1 => J) +
           LinearAlgebra.diagm(N - 1 => [-0.5 * PBC], -(N - 1) => [-0.5 * PBC])
end

function Build_B_TFI(N, J::AbstractVector, PBC::Number)
    @assert length(J) == N - 1
    return -0.5 * LinearAlgebra.diagm(-1 => J, 1 => -J) +
           LinearAlgebra.diagm(N - 1 => [-0.5 * PBC], -(N - 1) => [0.5 * PBC])
end

# Vector API: site-dependent J and h (AD-friendly)
function TFI_Hamiltonian(N, J::AbstractVector, h::AbstractVector; PBC = 0.0)
    A = Build_A_TFI(N, J, h, PBC)
    B = Build_B_TFI(N, J, PBC)
    return Hermitian([-A B; -B A])
end

# Scalar API: uniform couplings via angle θ (backward compatible)
function TFI_Hamiltonian(N, θ::Real; PBC = +1)
    J = ones(typeof(θ), N - 1)
    h = cot(θ) .* ones(typeof(θ), N)
    return TFI_Hamiltonian(N, J, h; PBC = PBC)
end

function Build_A_TFI_impurity(N, h, impurity, PBC; im_pos::Int=N÷2)
    prefactor = -0.5
    hopping = prefactor * ones(Float64, N - 1)
    hopping[im_pos] = prefactor * impurity
    onsite = prefactor * 2h * ones(Float64, N)

    M_A = LinearAlgebra.diagm(
        -1 => hopping,
         0 => onsite,
        +1 => hopping,
    )

    M_A[1, N] = prefactor*PBC
    M_A[N, 1] = prefactor*PBC

    return M_A
end


function Build_B_TFI_impurity(N, impurity, PBC; im_pos::Int=N÷2)

    prefactor = -0.5

    hopping = prefactor * ones(Float64, N - 1)
    hopping[im_pos] = prefactor * impurity


    M_B = LinearAlgebra.diagm(
        -1 => hopping,
        # 0 => zeros(Float64, N),
        +1 => -hopping,
    )
    M_B[1, N] = prefactor*PBC
    M_B[N, 1] = -prefactor*PBC

    return M_B
end

function TFI_Hamiltonian_impurity(N, h, impurity; PBC = +1, im_pos::Int=N÷2)
    A = Build_A_TFI_impurity(N, h, impurity, PBC; im_pos=im_pos)
    B = Build_B_TFI_impurity(N, impurity, PBC; im_pos=im_pos)
    return Hermitian([-A B; -B A])
end

function Build_A_TFI_FIX(N, θ, PBC)
    M_A = LinearAlgebra.diagm(
        -1 => ones(Float64, N - 1),
        0 => 2 * cot(θ) * ones(Float64, N),
        1 => ones(Float64, N - 1),
    )
    M_A[1, N] = PBC
    M_A[N, 1] = PBC
    M_A[1, 1] = 0
    M_A[1, 2] = 0
    M_A[2, 1] = 0

    return -1 / 2.0 .* M_A
end

function Build_B_TFI_FIX(N, PBC)
    M_B = LinearAlgebra.diagm(
        -1 => ones(Float64, N - 1),
        0 => zeros(Float64, N),
        1 => -ones(Float64, N - 1),
    )
    M_B[1, N] = 0
    M_B[N, 1] = -0
    M_B[1, 2] = 0
    M_B[2, 1] = -0

    return -1 / 2.0 .* M_B
end

function TFI_Hamiltonian_FIX(N, θ; PBC = +1)
    A = Build_A_TFI_FIX(N, θ, PBC)
    B = Build_B_TFI_FIX(N, PBC)
    return Hermitian([-A B; -B A])
end







function Build_A_JXJY(Jx::AbstractVector, Jy::AbstractVector, lambdas::AbstractVector)
    N = length(lambdas)
    JpJ = Jx + Jy
    return LinearAlgebra.diagm(
        -1 => -JpJ[1:N-1],
         0 => -2 .* lambdas,
         1 => -JpJ[1:N-1],
    ) + LinearAlgebra.diagm(N - 1 => [-JpJ[N]], -(N - 1) => [-JpJ[N]])
end

function Build_B_JXJY(Jx::AbstractVector, Jy::AbstractVector)
    N = length(Jx)
    JmJ = Jx - Jy
    return LinearAlgebra.diagm(
        -1 => -JmJ[1:N-1],
         1 =>  JmJ[1:N-1],
    ) + LinearAlgebra.diagm(N - 1 => [-JmJ[N]], -(N - 1) => [JmJ[N]])
end

function JXJY_Hamiltonian(N, Jx, Jy, lambda)
    A = 0.5 * Build_A_JXJY(Jx, Jy, lambda)
    B = 0.5 * Build_B_JXJY(Jx, Jy)
    return Hermitian([-A B; -B A])
end
