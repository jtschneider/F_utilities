using LinearAlgebra, MKL; BLAS.set_num_threads(2)

import F_utilities as Fu

function TFI_Hamiltonian(N; h=1.0, J::Real = 1.0, PBC::Real = 0)
  A = Build_A_TFI(N; h, J, PBC)
  B = Build_B_TFI(N; J, PBC)

  H = zeros(Float64, 2 * N, 2 * N)
  H[1:N, 1:N] = -A
  H[((1:N).+N), 1:N] = -B
  H[1:N, (1:N).+N] = B
  H[(1:N).+N, (1:N).+N] = A

  return H
end


function Build_A_TFI(N; h=1.0, PBC=0.0, J = 1.0)
  prefactor = -0.5 * J
  hopping = prefactor * ones(Float64, N - 1)
  onsite   = prefactor * 2h * ones(Float64, N)

  M_A = LinearAlgebra.diagm(
      -1 => hopping,
       0 => onsite,
      +1 => hopping,
  )

  M_A[1, N] = prefactor * PBC
  M_A[N, 1] = prefactor * PBC

  return M_A
end

function Build_B_TFI(N; PBC, J = 1.0)
  prefactor = -0.5 * J
  hopping = prefactor * ones(Float64, N - 1)

  M_B = LinearAlgebra.diagm(
      -1 => hopping,
      # 0 => zeros(Float64, N),
      +1 => -hopping,
  )
  M_B[1, N] = prefactor * PBC
  M_B[N, 1] = -prefactor * PBC

  return M_B
end



ν = 1.0
	
L_test = 50

ξ_test = 10
h = 1.0 - ξ_test^(-ν)

H = TFI_Hamiltonian(L_test; h = h , J=1.0, PBC=0.0)
HD, U = Fu.Diag_h(H,2)
Γ = Fu.GS_gamma(HD, U)

@show Fu.Energy(Γ, (HD, U))
L_A = L_test÷2
l_init = 1

Gamma_A =  Fu.Reduce_gamma(Γ,L_A,l_init)
λs = abs.(Fu.Eigenvalues_of_rho(Gamma_A))
#I compute the entangement entropy
@show Fu.VN_entropy(Gamma_A)