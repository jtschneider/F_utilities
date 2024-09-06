using Revise
import F_utilities as Fu

H = Fu.TFI_Hamiltonian_impurity(100,1.0,1.0; PBC=0.0)
HD, U = Fu.Diag_h(H,2)
modes = diag(HD)




function energies_free_fermions(N::Int, h::Real, impurity::Real; PBC::Real = 0.0, mode_cutoff::Int = 8, im_pos=N÷2)
	H = Fu.TFI_Hamiltonian_impurity(N, h, impurity; PBC=PBC, im_pos=im_pos)
	HD, U  = Fu.Diag_h(H,2)
	modes  = diag(HD)

	occupation_energies  = zeros(Float64, 2 ^(2mode_cutoff) -1);

	for d in 1:2 ^(2mode_cutoff) -1
		bits_selected = [fill(false,N-mode_cutoff)..., digits(Bool, d, base=2, pad = 2mode_cutoff)..., fill(false,N-mode_cutoff)...]
		occupation_energies[d]  = sum( modes[bits_selected] )
	end
	p = sortperm(occupation_energies)

	λs = occupation_energies[p] .- occupation_energies[p[1]]

	return  λs ./ (λs[2])
end


N = 100
h = 1.0
impurity=1.1
λ_OBC = energies_free_fermions(N,h,impurity; PBC=0.0)

round.(λ_OBC[1:30] * 1/(λ_OBC[2]), digits=2)