# Lagrangian_GP_PDE
Accompanying sourcecode to article

	Christian Offen
	Machine learning of discrete field theories with guaranteed convergence and uncertainty quantification (2024)
	arXiv Preprint arXiv:2407.07642
	Status: Preprint

Please access the article on <a href="https://arxiv.org/abs/2407.07642">arXiv</a>, <a href="https://ris.uni-paderborn.de/record/55159">LibCat</a>, <a href="https://www.researchgate.net/publication/382118733_Machine_learning_of_discrete_field_theories_with_guaranteed_convergence_and_uncertainty_quantification">ResearchGate</a>, <a href="https://arxiv.org/a/offen_c_1.html">arXiv author page</a>.

To reproduce the experiments, run:<br />
`/wave_equation/Ld_GP_PDE_Wave.jl`<br />
`/schroedinger/Schroedinger_GP_Ld_30.jl`

Please refer to the `Project.toml` files in the subfolders.

![predicted travelling wave](https://github.com/Christian-Offen/Lagrangian_GP_PDE/blob/master/schroedinger/plots/predict_unseen_pComparePrediction_2024-06-05_12:55:39.png?raw=true "predicted versus true solutions in machine learned model of a discrete Schrödinger equation")
<img alt="predicted travelling wave" src="https://github.com/Christian-Offen/Lagrangian_GP_PDE/blob/master/wave_equation/plots/predicted_evolution_waveInit.png?raw=true" title="predicted travelling wave in machine learned model of discrete wave equation" style="width:50%" />
