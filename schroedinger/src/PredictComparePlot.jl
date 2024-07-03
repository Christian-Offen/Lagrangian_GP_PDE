function PredictCompare(DELSolve_ml,DELSolve_ref, dt,dx, u0,u1,N,var_el; prefix="", postfix="")

	println("\nCall to PredictCompare: Predict motions, compute standard deviation, and compare to reference")
	println(prefix)
	println(postfix)


	M = size(u0,2)
	dimU = size(u0,1)

	XMesh = 0:dx:(M-1)*dx # periodic mesh
	XMeshbd = 0:dx:M*dx # periodic mesh
	TMesh = 0:dt:N*dt       
	
	
	pInitVals=plot(XMesh,u0',xlabel=L"x",label=[L"\mathrm{re}(u_0)" L"\mathrm{im}(u_0)"])
	plot!(XMesh,u1',xlabel=L"x",label=[L"\mathrm{re}(u_1)" L"\mathrm{im}(u_1)"])
	savefig(pInitVals,"plots/"*prefix*"pInitVals"*postfix*".pdf")
	
	#Learned_Ldxd(U0,U1) = sum([Ld_ml(U0[:,j],U1[:,j],U0[:,modInd(j+1)],U1[:,modInd(j+1)]) for j=1:M])
	#U_ref = DELSolve(Ldxd_refInstance, u0, u1, N);
	#U_ml = DELSolve(Learned_Ldxd, u0, u1, N);
	U_ref = DELSolve_ref(u0,u1,N)
	U_ml = DELSolve_ml(u0,u1,N)
	
	max_err = maximum(abs.(U_ml-U_ref))
	l2_err  = sqrt(sum((U_ml-U_ref).^2)*dx*dt)
	
	plotU, contourU = InstantiatePlotFun(dt,dx);
	
	pUml2Re =plotU(U_ml[1,:,:]')
	plot!(title=L"\mathrm{re}(U_\mathrm{ml})")
	pUml2Im =plotU(U_ml[2,:,:]')
	plot!(title=L"\mathrm{im}(U_\mathrm{ml})")

	pUref2Re=plotU(U_ref[1,:,:]')
	plot!(title=L"\mathrm{re}(U_\mathrm{ref})")
	pUref2Im=plotU(U_ref[2,:,:]')
	plot!(title=L"\mathrm{im}(U_\mathrm{ref})")
	pComparePrediction=plot(pUml2Re,pUref2Re,pUml2Im,pUref2Im,layout=(2,2))
	
	savefig(pUml2Re,"plots/"*prefix*"PredictionUml2Re"*postfix*".png");
	savefig(pUml2Im,"plots/"*prefix*"PredictionUml2Im"*postfix*".png");
	savefig(pUref2Re,"plots/"*prefix*"PredictionUref2Re"*postfix*".png");
	savefig(pUref2Im,"plots/"*prefix*"PredictionUref2Im"*postfix*".png");
	savefig(pComparePrediction,"plots/"*prefix*"pComparePrediction"*postfix*".png");
	
	
	stencils_Uml = CollectStencils(U_ml)
	variance_Uml = mapslices(var_el,stencils_Uml,dims=(2,3))
	variance_Uml = dropdims(variance_Uml,dims=3)
	st_deviation_Uml=sqrt.((x->max(0.,x)).(variance_Uml)); # map numerically negative entries to 0
	
	# re-arrange to special-temporal
	st_deviation_spatial_temp = Array{Union{Float64}}(undef,2,M,N+1)
	st_deviation_spatial_temp[:,:,[1,end]]  .= NaN;
	st_deviation_spatial_temp[1,:,2:end-1]  = reshape(st_deviation_Uml[:,1],(M,N-1));
	st_deviation_spatial_temp[2,:,2:end-1]  = reshape(st_deviation_Uml[:,2],(M,N-1));
	st_deviation_spatial_temp_bnd1=[st_deviation_spatial_temp[1,:,:]; st_deviation_spatial_temp[1,1,:]'];
	st_deviation_spatial_temp_bnd2=[st_deviation_spatial_temp[2,:,:]; st_deviation_spatial_temp[2,1,:]'];
	
	pStdEl1=heatmap(XMeshbd,TMesh,st_deviation_spatial_temp_bnd1',xlabel=L"x",ylabel=L"t");
	pStdEl2=heatmap(XMeshbd,TMesh,st_deviation_spatial_temp_bnd2',xlabel=L"x",ylabel=L"t");

	pStdEL_both = plot(pStdEl1,pStdEl2,size=(1200,300))
	
	savefig(pStdEl1,"plots/"*prefix*"pStdEl1"*postfix*".pdf")
	savefig(pStdEl2,"plots/"*prefix*"pStdEl2"*postfix*".pdf")
	savefig(pStdEL_both,"plots/"*prefix*"pStdEL_both"*postfix*".pdf")
	
	
	save("data/"*prefix*"data"*postfix*".jld2","U_ref",U_ref,"U_ml",U_ml,"st_deviation_Uml",st_deviation_Uml)
	
	return 	U_ml,U_ref,st_deviation_spatial_temp, (pInitVals,pUml2Re,pUref2Re,pUml2Im,pUref2Im,pStdEl1,pStdEl2), (max_err,l2_err)
	
end
	
	
