# function for plotting

function InstantiatePlotFun(dt,dx; plotArgs3d=Dict(),plotArgsC=Dict())

	function plotU(U)
	    n = size(U)[1]-1
	    M = size(U,2)
	    TMesh = 0:dt:n*dt
	    XMeshbd = 0:dx:M*dx # repeat boundary
	    UPlot = [U U[:,1]] # add repeated boundary for plotting
	    return plot(XMeshbd,TMesh,UPlot,st=:surface;plotArgs3d...)
	end

	function contourU(U)
	    n = size(U)[1]-1
    	    M = size(U,2)
	    TMesh = 0:dt:n*dt
	    XMeshbd = 0:dx:M*dx # repeat boundary
	    UPlot = [U U[:,1]] # add repeated boundary for plotting
	    return contour(XMeshbd,TMesh,UPlot;plotArgsC...)
	end
	
	
	
	function contourU!(U)
	    n = size(U)[1]-1
    	    M = size(U,2)
	    TMesh = 0:dt:n*dt
	    XMeshbd = 0:dx:M*dx # repeat boundary
	    UPlot = [U U[:,1]] # add repeated boundary for plotting
	    return contour!(XMeshbd,TMesh,UPlot;plotArgsC...)
	end
	

	
	
	function plotAllU(training_data)
	
	    p1 = Array{Plots.Plot{Plots.GRBackend}}(undef,size(training_data,4))
	    p2 = Array{Plots.Plot{Plots.GRBackend}}(undef,size(training_data,4))
	    p3 = Array{Plots.Plot{Plots.GRBackend}}(undef,size(training_data,4))
	    p4 = Array{Plots.Plot{Plots.GRBackend}}(undef,size(training_data,4))
	    for k=1:size(training_data,4)
		p1[k] = plotU(training_data[1,:,:,k])
		p2[k] = plotU(training_data[2,:,:,k])
		p3[k] = contourU(training_data[1,:,:,k])
		p4[k] = contourU(training_data[2,:,:,k])

	    end
	    return p1,p2,p3,p4 # plot(p1...)
	    
	end

	return plotU, contourU,contourU!, plotAllU
	
end