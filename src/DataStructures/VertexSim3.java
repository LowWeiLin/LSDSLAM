package DataStructures;

import LieAlgebra.SIM3;

public class VertexSim3 {

	int id;
	boolean fixed;
	SIM3 estimate;
	boolean marginalized;
	
	
	public void setId(int id) {
		this.id = id;
	}
	
	public void setFixed(boolean fixed) {
		this.fixed = fixed;
	}
	
	public void setEstimate(SIM3 estimate) {
		this.estimate = estimate;
	}
	
	public void setMarginalized(boolean marginalized) {
		this.marginalized = marginalized;
	}
	
}
