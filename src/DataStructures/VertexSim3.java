package DataStructures;

import LieAlgebra.SIM3;

public class VertexSim3 {

	public int id;
	public boolean fixed;
	public SIM3 estimate;
	public boolean marginalized;
	
	
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
