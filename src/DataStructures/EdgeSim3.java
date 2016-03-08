package DataStructures;

import g2o.g2o_RobustKernel;
import jeigen.DenseMatrix;
import LieAlgebra.SIM3;

public class EdgeSim3 {

	int id;
	SIM3 measurement;
	DenseMatrix information;
	
	g2o_RobustKernel robustKernel;
	
	
	public void setId(int id) {
		this.id = id;
	}
	
	
	public void setMeasurement(SIM3 measurement) {
		this.measurement = measurement;
	}
	
	public void setInformation(DenseMatrix information) {
		this.information = information;
	}
	
	public void resize(int size) {
		// TODO: implement
		return;
	}
	
	public void  setVertex(int i, VertexSim3 vertex) {
		// TODO: implement
	}
	
	public void setRobustKernel() {
		// TODO: implement
	}

	public void setRobustKernel(g2o_RobustKernel robustKernel) {
		this.robustKernel = robustKernel;
	}
}
