package DataStructures;

import g2o.g2o_RobustKernel;
import jeigen.DenseMatrix;
import LieAlgebra.SIM3;

public class EdgeSim3 {

	public int id;
	public SIM3 measurement;
	public DenseMatrix information;
	
	public g2o_RobustKernel robustKernel;
	
	public VertexSim3[] vertices;
	
	public float chi2;
	
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
		vertices = new VertexSim3[size];
	}
	
	public void  setVertex(int i, VertexSim3 vertex) {
		vertices[i] = vertex;
	}

	public void setRobustKernel(g2o_RobustKernel robustKernel) {
		this.robustKernel = robustKernel;
	}


	public float chi2() {
		return chi2;
	}
}
