package g2o;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import jeigen.DenseMatrix;
import DataStructures.EdgeSim3;
import DataStructures.VertexSim3;
import LieAlgebra.SIM3;

public class g2o_SparseOptimizer {

	public List<EdgeSim3> edges = new ArrayList<EdgeSim3>();
	public List<VertexSim3> vertices = new ArrayList<VertexSim3>();
	
	public boolean verbose = false;
	
	public List<EdgeSim3> edges() {
		return edges;
	}

	public void addEdge(EdgeSim3 edge) {
		edges.add(edge);
	}

	public void addVertex(VertexSim3 graphVertex) {
		vertices.add(graphVertex);
	}

	public void setVerbose(boolean b) {
		verbose = b;
	}

	public void initializeOptimization() {
		
		System.err.println("g2o_SparseOptimizer - initializeOptimization()");
		
		System.err.println("vertices: " + vertices.size());
		System.err.println("edges: " + edges.size());
		
		for (VertexSim3 v : vertices) {
			System.err.println("v - " + v.id + " - " + Arrays.toString(v.estimate().ln()));
		}

		for (EdgeSim3 e : edges) {
			System.err.println("e - " + e.vertices[0].id + " -> " + e.vertices[1].id + " - " 
						+ Arrays.toString(e.measurement.ln()));
		}
		
	}

	public int optimize(int num_iterations, boolean b) {
		// TODO Auto-generated method stub
		System.err.println("g2o_SparseOptimizer - optimize()");
		return 0;
	}
	

	public static void main(String[] args) {
		SIM3 v0 = new SIM3();

		
		SIM3 e01 = new SIM3();
		e01.se3.rotation.matrix = DenseMatrix.eye(3);
		e01.se3.translation = new DenseMatrix(new double[][]{{1},{0},{0}});
		
		SIM3 v1 = v0.mul(e01);
		

		SIM3 e12 = new SIM3();
		e12.se3.rotation.matrix = DenseMatrix.eye(3);
		e12.se3.translation = new DenseMatrix(new double[][]{{0},{1},{0}});
		
		SIM3 v2 = v1.mul(e12);
		
		
		System.out.println("test");

		System.out.println(Arrays.toString(v0.ln()));
		
		System.out.println(Arrays.toString(e01.ln()));
		
		System.out.println(Arrays.toString(v1.ln()));
		
		System.out.println(Arrays.toString(e12.ln()));
		
		System.out.println(Arrays.toString(v2.ln()));
		
	}

}
