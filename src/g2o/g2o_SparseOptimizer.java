package g2o;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import jeigen.DenseMatrix;
import main.Sim3_graph_libraryLibrary;
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
		if (true)
			return 0;
		System.err.println("g2o_SparseOptimizer - optimize()");
		
		
		// Clear
		System.err.println("CLEAR LIBRARY");
		Sim3_graph_libraryLibrary.clear();
		System.err.println("FINISHED CLEAR LIBRARY");
		
		
		// Add vertices
		for (VertexSim3 v : vertices) {
			addVertexToLib(v);
		}
		
		// Add edges
		for (EdgeSim3 e : edges) {
			addEdgeToLib(e);
		}
		
		
		// Start optimization
		System.err.println("g2o_SparseOptimizer - start optimize()");
		int result = Sim3_graph_libraryLibrary.optimize(num_iterations, false);
		System.err.println("OPTIMIZE RESULT: " + result);
		
		// Update all vertex estimates
		for (VertexSim3 v : vertices) {
			Sim3_graph_libraryLibrary.getVertexEstimate(v.id);
			v.estimate = SIM3.exp(getp7());
			
			// TODO: No idea why inverse gives the correct result.
			v.estimate = v.estimate.inverse();
			
		}
		
		// Update all edges
		for (EdgeSim3 e : edges) {
			e.chi2 = (float) Sim3_graph_libraryLibrary.getEdgeChi2(e.id);
			//Sim3_graph_libraryLibrary.getEdgeMeasurement(e.id);
			//e.measurement = SIM3.exp(getp7());
		}
		
		return 0;
	}
	
	
	public void addVertexToLib(VertexSim3 graphVertex) {

		// Set vertex in library
		setp7(graphVertex.estimate().ln());
		Sim3_graph_libraryLibrary.addVertex(graphVertex.id);
		
		// Test read, print.
		Sim3_graph_libraryLibrary.getVertexEstimate(graphVertex.id);
		System.err.println("LIBRARY TEST" + Arrays.toString(getp7()));
		
	}
	
	public void addEdgeToLib(EdgeSim3 edge) {

		// Set edge in library
		double[][] info = new double[7][7];
		for (int i=0 ; i<7 ; i++) {
			for (int j=0 ; j<7 ; j++) {
				info[i][j] = edge.information.get(i, j);
			}
		}
		
		setp7(edge.measurement.ln());
		setp77(info);
		Sim3_graph_libraryLibrary.addEdge(edge.vertices[0].id, edge.vertices[1].id, edge.robustKernel.kernelDelta);
		
	}
	
	
	public void setp7(double[] p7) {
		for (int i=0 ; i<7 ; i++) {
			Sim3_graph_libraryLibrary.setp7(i, p7[i]);
		}
	}
	
	public double[] getp7() {
		double[] result = new double[7];
		for (int i=0 ; i<7 ; i++) {
			result[i] = Sim3_graph_libraryLibrary.getp7(i);
		}
		return result;
	}
	
	public void setp77(double[][] p77) {
		for (int i=0 ; i<7 ; i++) {
			for (int j=0 ; j<7 ; j++) {
				Sim3_graph_libraryLibrary.setp77(i, j, p77[i][j]);
			}
		}
	}

	public double[][] getp77() {
		double[][] result = new double[7][7];
		for (int i=0 ; i<7 ; i++) {
			for (int j=0 ; j<7 ; j++) {
				result[i][j] = Sim3_graph_libraryLibrary.getp77(i, j);
			}
		}
		return result;
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
