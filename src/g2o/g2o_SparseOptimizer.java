package g2o;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import DataStructures.EdgeSim3;
import DataStructures.VertexSim3;

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
		// TODO Auto-generated method stub
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

}
