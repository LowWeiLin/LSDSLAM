package IO;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import jeigen.DenseMatrix;
import DataStructures.Frame;
import DataStructures.KeyFrameGraph;
import DataStructures.VertexSim3;
import Utils.PlyWriter;

public class KeyFrameGraphDisplay {
	
	KeyFrameGraph keyFrameGraph;

	public KeyFrameGraphDisplay(KeyFrameGraph keyFrameGraph) {
		this.keyFrameGraph = keyFrameGraph;
	}
	
	public void writePointCloudToFile(String filename
			) throws FileNotFoundException, UnsupportedEncodingException {
		
		Map<Integer, KeyFrameDisplay> idTokeyFramesDisplay = new TreeMap<Integer, KeyFrameDisplay>();
		
		// Get all KFD
		for (Frame kf : keyFrameGraph.keyframesAll) {
			idTokeyFramesDisplay.put(kf.id(), new KeyFrameDisplay(kf));
		}
		
		// Update camToWorld with estimate
		for (VertexSim3 v : keyFrameGraph.graph.vertices) {
			idTokeyFramesDisplay.get(v.id).camToWorld = v.estimate();
		}
		
		List<DenseMatrix> allPoints = new ArrayList<DenseMatrix>();
		
		int count = 0;
		for (KeyFrameDisplay kfd : idTokeyFramesDisplay.values()) {
			count++;
			
			System.out.println("WRITING KF " + kfd.keyframe.id());

			
			
			List<DenseMatrix> points = kfd.getPointCloud(1);
			List<DenseMatrix> cameraPoints = kfd.generateCameraPosePoints();

			// Camera positions
			for (DenseMatrix p : cameraPoints) {
				if (p != null) {
					allPoints.add(p);
				}
			}

			if (idTokeyFramesDisplay.size() > 5 && count < 5) {
				// Skip merging first few KFs, since they may not be accurate yet.
				continue;
			}
			// Point positions
			for (DenseMatrix p : points) {
				if (p != null) {
					allPoints.add(p);
				}
			}
			
		}
		

		// Write to file
		PlyWriter.writePoints(filename, allPoints);
		
	}
	
}
