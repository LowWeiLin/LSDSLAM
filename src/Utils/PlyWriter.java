package Utils;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.List;

import jeigen.DenseMatrix;

public class PlyWriter {

	public static void writePoints(String filename, List<DenseMatrix> points) throws FileNotFoundException, UnsupportedEncodingException {
		
		// Write header
		
		PrintWriter writer = new PrintWriter(filename, "ASCII");
		String header = "ply\n" +
						"format ascii 1.0\n" +
						"element vertex " + points.size() + "\n" +
						"property float x\n" +
						"property float y\n" +
						"property float z\n" +
						"property uchar red\n" +
						"property uchar green\n" +
						"property uchar blue\n" +
						"end_header\n";
		writer.printf(header);
		

		// Write points
		
		for (int i=0 ; i<points.size() ; i++) {
			assert(points.get(i) != null);
			
			writer.printf("%.6f ", points.get(i).get(0, 0));
			writer.printf("%.6f ", points.get(i).get(1, 0));
			writer.printf("%.6f ", points.get(i).get(2, 0));
			
			if (points.get(i).rows == 6) {
				// Write color if given
				writer.printf("%d ", (int)points.get(i).get(3, 0));
				writer.printf("%d ", (int)points.get(i).get(4, 0));
				writer.printf("%d ", (int)points.get(i).get(5, 0));
			} else {
				// Default to white
				writer.print("255 255 255");
			}
			
			writer.printf("\n");
 		}
		
		writer.close();
	}
	
}
