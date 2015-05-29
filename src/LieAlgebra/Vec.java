package LieAlgebra;

public class Vec {

	

	/**
	 * Calculates magnitude of a vector.
	 */
	public static double magnitude(double[] vec) {
		double magnitude = 0;
		for (int i=0 ; i<vec.length ; i++) {
			magnitude += vec[i] * vec[i];
		}
		return Math.sqrt(magnitude);
	}
	
	/**
	 * Calculates dot product of 2 vectors. Assumes vec0.length == vec1.length.
	 */
	public static double dot(double[] vec0, double[] vec1) {
		double dot = 0;
		for (int i=0 ; i<vec0.length ; i++) {
			dot += vec0[i] * vec1[i];
		}
		return dot;
	}
	
	/**
	 * Calculates cross product of 2 vectors. Assumes vectors of length 3
	 */
	public static double[] cross(double[] a, double[] b) {
		double[] result = {a[1]*b[2]-a[2]*b[1],
						   a[2]*b[0]-a[0]*b[2],
						   a[0]*b[1]-a[1]*b[0]};
		return result;
	}
	
	/**
	 * Scalar multiply s to vector vec0
	 */
	public static void scalarMult(double[] vec0, double s) {
		for (int i=0 ; i<vec0.length ; i++) {
			vec0[i] *= s;
		}
	}
	
	/**
	 * Scalar multiply s to vector vec0
	 */
	public static double[] scalarMult2(double[] vec0, double s) {
		double[] result = new double[vec0.length];
		for (int i=0 ; i<vec0.length ; i++) {
			result[i] = vec0[i] * s;
		}
		return result;
	}
	
	/**
	 * Scalar add s to vector vec0
	 */
	public static void scalarAdd(double[] vec0, double s) {
		for (int i=0 ; i<vec0.length ; i++) {
			vec0[i] += s;
		}
	}
	
	/**
	 * Scalar add s to vector vec0
	 */
	public static double[] scalarAdd2(double[] vec0, double s) {
		double[] result = new double[vec0.length];
		for (int i=0 ; i<vec0.length ; i++) {
			result[i] = vec0[i] + s;
		}
		return result;
	}
	
	/**
	 * Adds 2 vectors
	 */
	public static double[] vecAdd2(double[] vec0, double[] vec1) {
		double[] result = new double[vec0.length];
		for (int i=0 ; i<vec0.length ; i++) {
			result[i] = vec0[i] + vec1[i];
		}
		return result;
	}
	

	/**
	 * Minus 2 vectors
	 */
	public static void vecMinus(double[] vec0, double[] vec1) {
		for (int i=0 ; i<vec0.length ; i++) {
			vec0[i] -= vec1[i];
		}
	}
	
	/**
	 * Minus 2 vectors
	 */
	public static double[] vecMinus2(double[] vec0, double[] vec1) {
		double[] result = new double[vec0.length];
		for (int i=0 ; i<vec0.length ; i++) {
			result[i] = vec0[i] - vec1[i];
		}
		return result;
	}
	
	/**
	 * Make unit vector
	 */
	public static void unit(double[] vec) {
		scalarMult(vec, magnitude(vec));
	}
	

	/**
	 * Matrix to array
	 */
	public static double[][] toArray(jeigen.DenseMatrix mat) {
		return new double[][] {{mat.get(0, 0),mat.get(0, 1),mat.get(0, 2)},
   							   {mat.get(1, 0),mat.get(1, 1),mat.get(1, 2)},
							   {mat.get(2, 0),mat.get(2, 1),mat.get(2, 2)}};
	}
}
