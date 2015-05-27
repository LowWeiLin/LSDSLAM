package LieAlgebra;

import java.util.Arrays;

/**
 *	Represent a three-dimensional Euclidean transformation (a rotation and a translation).
 *
 *	This transformation is a member of the Special Euclidean Lie group SE3.
 *  These can be parameterised six numbers (in the space of the Lie Algebra).
 *  In this class, the first three parameters are a translation vector while
 *  the second three are a rotation vector, whose direction is the axis of rotation
 *  and length the amount of rotation (in radians), as for SO3
 *
 * 
 *  Translated from: http://www.edwardrosten.com/cvd/toon/html-user/se3_8h_source.html
 *  
 */
public class SE3 {

	// Rotation
	SO3 rotation;
	// Translation, 3x1 vector
	jeigen.DenseMatrix translation;
	
	/**
	 * Default constructor, initializes rotation to 0(identity) and translation to 0.
	 */
	public SE3() {
		rotation = new SO3();
		setTranslation(0,0,0);
	}
	
	public SE3(SO3 rotation, jeigen.DenseMatrix translation) {
		this.rotation = rotation;
		this.translation = translation;
	}
	
	public SO3 getRotation() {
		return rotation;
	}
	
	public jeigen.DenseMatrix getRotationMat() {
		return rotation.matrix;
	}
	
	public jeigen.DenseMatrix getTranslationMat() {
		return translation;
	}
	
	public double[] getTranslation() {
		return translation.getValues();
	}
	
	
	/**
	 * Exponentiate a Vector in the Lie Algebra to generate a new SE3.
	 * @param vec6 assumes size 6
	 * @return
	 */
	public static SE3 exp(double[] vec6) {
		final double one_6th = 1.0/6.0;
		final double one_20th = 1.0/20.0;
		
		SE3 result = new SE3();
		
		final double[] t = {vec6[0], vec6[1], vec6[2]}; // vec3, parameters for translation.
		final double[] w = {vec6[3], vec6[4], vec6[5]}; // vec3, parameters for rotation.
		final double theta_sq = dot(w, w);
		final double theta = Math.sqrt(theta_sq);
		double A;
		double B;
		
		final double[] cross = cross(w, t); // cross of rotation, translation.
	    if (theta_sq < 1e-12) {
	        A = 1.0 - one_6th * theta_sq;
	        B = 0.5;
	        result.setTranslation(vecAdd2(t, scalarMult2(cross, 0.5)));
	    } else {
	        double C;
	        if (theta_sq < 1e-12) {
	            C = one_6th*(1.0 - one_20th * theta_sq);
	            A = 1.0 - theta_sq * C;
	            B = 0.5 - 0.25 * one_6th * theta_sq;
	        } else {
	            final double inv_theta = 1.0/theta;
	            A = Math.sin(theta) * inv_theta;
	            B = (1.0 - Math.cos(theta)) * (inv_theta * inv_theta);
	            C = (1.0 - A) * (inv_theta * inv_theta);
	        }
	        result.setTranslation(vecAdd2(vecAdd2(t, scalarMult2(cross, B)), scalarMult2(cross(w, cross), C)));
	    }

	    result.rotation.matrix = SO3.rodrigues_so3_exp(w, A, B);
	    return result;
	}

	// Test
	public static void main(String[] args) {
		
		double[] vec6 = new double[]{1,1,1,0.5,0,0};//{0.9257751770440343, 0.6696269810349835, 0.7949576278265412, 2.2639318115967644, 0.45278636231935293, 0.45278636231935293};//{1,1,1,0.5,0.1,0.1};
		SE3 se3 = SE3.exp(vec6);
		
		for (int i=0 ; i<1000 ; i++) {
			//System.out.println("rot mat = "+se3.rotation.matrix);
			//System.out.println("trans mat = "+se3.getTranslationMat());
			System.out.println("=" +Arrays.toString(SE3.ln(se3)));
			
			vec6 = SE3.ln(se3);
			se3 = SE3.exp(vec6);
			
		}
	}
	
	static double[] ln(SE3 se3) {
		double[] rot = se3.getRotation().ln();
		
		
		final double theta = Math.sqrt(dot(rot, rot));
		

		double shtot = 0.5;
		if(theta > 1e-12) {
		    shtot = Math.sin(theta/2.0)/theta;
		}

		// now do the rotation
		final SO3 halfrotator = new SO3(SO3.exp(scalarMult2(rot, -0.5)));
		double[] rottrans = halfrotator.matrix.mmul(se3.getTranslationMat()).getValues();

		if(theta > 1e-12){
		    vecMinus(rottrans, scalarMult2(rot, (dot(se3.getTranslation(), rot) * (1.0-2.0*shtot) / dot(rot, rot))));
		} else {
		    vecMinus(rottrans, scalarMult2(rot, (dot(se3.getTranslation(), rot)/24.0)));
		}

		scalarMult(rottrans,1.0/(2.0 * shtot));

		double[] result = {rottrans[0],rottrans[1],rottrans[2],rot[0],rot[1],rot[2]};
		return result;
		
	}
	
	
	
	
	
	
	
	public void setTranslation(double[] vec3) {
		double[][] translationVec3 = {{vec3[0]},{vec3[1]},{vec3[2]}};
		translation = new jeigen.DenseMatrix(translationVec3);
	}
	
	public void setTranslation(double x, double y, double z) {
		double[][] translationVec3 = {{x},{y},{z}};
		translation = new jeigen.DenseMatrix(translationVec3);
	}
	
	
	
	
	
	

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
}