package LieAlgebra;

import java.util.Arrays;

import jeigen.DenseMatrix;

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
	public SO3 rotation;
	// Translation, 3x1 vector
	public jeigen.DenseMatrix translation;
	
	/**
	 * Default constructor, initializes rotation to 0(identity) and translation to 0.
	 */
	public SE3() {
		rotation = new SO3();
		setTranslation(0,0,0);
		assertNotNaN();
	}
	
	public SE3(SO3 rotation, jeigen.DenseMatrix translation) {
		this.rotation = rotation;
		this.translation = translation;
		assertNotNaN();
	}
	
	public SE3(SE3 se3) {
		this.translation = new DenseMatrix(se3.translation);
		this.rotation = new SO3(se3.rotation);
		assertNotNaN();
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
		final double theta_sq = Vec.dot(w, w);
		final double theta = Math.sqrt(theta_sq);
		double A;
		double B;
		
		final double[] cross = Vec.cross(w, t); // cross of rotation, translation.
	    if (theta_sq < 1e-12) {
	        A = 1.0 - one_6th * theta_sq;
	        B = 0.5;
	        result.setTranslation(Vec.vecAdd2(t, Vec.scalarMult2(cross, 0.5)));
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
	        result.setTranslation(Vec.vecAdd2(Vec.vecAdd2(t, Vec.scalarMult2(cross, B)), Vec.scalarMult2(Vec.cross(w, cross), C)));
	    }

	    result.rotation.matrix = SO3.rodrigues_so3_exp(w, A, B);
	    return result;
	}

	public static double[] ln(SE3 se3) {
		double[] rot = se3.getRotation().ln();
		
		
		final double theta = Math.sqrt(Vec.dot(rot, rot));
		

		double shtot = 0.5;
		if(theta > 1e-12) {
		    shtot = Math.sin(theta/2.0)/theta;
		}

		// now do the rotation
		final SO3 halfrotator = new SO3(SO3.exp(Vec.scalarMult2(rot, -0.5)));
		double[] rottrans = halfrotator.matrix.mmul(se3.getTranslationMat()).getValues();

		if(theta > 1e-12){
			Vec.vecMinus(rottrans, Vec.scalarMult2(rot, (Vec.dot(se3.getTranslation(), rot) * (1.0-2.0*shtot) / Vec.dot(rot, rot))));
		} else {
			Vec.vecMinus(rottrans, Vec.scalarMult2(rot, (Vec.dot(se3.getTranslation(), rot)/24.0)));
		}

		Vec.scalarMult(rottrans,1.0/(2.0 * shtot));

		double[] result = {rottrans[0],rottrans[1],rottrans[2],rot[0],rot[1],rot[2]};
		
		// Assert ln does not return any NaNs
		for(double d : result) {
			assert(!Double.isNaN(d));
		}
		
		return result;
		
	}

	public double[] ln() {
		return SE3.ln(this);
	}
	
	public static SE3 inverse(SE3 se3) {
		//SE3 inverse = new SE3(SO3.inverse(se3.getRotation()), se3.translation.mul(-1));
		
		SE3 inverse = new SE3();
		inverse.rotation = SO3.inverse(se3.rotation);
		inverse.translation = (inverse.getRotationMat().mmul(se3.translation)).mul(-1);
		
		inverse.assertNotNaN();
		return inverse;
	}
	
	public SE3 inverse() {
		return SE3.inverse(this);
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
	 * Right multiply by given SE3
	 */
	public void mulEq(SE3 se3) {
		this.translation = this.translation.add(rotation.matrix.mmul(se3.translation));
		this.rotation.matrix = this.rotation.matrix.mmul(se3.rotation.matrix);
		assertNotNaN();
	}
	
	/**
	 * Right multiply by given SE3
	 */
	public SE3 mul(SE3 se3) {
		DenseMatrix translation = this.translation.add(rotation.matrix.mmul(se3.translation));
		DenseMatrix rotation = this.rotation.matrix.mmul(se3.rotation.matrix);
		return new SE3(new SO3(rotation), translation);
	}
	

	public void assertNotNaN() {
		assert(!Double.isNaN(this.translation.get(0, 0)));
		assert(!Double.isNaN(this.translation.get(1, 0)));
		assert(!Double.isNaN(this.translation.get(2, 0)));
		rotation.assertNotNaN();
	}

	// Test
	public static void main(String[] args) {
		
		double[] vec6 = new double[]{1,1,1,0.53,0,0};//{0.9257751770440343, 0.6696269810349835, 0.7949576278265412, 2.2639318115967644, 0.45278636231935293, 0.45278636231935293};//{1,1,1,0.5,0.1,0.1};
		SE3 se3 = SE3.exp(vec6);

//		// Test for ln/exp number drift
//		for (int i=0 ; i<1000000 ; i++) {
//			//System.out.println("rot mat = "+se3.rotation.matrix);
//			//System.out.println("trans mat = "+se3.getTranslationMat());
//			System.out.println("vec = " +Arrays.toString(SE3.ln(se3)));
//			
//			vec6 = SE3.ln(se3);
//			se3 = SE3.exp(vec6);
//			
//		}

		// Test for inverse number drift
		vec6 = new double[]{-0.1687722544216055, -0.11161075918408007, -0.01406770893311903, -0.2680809945869649, 0.14390299341810078, 0.10327471107537695};//{1,1,1,0.5,0.1,0.1};
		se3 = SE3.exp(vec6);
		
		// Inverse test
		System.out.println("vec = " +Arrays.toString(SE3.ln(se3.mul(se3.inverse()))));
		
		
		System.out.println("vec = " +Arrays.toString(SE3.ln(se3)));
		for (int i=0 ; i<100 ; i++) {
			//System.out.println("rot mat = "+se3.rotation.matrix);
			//System.out.println("trans mat = "+se3.getTranslationMat());
			
			se3 = se3.inverse();
			se3 = se3.inverse();
			
			System.out.println("vec = " +Arrays.toString(SE3.ln(se3)));
			
			
			vec6 = SE3.ln(se3);
			se3 = SE3.exp(vec6);
			
		}
		
	}
}
