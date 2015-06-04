package LieAlgebra;

import jeigen.DenseMatrix;

/**
 * Class to represent a three-dimensional rotation matrix.
 * 
 * Three-dimensional rotation matrices are members of the Special Orthogonal Lie group SO3.
 * This group can be parameterised three numbers (a vector in the space of the Lie Algebra).
 * In this class, the three parameters are the finite rotation vector,
 * i.e. a three-dimensional vector whose direction is the axis of rotation and whose length is the angle of rotation in radians.
 * Exponentiating this vector gives the matrix, and the logarithm of the matrix gives this vector.
 * 
 * 
 * 
 * Translated from: http://www.edwardrosten.com/cvd/toon/html-user/so3_8h_source.html
 * 
 */

public class SO3 {

	// Constants
	public static final double M_SQRT1_2 = Math.sqrt(0.5);
	public static final double M_PI = Math.PI;
	
	
	// 3x3 matrix representing the rotation.
	public jeigen.DenseMatrix matrix;
	
	
	public SO3() {
		matrix = jeigen.DenseMatrix.eye(3);
	}
	
	public SO3(jeigen.DenseMatrix mat33) {
		set33(mat33);
	}
	
	// Construct from 3x1 vector, representing axis of rotation, and magnitude representing angle.
	public void set31(double[] vec3) {
		set33(exp(vec3));
	}
	
	// Sets 3x3 rotation matrix
	public void set33(jeigen.DenseMatrix mat33) {
		this.matrix = mat33;
	}

	/**
	 * Performs exponential, returns SO3 matrix.
	 * @param vec3 assumed to be size 3
	 * @return
	 */
	public static jeigen.DenseMatrix exp(double[] vec3) {

		final double one_6th = 1.0/6.0;
		final double one_20th = 1.0/20.0;
				
		jeigen.DenseMatrix result = null;
		
		double theta_sq = Vec.dot(vec3, vec3);
		double theta = Math.sqrt(theta_sq);
		double A;
		double B;
		
		//Use a Taylor series expansion near zero. This is required for
		//accuracy, since sin t / t and (1-cos t)/t^2 are both 0/0.
		if (theta_sq < 1e-12) {
		    A = 1.0 - one_6th * theta_sq;
		    B = 0.5;
		} else {
		    if (theta_sq < 1e-12) {
		        B = 0.5 - 0.25 * one_6th * theta_sq;
		        A = 1.0 - theta_sq * one_6th*(1.0 - one_20th * theta_sq);
		    } else {
		        final double inv_theta = 1.0/theta;
		        A = Math.sin(theta) * inv_theta;
		        B = (1.0 - Math.cos(theta)) * (inv_theta * inv_theta);
		    }
		}
		
		result = rodrigues_so3_exp(vec3, A, B);
		
		return result;
	}
	
	/**
	 * Compute a rotation exponential using the Rodrigues Formula.
	 * 
	 * @param vec3 assumed to be size 3
	 * @param A
	 * @param B
	 * @param result Matrix 
	 * @return 
	 */
	public static DenseMatrix rodrigues_so3_exp(double[] w, double A, double B) {
		double[][] R = new double[3][3];
		{
		    final double wx2 = (double)w[0]*w[0];
		    final double wy2 = (double)w[1]*w[1];
		    final double wz2 = (double)w[2]*w[2];
		    R[0][0] = 1.0 - B*(wy2 + wz2);
		    R[1][1] = 1.0 - B*(wx2 + wz2);
		    R[2][2] = 1.0 - B*(wx2 + wy2);
		}
		{
			final double a = A*w[2];
			final double b = B*(w[0]*w[1]);
			R[0][1] = b - a;
			R[1][0] = b + a;
		}
		{
			final double a = A*w[1];
			final double b = B*(w[0]*w[2]);
			R[0][2] = b + a;
			R[2][0] = b - a;
		}
		{
			final double a = A*w[0];
			final double b = B*(w[1]*w[2]);
			R[1][2] = b - a;
			R[2][1] = b + a;
		}
		return new jeigen.DenseMatrix(R);
	}
	
	/**
	 * Take the logarithm of the matrix, generating the corresponding vector in the Lie Algebra.
	 * @return
	 */
	public double[] ln() {
		// 3x1 vector
		double[] result = new double[3];
		
		double[][] my_matrix = Vec.matToArray(matrix);
		
		final double cos_angle = (my_matrix[0][0] + my_matrix[1][1] + my_matrix[2][2] - 1.0) * 0.5;
		result[0] = (my_matrix[2][1]-my_matrix[1][2])/2.0;
		result[1] = (my_matrix[0][2]-my_matrix[2][0])/2.0;
		result[2] = (my_matrix[1][0]-my_matrix[0][1])/2.0;
		
		double sin_angle_abs = Math.sqrt(Vec.dot(result,result));
		if (cos_angle > M_SQRT1_2) {            // [0 - Pi/4[ use asin
		    if(sin_angle_abs > 0){
		    	final double s = Math.asin(sin_angle_abs) / sin_angle_abs;
		    	Vec.scalarMult(result, s);
		    }
		} else if( cos_angle > -M_SQRT1_2) {    // [Pi/4 - 3Pi/4[ use acos, but antisymmetric part
			final double angle_s = Math.acos(cos_angle) / sin_angle_abs;
			Vec.scalarMult(result, angle_s);
		} else {  // rest use symmetric part
		    // antisymmetric part vanishes, but still large rotation, need information from symmetric part
			final double angle = M_PI - Math.asin(sin_angle_abs);
			
			final double d0 = my_matrix[0][0] - cos_angle,
		        d1 = my_matrix[1][1] - cos_angle,
		        d2 = my_matrix[2][2] - cos_angle;
		    double[] r2 = new double[3];
		    if(d0*d0 > d1*d1 && d0*d0 > d2*d2){ // first is largest, fill with first column
		        r2[0] = d0;
		        r2[1] = (my_matrix[1][0]+my_matrix[0][1])/2.0;
		        r2[2] = (my_matrix[0][2]+my_matrix[2][0])/2.0;
		    } else if(d1*d1 > d2*d2) {              // second is largest, fill with second column
		        r2[0] = (my_matrix[1][0]+my_matrix[0][1])/2.0;
		        r2[1] = d1;
		        r2[2] = (my_matrix[2][1]+my_matrix[1][2])/2.0;
		    } else {                                // third is largest, fill with third column
		        r2[0] = (my_matrix[0][2]+my_matrix[2][0])/2.0;
		        r2[1] = (my_matrix[2][1]+my_matrix[1][2])/2.0;
		        r2[2] = d2;
		    }
		    // flip, if we point in the wrong direction!
		    if(Vec.dot(r2, result) < 0)
		    	Vec.scalarMult(r2, -1);

		    Vec.unit(r2);
		    Vec.scalarMult(r2,angle);
		    result = r2;
		}

		return result;
	}
	
	public static SO3 inverse(SO3 so3) {
		SO3 inverse = new SO3(so3.matrix.t());
		return inverse;
	}
	
	/**
	 * Right multiply by given SO3
	 */
	public void mulEq(SO3 so3) {
		this.matrix = this.matrix.mmul(so3.matrix);
	}
	
}
