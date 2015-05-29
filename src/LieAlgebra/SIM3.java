package LieAlgebra;

/**
 * Represent a three-dimensional similarity transformation (a rotation, a scale factor and a translation).
 *  
 * This transformation is a member of the Lie group SIM3.
 * These can be parameterised with seven numbers (in the space of the Lie Algebra).
 * In this class, the first three parameters are a translation vector while the
 * second three are a rotation vector, whose direction is the axis of rotation
 * and length the amount of rotation (in radians), as for SO3.
 * The seventh parameter is the log of the scale of the transformation.
 * 
 * Translated from: http://www.edwardrosten.com/cvd/toon/html-user/sim3_8h_source.html
 * 
 */
public class SIM3 {

	SE3 se3;
	double scale;
	
	/**
	 * Default constructor, zero translation/rotation, scale 1.
	 */
	SIM3() {
		se3 = new SE3();
		scale = 1;
	}
	
	
	
	
}
