package LieAlgebra;

import jeigen.DenseMatrix;

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
	public SIM3() {
		se3 = new SE3();
		scale = 1;
	}
	
	public SIM3(SIM3 sim3) {
		
		this.se3 = new SE3(sim3.se3);
		this.scale = sim3.scale;
		
	}

	public SIM3(SE3 se3, double scale) {
		this.se3 = se3;
		this.scale = scale;
	}

	public SE3 getSE3() {
		return se3;
	}
	
	public double getScale() {
		return scale;
	}
	
	public double[] getTranslation() {
		return se3.getTranslation();
	}
	
	public DenseMatrix getTranslationMat() {
		return se3.getTranslationMat();
	}
	
	public SO3 getRotation() {
		return se3.getRotation();
	}
	
	public DenseMatrix getRotationMat() {
		return se3.getRotationMat();
	}
	
	public SIM3 mul(SIM3 sim3) {
		
		SIM3 newSim3 = new SIM3(this);
				
//		getTranslation() += get_rotation() * (get_scale() * rhs.get_translation());
//		getrotation() *= rhs.get_rotation();
//		getscale() *= rhs.get_scale();
		
		newSim3.se3.translation = newSim3.se3.translation.add(
				this.getRotationMat().mul(sim3.getTranslationMat().mul(getScale())));
		newSim3.se3.rotation.mulEq(sim3.getRotation());
		newSim3.scale *= sim3.getScale();
		
		return this;
	}
	
	public static SIM3 inverse(SIM3 sim3) {
		SIM3 inverse = new SIM3(SE3.inverse(sim3.se3), sim3.scale);
		return inverse;
	}
	
	public SIM3 inverse() {
		return SIM3.inverse(this);
	}
}
