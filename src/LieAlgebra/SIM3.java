package LieAlgebra;

import java.util.Arrays;

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

	public SE3 se3;
	public double scale;
	
	/**
	 * Default constructor, zero translation/rotation, scale 1.
	 */
	public SIM3() {
		se3 = new SE3();
		scale = 1;
		assertNotNaN();
	}
	
	public SIM3(SIM3 sim3) {
		
		this.se3 = new SE3(sim3.se3);
		this.scale = sim3.scale;
		assertNotNaN();
	}

	public SIM3(SE3 se3, double scale) {
		this.se3 = se3;
		this.scale = scale;
		assertNotNaN();
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
				this.getRotationMat().mmul(sim3.getTranslationMat().mul(getScale())));
		newSim3.se3.rotation.mulEq(sim3.getRotation());
		newSim3.scale *= sim3.getScale();
		
		newSim3.assertNotNaN();
		
		return newSim3;
	}
	
	public DenseMatrix mul(DenseMatrix point) {
		return new DenseMatrix(this.getTranslationMat().add(
				this.getRotationMat().mmul(
						point.mul(this.getScale()))));
	}
	
	public static SIM3 inverse(SIM3 sim3) {
		SIM3 inverse = new SIM3(SE3.inverse(sim3.se3), 1./sim3.scale);
		inverse.assertNotNaN();
		return inverse;
	}
	
	public SIM3 inverse() {
		return SIM3.inverse(this);
	}

	public double[] ln() {
		return SIM3.ln(this);
	}
	
	public static double[] ln(SIM3 sim3) {
		
		double[] result = new double[7];
	    
	    // rotation
		double[] rotResult = sim3.getRotation().ln();
	    double theta = Vec.magnitude(rotResult);

	    // scale 
	    double s = Math.log(sim3.getScale());
	    result[6] = s;

	    // Translation
	    double[] coeff = compute_rodrigues_coefficients_sim3(s, theta);
	    DenseMatrix cross = cross_product_matrix(rotResult);
	    DenseMatrix W = (DenseMatrix.eye(3).mul(coeff[0])).add(
	    				(cross.mul(coeff[1]))).add(
	    				(cross.mmul(cross).mul(coeff[2])));
	    
	    DenseMatrix transResultMat = W.fullPivHouseholderQRSolve(sim3.getTranslationMat());
	    
	    result[0] = transResultMat.get(0, 0);
	    result[1] = transResultMat.get(1, 0);
	    result[2] = transResultMat.get(2, 0);
	    result[3] = rotResult[0];
	    result[4] = rotResult[1];
	    result[5] = rotResult[2];
	    
	    return result;
	}
	

	/**
	 * Exponentiate a Vector in the Lie Algebra to generate a new SIM3.
	 * @param vec7 assumes size 7
	 * @return
	 */
	public static SIM3 exp(double[] vec7) {
		
		double[] transVec = new double[]{vec7[0],vec7[1],vec7[2]};
		double[] rotVec = new double[]{vec7[3],vec7[4],vec7[5]};
		
		// Scale
		double scale = Math.exp(vec7[6]);
		
		// Rotation
		SO3 rotation = new SO3(SO3.exp(rotVec));
		double t = Vec.magnitude(rotVec);

		// Translation
		double[] coeff = compute_rodrigues_coefficients_sim3(vec7[6], t); 
		double[] cross = Vec.cross(rotVec, transVec);
		
		// vec3
		double[] trans = Vec.vecAdd2(Vec.vecAdd2(Vec.scalarMult2(transVec, coeff[0]),
										   			   Vec.scalarMult2(cross, coeff[1])),
										   Vec.scalarMult2(Vec.cross(rotVec, cross), coeff[2]));
		
		SE3 se3 = new SE3();
		se3.setTranslation(trans);
		se3.rotation = rotation;
		
		SIM3 result = new SIM3(se3, scale);
		
		result.assertNotNaN();
		return result;
	}
	
	/// creates an returns a cross product matrix M from a 3 vector v, such that for all vectors w, the following holds: v ^ w = M * w
	/// @param vec the 3 vector input
	/// @return the 3x3 matrix to set to the cross product matrix
	static DenseMatrix cross_product_matrix(double[] vec)
	{
		DenseMatrix result = new DenseMatrix(new double[][]{
				{0, -vec[2], vec[1]},
				{vec[2], 0, -vec[0]},
				{-vec[1], vec[0], 0}
		});
		
	    return result;
	}
	
	static double[] compute_rodrigues_coefficients_sim3(double s, double t){

	    double[] coeff = new double[3];
	    double es = Math.exp(s);

	    // 4 cases for s -> 0 and/or theta -> 0
	    // the Taylor expansions were calculated with Maple 12 and truncated at the 3rd power,
	    // such that eps^3 < 1e-18 which results in approximately 1 + eps^3 = 1
	    double eps = 1e-6;

	    if(Math.abs(s) < eps && Math.abs(t) < eps){
	        coeff[0] = 1 + s/2 + s*s/6;
	        coeff[1] = 1/2 + s/3 - t*t/24 + s*s/8;
	        coeff[2] = 1/6 + s/8 - t*t/120 + s*s/20;
	    } else if(Math.abs(s) < eps) {
	        coeff[0] = 1 + s/2 + s*s/6;
	        coeff[1] = (1-Math.cos(t))/(t*t) + (Math.sin(t)-Math.cos(t)*t)*s/(t*t*t)+(2*Math.sin(t)*t-t*t*Math.cos(t)-2+2*Math.cos(t))*s*s/(2*t*t*t*t);
	        coeff[2] = (t-Math.sin(t))/(t*t*t) - (-t*t-2+2*Math.cos(t)+2*Math.sin(t)*t)*s/(2*t*t*t*t) - (-t*t*t+6*Math.cos(t)*t+3*Math.sin(t)*t*t-6*Math.sin(t))*s*s/(6*t*t*t*t*t);
	    } else if(Math.abs(t) < eps) {
	        coeff[0] = (es - 1)/s;
	        coeff[1] = (s*es+1-es)/(s*s) - (6*s*es+6-6*es+es*s*s*s-3*es*s*s)*t*t/(6*s*s*s*s);
	        coeff[2] = (es*s*s-2*s*es+2*es-2)/(2*s*s*s) - (es*s*s*s*s-4*es*s*s*s+12*es*s*s-24*s*es+24*es-24)*t*t/(24*s*s*s*s*s);
	    } else {
	        double a = es * Math.sin(t);
	        double b = es * Math.cos(t);
	        double inv_s_theta = 1/(s*s + t*t);

	        coeff[0] = (es - 1)/s;
	        coeff[1] = (a*s + (1-b)*t) * inv_s_theta / t;
	        coeff[2] = (coeff[0] - ((b-1)*s + a*t) * inv_s_theta) / (t*t);
	    }

	    return coeff;
	}
	
	
	public DenseMatrix adjointVec(DenseMatrix vect7) {
		//	   Vector<7, Precision> result;
		//     result.template slice<3,3>() = get_rotation() * vect.template slice<3,3>();
		//     result.template slice<0,3>() = get_rotation() * vect.template slice<0,3>();
		//     result.template slice<0,3>() += get_translation() ^ result.template slice<3,3>();
		//     return result;
	    
		DenseMatrix vect33 = vect7.slice(3, 6, 0, 1);
		DenseMatrix vect03 = vect7.slice(0, 3, 0, 1);
		
	    DenseMatrix rotMulVect33 = getRotationMat().mmul(vect33);
	    DenseMatrix rotMulVect03 = getRotationMat().mmul(vect03);
	    
		DenseMatrix trans_rotMulVect33 = Vec.cross(getTranslationMat(), rotMulVect33);
		
	    DenseMatrix result = new DenseMatrix(new double[][]{
	    		{rotMulVect03.get(0, 0)},{rotMulVect03.get(1, 0)},{rotMulVect03.get(2, 0)},
	    		{rotMulVect33.get(0, 0) + trans_rotMulVect33.get(0, 0)},
	    		{rotMulVect33.get(1, 0) + trans_rotMulVect33.get(1, 0)},
	    		{rotMulVect33.get(2, 0) + trans_rotMulVect33.get(2, 0)},{0}}); 
		
		return result;
	}
	
	/*public DenseMatrix adjoint() {
		
//		 Matrix<7,7,Precision> result;
//		 for(int i=0; i<7; i++){
//		 	result.T()[i] = adjoint(M.T()[i]);
//		 }
//		 for(int i=0; i<7; i++){
//			result[i] = adjoint(result[i]);
//		 }
//		 return result;

		for(int i=0; i<7; i++){
//		 	result.T()[i] = adjoint(M.T()[i]);
//		 }
		
		
		return null;
	}*/
	
	// so3 hat
	DenseMatrix hat(DenseMatrix a) {
	    DenseMatrix A = new DenseMatrix(
	    	new double[][] {
	    			{0           ,-a.get(2, 0) ,a.get(1, 0) },
	    			{-a.get(2, 0), 0           , -a.get(0, 0)},
	    			{-a.get(1, 0), a.get(0, 0) , 0           }
	    	});
	    return A;
	  }
	
	public DenseMatrix adjoint() {
		
		DenseMatrix result = null;
		DenseMatrix rot = getRotationMat();
		DenseMatrix trans = getTranslationMat();
		DenseMatrix hatTrans = hat(trans);
		
		DenseMatrix scaleMulR = rot.mul(scale);
		DenseMatrix unitTransMulR = hatTrans.mmul(rot);
		
		
		result = new DenseMatrix(new double[][]{
			{scaleMulR.get(0, 0),scaleMulR.get(0, 1),scaleMulR.get(0, 2),unitTransMulR.get(0, 0),unitTransMulR.get(0, 1),unitTransMulR.get(0, 2),-trans.get(0, 0)},
			{scaleMulR.get(1, 0),scaleMulR.get(1, 1),scaleMulR.get(1, 2),unitTransMulR.get(1, 0),unitTransMulR.get(1, 1),unitTransMulR.get(1, 2),-trans.get(1, 0)},
			{scaleMulR.get(2, 0),scaleMulR.get(2, 1),scaleMulR.get(2, 2),unitTransMulR.get(2, 0),unitTransMulR.get(2, 1),unitTransMulR.get(2, 2),-trans.get(2, 0)},
			{0,0,0,rot.get(0, 0),rot.get(0, 1),rot.get(0, 2),0},
			{0,0,0,rot.get(1, 0),rot.get(1, 1),rot.get(1, 2),0},
			{0,0,0,rot.get(2, 0),rot.get(2, 1),rot.get(2, 2),0},
			{0,0,0,0,0,0,1},
		});
		
		
		return result;
	}
	
	public String toString() {
		return Arrays.toString(SE3.ln(this.se3)) + "["+this.scale+"]";
	}
	
	public void assertNotNaN() {
		assert(!Double.isNaN(scale));
		assert(!Double.isInfinite(scale));
		se3.assertNotNaN();
	}
	
	public static void main(String[] args) {
		
		SIM3 FtoC = SIM3.exp(new double[]{0.0656, -0.1039, 0.0033, -0.0308, 0.0010, 0.0184, 0.0180}).inverse();
		//SIM3 FtoC = SIM3.exp(new double[]{1,1,1,1,1,1,1}).inverse();
		//SIM3 FtoC = new SIM3(new SE3(), 1);
		
		
		
		System.out.println(Arrays.toString(SIM3.ln(FtoC)));
		
		//FtoC = FtoC.inverse();
		//System.out.println(Arrays.toString(SIM3.ln(FtoC)));
		
		for (int i=0 ; i<100 ; i++) {
			FtoC = SIM3.exp(FtoC.ln());
			System.out.println(Arrays.toString(SIM3.ln(FtoC)));
		}
	}
	
}
