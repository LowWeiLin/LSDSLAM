package Utils;

import jeigen.DenseMatrix;

public class Constants {

	public static final float MAX_VAR = 0.5f * 0.5f; // initial variance on creation - if variance becomes larger than this, hypothesis is removed.
	public static final float VAR_RANDOM_INIT_INITIAL = 0.5f * MAX_VAR; // initial variance for Random Initialization
	
	public static final int PYRAMID_LEVELS = 5;
	public static final int SE3TRACKING_MAX_LEVEL = 5;
	public static final int SE3TRACKING_MIN_LEVEL = 0;
	
	public static final float MIN_GOODPERALL_PIXEL_ABSMIN = 0.01f;
	
	public static final float MAX_DIFF_CONSTANT = (40.0f*40.0f);
	public static final float MAX_DIFF_GRAD_MULT = (0.5f*0.5f);
	
	
	
	

	// Camera matrix K
	public static final jeigen.DenseMatrix[] K = new DenseMatrix[PYRAMID_LEVELS];
	public static final double[] fx = new double[PYRAMID_LEVELS];
	public static final double[] fy = new double[PYRAMID_LEVELS];
	public static final double[] cx = new double[PYRAMID_LEVELS];
	public static final double[] cy = new double[PYRAMID_LEVELS];
	// Inverse Camera matrix KInv
	public static final jeigen.DenseMatrix[] KInv = new DenseMatrix[PYRAMID_LEVELS];		
	public static final double[] fxInv = new double[PYRAMID_LEVELS];
	public static final double[] fyInv = new double[PYRAMID_LEVELS];
	public static final double[] cxInv = new double[PYRAMID_LEVELS];
	public static final double[] cyInv = new double[PYRAMID_LEVELS];
	
	/**
	 * Sets camera matrix for all pyramid levels.
	 * Pass in parameters for level 0.
	 * @param fx
	 * @param fy
	 * @param cx
	 * @param cy
	 */
	public static void setK(double fx, double fy, double cx, double cy) {
		Constants.fx[0] = fx;
		Constants.fy[0] = fy;
		Constants.cx[0] = cx;
		Constants.cy[0] = cy;
		
		K[0] = new DenseMatrix(new double[][]{{Constants.fx[0],0,Constants.cx[0]},
				{0,Constants.fy[0],Constants.cy[0]},
				{0,0,1}});
		KInv[0] = K[0].fullPivHouseholderQRSolve(DenseMatrix.eye(3));

		fxInv[0] = KInv[0].get(0, 0);
		fyInv[0] = KInv[0].get(1, 1);
		cxInv[0] = KInv[0].get(0, 2);
		cyInv[0] = KInv[0].get(1, 2);
		

		for (int level = 0; level < PYRAMID_LEVELS; level++)
		{
			if (level > 0)
			{
				// Get/calculate K parameters for lower levels.
				Constants.fx[level] = Constants.fx[level-1] * 0.5;
				Constants.fy[level] = Constants.fy[level-1] * 0.5;
				Constants.cx[level] = (Constants.cx[0] + 0.5) / ((int)1<<level) - 0.5;
				Constants.cy[level] = (Constants.cy[0] + 0.5) / ((int)1<<level) - 0.5;

				K[level] = new DenseMatrix(new double[][]{{Constants.fx[level],0,Constants.cx[level]},
						{0,Constants.fy[level],Constants.cy[level]},
						{0,0,1}});
				KInv[level] = K[level].fullPivHouseholderQRSolve(DenseMatrix.eye(3));

				fxInv[level] = KInv[level].get(0, 0);
				fyInv[level] = KInv[level].get(1, 1);
				cxInv[level] = KInv[level].get(0, 2);
				cyInv[level] = KInv[level].get(1, 2);
			}
		}
		
	}
	
	

}
