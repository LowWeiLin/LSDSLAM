package Utils;

import jeigen.DenseMatrix;

public class Constants {
	
	public static final float DIVISION_EPS  = 1e-10f;

	public static final float MAX_VAR = 0.5f * 0.5f; // initial variance on creation - if variance becomes larger than this, hypothesis is removed.
	public static final float VAR_RANDOM_INIT_INITIAL = 0.5f * MAX_VAR; // initial variance for Random Initialization
	
	public static final int PYRAMID_LEVELS = 5;
	public static final int SE3TRACKING_MAX_LEVEL = 5;
	public static final int SE3TRACKING_MIN_LEVEL = 0;
	
	public static final float MIN_USE_GRAD = 5; // TODO: setting, not constant.
	public static final float MIN_ABS_GRAD_CREATE = MIN_USE_GRAD;
	public static final float MIN_ABS_GRAD_DECREASE = MIN_USE_GRAD;
	
	
	/** ============== constants for validity handling ======================= */

	// validity can take values between 0 and X, where X depends on the abs. gradient at that location:
	// it is calculated as VALIDITY_COUNTER_MAX + (absGrad/255)*VALIDITY_COUNTER_MAX_VARIABLE
	public static final float VALIDITY_COUNTER_MAX = (5.0f); // validity will never be higher than this
	public static final float VALIDITY_COUNTER_MAX_VARIABLE = (250.0f); // validity will never be higher than this

	public static final int VALIDITY_COUNTER_INC = 5;		// validity is increased by this on sucessfull stereo
	public static final int VALIDITY_COUNTER_DEC = 5;		// validity is decreased by this on failed stereo
	public static final int VALIDITY_COUNTER_INITIAL_OBSERVE = 5;	// initial validity for first observations

	public static final int VAL_SUM_MIN_FOR_CREATE = 30; // minimal summed validity over 5x5 region to create a new hypothesis for non-blacklisted pixel (hole-filling)
	public static final int VAL_SUM_MIN_FOR_KEEP = 24; // minimal summed validity over 5x5 region to keep hypothesis (regularization)
	public static final int VAL_SUM_MIN_FOR_UNBLACKLIST = 100; // if summed validity surpasses this, a pixel is un-blacklisted.

	public static final int MIN_BLACKLIST = -1;	// if blacklist is SMALLER than this, pixel gets ignored. blacklist starts with 0.

	
	
	// ============== initial stereo pixel selection ======================
	public static final float MIN_EPL_GRAD_SQUARED = (2.0f*2.0f);
	public static final float MIN_EPL_LENGTH_SQUARED = (1.0f*1.0f);
	public static final float MIN_EPL_ANGLE_SQUARED = (0.3f*0.3f);
	
	
	// ============== stereo & gradient calculation ======================
	public static final float MIN_DEPTH = 0.05f; // this is the minimal depth tested for stereo.

	// Particularly important for initial pixel.
	public static final float MAX_EPL_LENGTH_CROP = 30.0f; // maximum length of epl to search.
	public static final float MIN_EPL_LENGTH_CROP = 3.0f; // minimum length of epl to search.

	// this is the distance of the sample points used for the stereo descriptor.
	public static final float GRADIENT_SAMPLE_DIST = 1.0f;

	// pixel a point needs to be away from border
	public static final int SAMPLE_POINT_TO_BORDER = 7;

	// pixels with too big an error are definitely thrown out.
	public static final float MAX_ERROR_STEREO = 1300.0f; // maximal photometric error for stereo to be successful (sum over 5 squared intensity differences)
	public static final float MIN_DISTANCE_ERROR_STEREO = 1.5f; // minimal multiplicative difference to second-best match to not be considered ambiguous.

	// defines how large the stereo-search region is. it is [mean] +/- [std.dev]*STEREO_EPL_VAR_FAC
	public static final float STEREO_EPL_VAR_FAC = 2.0f;
	

	// ============== RE-LOCALIZATION, KF-REACTIVATION etc. ======================
	// defines the level on which we do the quick tracking-check for relocalization.

	public static final float MAX_DIFF_CONSTANT = (40.0f*40.0f);
	public static final float MAX_DIFF_GRAD_MULT = (0.5f*0.5f);

	public static final float MIN_GOODPERGOODBAD_PIXEL = (0.5f);
	public static final float MIN_GOODPERALL_PIXEL = (0.04f);
	public static final float MIN_GOODPERALL_PIXEL_ABSMIN = (0.01f);

	public static final int INITIALIZATION_PHASE_COUNT = 5;

	public static final int MIN_NUM_MAPPED = 5;
	
	// ============== Smoothing and regularization ======================
	// distance factor for regularization.
	// is used as assumed inverse depth variance between neighbouring pixel.
	// basically determines the amount of spacial smoothing (small -> more smoothing).
	static float depthSmoothingFactor = 1;
	public static final float REG_DIST_VAR = (0.075f*0.075f*depthSmoothingFactor*depthSmoothingFactor);

	// define how strict the merge-processes etc. are.
	// are multiplied onto the difference, so the larger, the more restrictive.
	public static final float DIFF_FAC_SMOOTHING = (1.0f*1.0f);
	public static final float DIFF_FAC_OBSERVE = (1.0f*1.0f);
	public static final float DIFF_FAC_PROP_MERGE = (1.0f*1.0f);
	public static final float DIFF_FAC_INCONSISTENT = (1.0f * 1.0f);
	
	
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
