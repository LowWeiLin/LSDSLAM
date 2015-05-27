package Utils;

import jeigen.DenseMatrix;

public class Constants {

	public static final float MAX_VAR = 0.5f * 0.5f; // initial variance on creation - if variance becomes larger than this, hypothesis is removed.
	public static final float VAR_RANDOM_INIT_INITIAL = 0.5f * MAX_VAR; // initial variance for Random Initialization
	
	
	// TODO: Random test values. To change later.
	public static final double fx = 500;
	public static final double fy = 500;
	public static final double cx = 350;//640/2;
	public static final double cy = 240;//480/2;

	// Camera matrix K
	public static final jeigen.DenseMatrix K = new DenseMatrix(new double[][]{{fx,0,cx},{0,fy,cy},{0,0,1}});

	// Inverse Camera matrix KInv
	public static final jeigen.DenseMatrix KInv = K.fullPivHouseholderQRSolve(DenseMatrix.eye(3));
	
	public static final double fxInv = KInv.get(0, 0);
	public static final double fyInv = KInv.get(1, 1);
	public static final double cxInv = KInv.get(0, 2);
	public static final double cyInv = KInv.get(1, 2);
	

}
