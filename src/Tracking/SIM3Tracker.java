package Tracking;

import java.util.Arrays;

import DataStructures.Frame;
import DataStructures.TrackingReference;
import LieAlgebra.SIM3;
import LieAlgebra.Vec;
import Utils.Constants;
import jeigen.DenseMatrix;


class Sim3ResidualStruct
{
	float sumResD;
	float sumResP;
	int numTermsD;
	int numTermsP;

	float meanD;
	float meanP;
	float mean;
	
	Sim3ResidualStruct()
	{
		meanD = 0;
		meanP = 0;
		mean = 0;
		numTermsD = numTermsP = (int) (sumResD = sumResP = 0);
	}
};

public class SIM3Tracker {
	
	// Settings variables
	public static final int maxIterations[] = {5, 20, 50, 100, 100, 100};
	public static final int maxItsPerLvl[] = {5, 20, 50, 100, 100, 100};
	public static float[] lambdaInitial = new float[Constants.PYRAMID_LEVELS];
	public static float[] convergenceEps = new float[Constants.PYRAMID_LEVELS];
	public static final float varWeight = 1.0f;
	public static final float huberD = 3.0f;
	public static final float cameraPixelNoise2 = 4*4;
	
	public static final float lambdaSuccessFac = 0.5f;
	public static final float lambdaFailFac = 2.0f;
	public static final float stepSizeMin[] = {1e-8f, 1e-8f, 1e-8f, 1e-8f, 1e-8f, 1e-8f};
	
	

	public static final float lambdaInitialTestTrack = 0f;
	public static final float stepSizeMinTestTrack = 1e-3f;
	public static final float convergenceEpsTestTrack = 0.98f;
	public static final int maxItsTestTrack = 5;

	
	//
	public static float var_weight = 1.0f;
	public static float huber_d = 3f;
	
	
	//
	
	int buf_warped_size = 0;
	
	float[] buf_warped_residual;
	float[] buf_warped_weights;
	float[] buf_warped_dx;
	float[] buf_warped_dy;
	float[] buf_warped_x;
	float[] buf_warped_y;
	float[] buf_warped_z;

	float[] buf_d;
	float[] buf_residual_d;
	float[] buf_idepthVar;
	float[] buf_warped_idepthVar;
	float[] buf_weight_p;
	float[] buf_weight_d;

	
	//
	
	public DenseMatrix lastSim3Hessian;// 7x7
	
	public float pointUsage;
	public float lastResidual, lastDepthResidual, lastPhotometricResidual;
	public float lastResidualUnweighted, lastDepthResidualUnweighted, lastPhotometricResidualUnweighted;


	public boolean diverged;
	
	// used for image saving
	public int iterationNumber;
	
	
	private float affineEstimation_a = 1;
	private float affineEstimation_b = 0;
	
	private boolean initialized = false;
	
	public SIM3Tracker() {

		lastResidual = 0;
		iterationNumber = 0;
		lastDepthResidual = lastPhotometricResidual = lastDepthResidualUnweighted = lastPhotometricResidualUnweighted = lastResidualUnweighted = 0;
		pointUsage = 0;
		
//
//		// **
//		// Do not use more than 4 levels for odometry tracking
//		for (int level = 4; level < Constants.PYRAMID_LEVELS; ++level)
//			maxItsPerLvl[level] = 0;
		
	}
	
	public void initialize(int width, int height) {
		// Only initialize once
		if (!this.initialized) {
			this.initialized = true;

			// Create buffer arrays
			int size = width * height;
			
			buf_warped_size = 0;

			buf_warped_residual = new float[size];
			buf_warped_weights = new float[size];
			buf_warped_dx = new float[size];
			buf_warped_dy = new float[size];
			buf_warped_x = new float[size];
			buf_warped_y = new float[size];
			buf_warped_z = new float[size];

			buf_d = new float[size];
			buf_residual_d = new float[size];
			buf_idepthVar = new float[size];
			buf_warped_idepthVar = new float[size];
			buf_weight_p = new float[size];
			buf_weight_d = new float[size];
			
		}
	}

	public SIM3 trackFrameSim3(
			TrackingReference reference,
			Frame frame,
			SIM3 frameToReference_initialEstimate,
			int startLevel, int finalLevel)
	{
		//boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();
	
		initialize(frame.width(0), frame.height(0));
		
		diverged = false;
	
		System.out.println("trackFrameSim3 init: " + Arrays.toString(frameToReference_initialEstimate.ln()));
	
		// ============ track frame ============
	    SIM3 referenceToFrame = frameToReference_initialEstimate.inverse();
		LGS7 ls7 = new LGS7();
	
	
		int[] numCalcResidualCalls = new int[Constants.PYRAMID_LEVELS];
		int[] numCalcWarpUpdateCalls = new int[Constants.PYRAMID_LEVELS];
	
		Sim3ResidualStruct finalResidual = new Sim3ResidualStruct();
	
		boolean warp_update_up_to_date = false;
	
		for(int lvl=startLevel;lvl >= finalLevel;lvl--)
		{
			numCalcResidualCalls[lvl] = 0;
			numCalcWarpUpdateCalls[lvl] = 0;
	
			if(maxItsPerLvl[lvl] == 0) {
				continue;
			}
	
			reference.createPointCloud(lvl);
	
			// evaluate baseline-residual.
			calcSim3Buffers(reference, frame, referenceToFrame, lvl);
			if(buf_warped_size < 0.5 * Constants.MIN_GOODPERALL_PIXEL_ABSMIN *
					(frame.width(0)>>lvl)*(frame.height(0)>>lvl) || buf_warped_size < 10)
			{
				diverged = true;
				System.out.println("trackFrameSim3 return 1");
				return new SIM3();
			}
	
			Sim3ResidualStruct lastErr = calcSim3WeightsAndResidual(referenceToFrame);
			/*
			if(plotSim3TrackingIterationInfo)
				calcSim3Buffers(reference, frame, referenceToFrame, lvl, true);
			*/
			numCalcResidualCalls[lvl]++;
	
			/*
			if(useAffineLightningEstimation)
			{
				affineEstimation_a = affineEstimation_a_lastIt;
				affineEstimation_b = affineEstimation_b_lastIt;
			}*/
	
			float LM_lambda = lambdaInitial[lvl];
	
			warp_update_up_to_date = false;
			for(int iteration=0; iteration < maxItsPerLvl[lvl]; iteration++)
			{
				System.out.println("----- iteration: " + iteration + " -----");
	
				// calculate LS System, result is saved in ls.
				calcSim3LGS(ls7);
				warp_update_up_to_date = true;
				numCalcWarpUpdateCalls[lvl]++;
	
				iterationNumber = iteration;
	
	
				int incTry=0;
				while(true)
				{
					// solve LS system with current lambda
					DenseMatrix b = ls7.b.div(-ls7.num_constraints);//7x1
					DenseMatrix A = ls7.A.div(ls7.num_constraints);//7x7
					
					System.out.println("A " + ls7.A);
					System.out.println("b " + ls7.b);
					System.out.println("nc " + ls7.num_constraints);
					
					for(int i=0;i<7;i++)
						 A.set(i, i, A.get(i,i) * (1+LM_lambda));
					DenseMatrix inc = A.ldltSolve(b);//7x1
					incTry++;
	
					

					double[] incVec = Vec.vec7ToArray(inc);
					
					System.out.println("incVec: " + Arrays.toString(incVec));
					
					float absInc = (float) Vec.dot(incVec, incVec);
					if(!(absInc >= 0 && absInc < 1))
					{
						// ERROR tracking diverged.
						System.err.println("absInc: " + absInc);
						lastSim3Hessian = jeigen.Shortcuts.zeros(7, 7);
						System.out.println("trackFrameSim3 return 2");
						return new SIM3();
					}
	
					// apply increment. pretty sure this way round is correct, but hard to test.
					SIM3 new_referenceToFrame = SIM3.exp(Vec.vec7ToArray(inc)).mul(referenceToFrame);
					//Sim3 new_referenceToFrame = referenceToFrame * Sim3::exp((inc));
	
	
					// re-evaluate residual
					calcSim3Buffers(reference, frame, new_referenceToFrame, lvl);
					if(buf_warped_size < 0.5 * Constants.MIN_GOODPERALL_PIXEL_ABSMIN
							* (frame.width(0)>>lvl)*(frame.height(0)>>lvl) || buf_warped_size < 10)
					{
						diverged = true;
						System.out.println("trackFrameSim3 return 3");
						return new SIM3();
					}
	
					Sim3ResidualStruct error = calcSim3WeightsAndResidual(new_referenceToFrame);
					//if(plotSim3TrackingIterationInfo)
					//	calcSim3Buffers(reference, frame, new_referenceToFrame, lvl, true);
					numCalcResidualCalls[lvl]++;
	
	
					// accept inc?
					if(error.mean < lastErr.mean)
					{
						// accept inc
						referenceToFrame = new_referenceToFrame;
						warp_update_up_to_date = false;
	
						/*
						if(useAffineLightningEstimation)
						{
							affineEstimation_a = affineEstimation_a_lastIt;
							affineEstimation_b = affineEstimation_b_lastIt;
						}*/
	
						/*
						if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							// debug output
							printf("(%d-%d): ACCEPTED increment of %f with lambda %.1f, residual: %f -> %f\n",
									lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr.mean, error.mean);
	
							printf("         p=%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
									referenceToFrame.log()[0],referenceToFrame.log()[1],referenceToFrame.log()[2],
									referenceToFrame.log()[3],referenceToFrame.log()[4],referenceToFrame.log()[5],
									referenceToFrame.log()[6]);
						}*/
	
						// converged?
						if(error.mean / lastErr.mean > convergenceEps[lvl])
						{
							/*
							if(enablePrintDebugInfo && printTrackingIterationInfo)
							{
								printf("(%d-%d): FINISHED pyramid level (last residual reduction too small).\n",
										lvl,iteration);
							}*/
							iteration = maxItsPerLvl[lvl];
						}
	
						finalResidual = lastErr = error;
	
						if(LM_lambda <= 0.2)
							LM_lambda = 0;
						else
							LM_lambda *= lambdaSuccessFac;
	
						break;
					}
					else
					{
						/*
						if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							printf("(%d-%d): REJECTED increment of %f with lambda %.1f, (residual: %f -> %f)\n",
									lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr.mean, error.mean);
						}*/

						incVec = Vec.vec7ToArray(inc);
						double incVecDot = Vec.dot(incVec, incVec);
						if(!(incVecDot > stepSizeMin[lvl]))
						{
							/*
							if(enablePrintDebugInfo && printTrackingIterationInfo)
							{
								printf("(%d-%d): FINISHED pyramid level (stepsize too small).\n",
										lvl,iteration);
							}*/
							iteration = maxItsPerLvl[lvl];
							break;
						}
	
						if(LM_lambda == 0)
							LM_lambda = 0.2f;
						else
							LM_lambda *= Math.pow(lambdaFailFac, incTry);
					}
				}
			}
		}
	
	
	
		/*
		if(enablePrintDebugInfo && printTrackingIterationInfo)
		{
			printf("Tracking: ");
				for(int lvl=PYRAMID_LEVELS-1;lvl >= 0;lvl--)
				{
					printf("lvl %d: %d (%d); ",
						lvl,
						numCalcResidualCalls[lvl],
						numCalcWarpUpdateCalls[lvl]);
				}
	
			printf("\n");
	
	
			printf("pOld = %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
					frameToReference_initialEstimate.inverse().log()[0],frameToReference_initialEstimate.inverse().log()[1],frameToReference_initialEstimate.inverse().log()[2],
					frameToReference_initialEstimate.inverse().log()[3],frameToReference_initialEstimate.inverse().log()[4],frameToReference_initialEstimate.inverse().log()[5],
					frameToReference_initialEstimate.inverse().log()[6]);
			printf("pNew = %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
					referenceToFrame.log()[0],referenceToFrame.log()[1],referenceToFrame.log()[2],
					referenceToFrame.log()[3],referenceToFrame.log()[4],referenceToFrame.log()[5],
					referenceToFrame.log()[6]);
			printf("final res mean: %f meanD %f, meanP %f\n", finalResidual.mean, finalResidual.meanD, finalResidual.meanP);
		}
		*/
	
	
		// Make sure that there is a warp update at the final position to get the correct information matrix
		if (!warp_update_up_to_date)
		{
			reference.createPointCloud(finalLevel);
			calcSim3Buffers(reference, frame, referenceToFrame, finalLevel);
		    finalResidual = calcSim3WeightsAndResidual(referenceToFrame);
		    calcSim3LGS(ls7);
		}
	
		lastSim3Hessian = ls7.A;
	
	
		if(referenceToFrame.getScale() <= 0 )
		{
			diverged = true;
			System.out.println("trackFrameSim3 return 4");
			return new SIM3();
		}
	
		lastResidual = finalResidual.mean;
		lastDepthResidual = finalResidual.meanD;
		lastPhotometricResidual = finalResidual.meanP;
	
		System.out.println("TrackFrameSim3 referenceToFrame: " + Arrays.toString(referenceToFrame.ln()));
		return referenceToFrame.inverse();
	}
	

	void calcSim3Buffers(
			TrackingReference reference,
			Frame frame,
			SIM3 referenceToFrame,
			int level)
	{
		/*
		if(plotSim3TrackingIterationInfo)
		{
			cv::Vec3b col = cv::Vec3b(255,170,168);
			fillCvMat(&debugImageResiduals,col);
			fillCvMat(&debugImageOldImageSource,col);
			fillCvMat(&debugImageOldImageWarped,col);
			fillCvMat(&debugImageDepthResiduals,col);
		}
		if(plotWeights && plotSim3TrackingIterationInfo)
		{
			cv::Vec3b col = cv::Vec3b(255,170,168);
			fillCvMat(&debugImageHuberWeight,col);
			fillCvMat(&debugImageWeightD,col);
			fillCvMat(&debugImageWeightP,col);
			fillCvMat(&debugImageWeightedResP,col);
			fillCvMat(&debugImageWeightedResD,col);
		}*/
	
		// get static values
		int w = frame.width(level);
		int h = frame.height(level);
		DenseMatrix KLvl = Constants.K[level];
		float fx_l = (float) KLvl.get(0,0);
		float fy_l = (float) KLvl.get(1,1);
		float cx_l = (float) KLvl.get(0,2);
		float cy_l = (float) KLvl.get(1,2);
	
		
		DenseMatrix rotMat = referenceToFrame.getRotationMat();			// 3x3
		DenseMatrix rotMatUnscaled = referenceToFrame.getRotationMat();	// 3x3
		DenseMatrix transVec = referenceToFrame.getTranslationMat();	// 3x1
	
		// Calculate rotation around optical axis for rotating source frame gradients
		//DenseMatrix forwardVector = new DenseMatrix(new double[][]{{0}, {0}, {-1}});		// 3x1
		
		//System.out.println("forwardVector " + forwardVector.rows + ", " + forwardVector.cols);
		//System.out.println("rotMatUnscaled " + rotMatUnscaled.rows + ", " + rotMatUnscaled.cols);
		
		//DenseMatrix rotatedForwardVector = rotMatUnscaled.mul(forwardVector);			// 3x1
		
		// TODO: define these.
		/*
		Eigen::Quaternionf shortestBackRotation;
		shortestBackRotation.setFromTwoVectors(rotatedForwardVector, forwardVector);
		DenseMatrix rollMat = shortestBackRotation.toRotationMatrix() * rotMatUnscaled;		// 3x3
		float xRoll0 = (float) rollMat.get(0, 0);
		float xRoll1 = (float) rollMat.get(0, 1);
		float yRoll0 = (float) rollMat.get(1, 0);
		float yRoll1 = (float) rollMat.get(1, 1);
		*/
	
	
		DenseMatrix[] refPoint = reference.posDataLvl[level];
		DenseMatrix[] refColVar = reference.colorAndVarDataLvl[level];
		//Eigen::Vector2f* refGrad = reference.gradDataLvl[level];
	
		float[] frame_idepth = frame.inverseDepthLvl[level];
		float[] frame_idepthVar = frame.inverseDepthVarianceLvl[level];
		//Eigen::Vector4f* frame_intensityAndGradients = frame.gradients[level];
	
		float[] frame_gradx = frame.imageGradientXArrayLvl[level];
		float[] frame_grady = frame.imageGradientYArrayLvl[level];
		float[] frame_gray = frame.imageArrayLvl[level];
		
	
		float sxx=0,syy=0,sx=0,sy=0,sw=0;
	
		float usageCount = 0;
	
		int idx=0;
		
		int numValidPoints = 0;
		
		// TODO: REMOVE
		float sumResidual = 0;
		
		System.out.println("rotMat: " + rotMat.toString());
		System.out.println("transVec: " + transVec.toString());
		
		System.out.println("level " + level);
		
		for(int i=0 ; i<refPoint.length ; i++) {
			
			if (refPoint[i] == null) {
				continue;
			}
			numValidPoints++;
			
			DenseMatrix point = refPoint[i];
			
			DenseMatrix Wxp = rotMat.mmul(point).add(transVec); // 3x1
			float u_new = (float) ((Wxp.get(0,0)/Wxp.get(2,0))*fx_l + cx_l);
			float v_new = (float) ((Wxp.get(1,0)/Wxp.get(2,0))*fy_l + cy_l);
	
			// step 1a: coordinates have to be in image:
			// (inverse test to exclude NANs)
			if(!(u_new > 1 && v_new > 1 && u_new < w-2 && v_new < h-2))
				continue;
	
			buf_warped_x[idx] = (float) Wxp.get(0,0);
			buf_warped_y[idx] = (float) Wxp.get(1,0);
			buf_warped_z[idx] = (float) Wxp.get(2,0);
	
			//DenseMatrix resInterp = getInterpolatedElement43(
			//		frame_intensityAndGradients, u_new, v_new, w);	// 3x1
			
			float frame_gradx_interp = Utils.Utils.interpolatedValue(frame_gradx, u_new, v_new, w);
			float frame_grady_interp = Utils.Utils.interpolatedValue(frame_grady, u_new, v_new, w);
			float frame_frame_gray_interp = Utils.Utils.interpolatedValue(frame_gray, u_new, v_new, w);
			
			
	
			// TODO: suppposed to USE_ESM_TRACKING
			// save values
	//#if USE_ESM_TRACKING == 1
			// get rotated gradient of point
//			float rotatedGradX = xRoll0 * (*refGrad)[0] + xRoll1 * (*refGrad)[1];
//			float rotatedGradY = yRoll0 * (*refGrad)[0] + yRoll1 * (*refGrad)[1];
//	
//			buf_warped_dx[idx] = fx_l * 0.5f * (frame_gradx_interp + rotatedGradX);
//			buf_warped_dy[idx] = fy_l * 0.5f * (frame_grady_interp + rotatedGradY);
	//#else
			buf_warped_dx[idx] = fx_l * frame_gradx_interp;
			buf_warped_dy[idx] = fy_l * frame_grady_interp;
	//#endif
	
	
			float c1 = (float) (affineEstimation_a * refColVar[i].get(0, 0) + affineEstimation_b);
			float c2 = frame_frame_gray_interp;
			float residual_p = c1 - c2;
	
			float weight = Math.abs(residual_p) < 2.0f ? 1 : 2.0f / Math.abs(residual_p);
			sxx += c1*c1*weight;
			syy += c2*c2*weight;
			sx += c1*weight;
			sy += c2*weight;
			sw += weight;
	
	
			buf_warped_residual[idx] = residual_p;
			buf_idepthVar[idx] = (float) refColVar[i].get(1, 0);
			
			sumResidual += residual_p;
	
			// new (only for Sim3):
			int idx_rounded = (int)(u_new+0.5f) + w*(int)(v_new+0.5f);
			float var_frameDepth = frame_idepthVar[idx_rounded];
			float ref_idepth = (float) (1.0f / Wxp.get(2, 0));
			buf_d[idx] = (float) (1.0f / refPoint[i].get(2,0));
			if(var_frameDepth > 0)
			{
				float residual_d = ref_idepth - frame_idepth[idx_rounded];
				buf_residual_d[idx] = residual_d;
				buf_warped_idepthVar[idx] = var_frameDepth;
			}
			else
			{
				buf_residual_d[idx] = -1;
				buf_warped_idepthVar[idx] = -1;
			}
	
	
			// DEBUG STUFF
			/*
			if(plotSim3TrackingIterationInfo)
			{
				// for debug plot only: find x,y again.
				// horribly inefficient, but who cares at this point...
				Eigen::Vector3f point = KLvl * (*refPoint);
				int x = point[0] / point[2] + 0.5f;
				int y = point[1] / point[2] + 0.5f;
	
				setPixelInCvMat(&debugImageOldImageSource,getGrayCvPixel((float)resInterp[2]),u_new+0.5,v_new+0.5,(width/w));
				setPixelInCvMat(&debugImageOldImageWarped,getGrayCvPixel((float)resInterp[2]),x,y,(width/w));
				setPixelInCvMat(&debugImageResiduals,getGrayCvPixel(residual_p+128),x,y,(width/w));
	
				if(*(buf_warped_idepthVar+idx) >= 0)
				{
					setPixelInCvMat(&debugImageDepthResiduals,getGrayCvPixel(128 + 800 * *(buf_residual_d+idx)),x,y,(width/w));
	
					if(plotWeights)
					{
						setPixelInCvMat(&debugImageWeightD,getGrayCvPixel(255 * (1/60.0f) * sqrtf(*(buf_weight_VarD+idx))),x,y,(width/w));
						setPixelInCvMat(&debugImageWeightedResD,getGrayCvPixel(128 + (128/5.0f) * sqrtf(*(buf_weight_VarD+idx)) * *(buf_residual_d+idx)),x,y,(width/w));
					}
				}
	
	
				if(plotWeights)
				{
					setPixelInCvMat(&debugImageWeightP,getGrayCvPixel(255 * 4 * sqrtf(*(buf_weight_VarP+idx))),x,y,(width/w));
					setPixelInCvMat(&debugImageHuberWeight,getGrayCvPixel(255 * *(buf_weight_Huber+idx)),x,y,(width/w));
					setPixelInCvMat(&debugImageWeightedResP,getGrayCvPixel(128 + (128/5.0f) * sqrtf(*(buf_weight_VarP+idx)) * *(buf_warped_residual+idx)),x,y,(width/w));
				}
			}*/
	
			idx++;
	
			float depthChange = (float) (refPoint[i].get(2,0) / Wxp.get(2, 0));
			usageCount += depthChange < 1 ? depthChange : 1;
		}
		buf_warped_size = idx;
	
	
		pointUsage = usageCount / (float)numValidPoints;
	
		//affineEstimation_a_lastIt = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));
		//affineEstimation_b_lastIt = (sy - affineEstimation_a_lastIt*sx)/sw;
		
		System.out.println("SUMRESIDUAL " + sumResidual);
		System.out.println("numValid " + numValidPoints);
		System.out.println("usageCount " + usageCount);
		
		
		/*
		if(plotSim3TrackingIterationInfo)
		{
			Util::displayImage( "P Residuals", debugImageResiduals );
			Util::displayImage( "D Residuals", debugImageDepthResiduals );
	
			if(plotWeights)
			{
				Util::displayImage( "Huber Weights", debugImageHuberWeight );
				Util::displayImage( "DV Weights", debugImageWeightD );
				Util::displayImage( "IV Weights", debugImageWeightP );
				Util::displayImage( "WP Res", debugImageWeightedResP );
				Util::displayImage( "WD Res", debugImageWeightedResD );
			}
		}*/
	
	}

	Sim3ResidualStruct calcSim3WeightsAndResidual(SIM3 referenceToFrame)
	{
		float tx = (float) referenceToFrame.getTranslation()[0];
		float ty = (float) referenceToFrame.getTranslation()[1];
		float tz = (float) referenceToFrame.getTranslation()[2];

		Sim3ResidualStruct sumRes = new Sim3ResidualStruct();
		//memset(&sumRes, 0, sizeof(Sim3ResidualStruct));

		//float sum_rd=0, sum_rp=0, sum_wrd=0, sum_wrp=0, sum_wp=0, sum_wd=0, sum_num_d=0, sum_num_p=0;

		float sumpx = 0;
		float sumpy = 0;
		float sumpz = 0;
		float sumd  = 0;
		float sumrp = 0;
		float sumrd = 0;
		float sumgx = 0;
		float sumgy = 0;
		float sums  = 0;
		float sumsv = 0;
		
		for(int i=0;i<buf_warped_size;i++)
		{
			float px = buf_warped_x[i];	// x'
			float py = buf_warped_y[i];	// y'
			float pz = buf_warped_z[i];	// z'

			float d  = buf_d[i];	// d

			float rp = buf_warped_residual[i]; 	// r_p
			float rd = buf_residual_d[i];	 	// r_d

			float gx = buf_warped_dx[i];	// \delta_x I
			float gy = buf_warped_dy[i];	// \delta_y I

			float s = var_weight * buf_idepthVar[i];	// \sigma_d^2
			float sv = var_weight * buf_warped_idepthVar[i];	// \sigma_d^2'

			sumpx += px;
			sumpy += py;
			sumpz += pz;
			sumd  += d ;
			sumrp += rp;
			sumrd += rd;
			sumgx += gx;
			sumgy += gy;
			sums  += s ;
			sumsv += sv;

			// calc dw/dd (first 2 components):
			float g0 = (tx * pz - tz * px) / (pz*pz*d);
			float g1 = (ty * pz - tz * py) / (pz*pz*d);
			float g2 = (pz - tz) / (pz*pz*d);

			// calc w_p
			float drpdd = gx * g0 + gy * g1;	// ommitting the minus
			float w_p = 1.0f / (cameraPixelNoise2 + s * drpdd * drpdd);

			float w_d = 1.0f / (sv + g2*g2*s);

			float weighted_rd = (float) Math.abs(rd*Math.sqrt((w_d)));
			float weighted_rp = (float) Math.abs(rp*Math.sqrt((w_p)));


			float weighted_abs_res = sv > 0 ? weighted_rd+weighted_rp : weighted_rp;
			float wh = Math.abs(weighted_abs_res < huber_d ? 
					1 : huber_d / weighted_abs_res);

			if(sv > 0)
			{
				sumRes.sumResD += wh * w_d * rd*rd;
				sumRes.numTermsD++;
			}

			sumRes.sumResP += wh * w_p * rp*rp;
			sumRes.numTermsP++;


			/*
			if(plotSim3TrackingIterationInfo)
			{
				// for debug
				*(buf_weight_Huber+i) = wh;
				*(buf_weight_VarP+i) = w_p;
				*(buf_weight_VarD+i) = sv > 0 ? w_d : 0;


				sum_rp += Math.abs(rp);
				sum_wrp += Math.abs(weighted_rp);
				sum_wp += Math.sqrt(w_p);
				sum_num_p++;

				if(sv > 0)
				{
					sum_rd += fabs(weighted_rd);
					sum_wrd += fabs(rd);
					sum_wd += Math.sqrt((w_d);
					sum_num_d++;
				}
			}
			*/

			buf_weight_p[i] = wh * w_p;

			if(sv > 0)
				buf_weight_d[i] = wh * w_d;
			else
				buf_weight_d[i] = 0;

		}

		sumRes.mean = (sumRes.sumResD + sumRes.sumResP) / (sumRes.numTermsD + sumRes.numTermsP);
		sumRes.meanD = (sumRes.sumResD) / (sumRes.numTermsD);
		sumRes.meanP = (sumRes.sumResP) / (sumRes.numTermsP);

		System.out.println("sumRes " + sumRes.mean + ", " + sumRes.meanD + ", " + sumRes.meanP
				 			+ ", " + sumRes.numTermsD + ", " + sumRes.numTermsP
				 			 + ", " + sumRes.sumResD + ", " + sumRes.sumResP);
		
		System.out.println("sums "+ sumpx + " " +
									sumpy + " " +
									sumpz + " " +
									sumd  + " " +
									sumrp + " " +
									sumrd + " " +
									sumgx + " " +
									sumgy + " " +
									sums  + " " +
									sumsv + " " );

		
		/*
		if(plotSim3TrackingIterationInfo)
		{
			printf("rd %f, rp %f, wrd %f, wrp %f, wd %f, wp %f\n ",
					sum_rd/sum_num_d,
					sum_rp/sum_num_p,
					sum_wrd/sum_num_d,
					sum_wrp/sum_num_p,
					sum_wd/sum_num_d,
					sum_wp/sum_num_p);
		}
		*/
		return sumRes;
	}
	
	void calcSim3LGS(LGS7 ls7)
	{
		LGS4 ls4 = new LGS4();
		LGS6 ls6 = new LGS6();
//		ls6.initialize(width*height);
//		ls4.initialize(width*height);

		float z_sqr_sum = 0;
		float rd_sum = 0;
		float wd_sum = 0;
		
		for(int i=0;i<buf_warped_size;i++)
		{
			float px = buf_warped_x[i];	// x'
			float py = buf_warped_y[i];	// y'
			float pz = buf_warped_z[i];	// z'

			float wp = buf_weight_p[i];	// wr/wp
			float wd = buf_weight_d[i];	// wr/wd

			float rp = buf_warped_residual[i]; // r_p
			float rd = buf_residual_d[i];	 // r_d

			float gx = buf_warped_dx[i];  // \delta_x I
			float gy = buf_warped_dy[i];  // \delta_y I

					
			float z = 1.0f / pz;
			float z_sqr = 1.0f / (pz*pz);
			DenseMatrix v;		// 6x1
			DenseMatrix v4;		// 4x1
			
			v = new DenseMatrix(new double[][]{
					{z*gx},
					{z*gy},
					{(-px * z_sqr) * gx + (-py * z_sqr) * gy},
					{(-px * py * z_sqr) * gx + (-(1.0 + py * py * z_sqr)) * gy},
					{(1.0 + px * px * z_sqr) * gx + (px * py * z_sqr) * gy},
					{(-py * z) * gx + (px * z) * gy}
			});
			
			
			// new:			
			v4 = new DenseMatrix(new double[][]{
				{z_sqr},
				{z_sqr * py},
				{-z_sqr * px},
				{z}
			});
			
			// TODO: REMOVE
			z_sqr_sum += z_sqr;
			rd_sum += rd;
			wd_sum += wd;
			

			ls6.update(v, rp, wp);	// Jac = - v
			ls4.update(v4, rd, wd);	// Jac = v4

		}

		System.out.println("==" + z_sqr_sum + " " + rd_sum + " " + wd_sum);
		
		//ls4.finishNoDivide();
		//ls6.finishNoDivide();

		System.out.println("ls4.A " + ls4.A);
		System.out.println("ls4.b " + ls4.b);
		System.out.println("ls4.nc " + ls4.num_constraints);
		System.out.println("ls4.error " + ls4.error);
		
		System.out.println("ls6.A " + ls6.A);
		System.out.println("ls6.b " + ls6.b);
		System.out.println("ls6.nc " + ls6.numConstraints);
		System.out.println("ls6.error " + ls6.error);
		
		ls7.initializeFrom(ls6, ls4);

	}
	
}
