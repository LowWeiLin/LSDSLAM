package DepthEstimation;

import java.util.Deque;
import java.util.List;
import java.util.Random;

import Utils.Constants;
import DataStructures.Frame;
import LieAlgebra.SIM3;

/**
 * Keeps a detailed depth map (consisting of DepthMapPixelHypothesis) and does
 * stereo comparisons and regularization to update it.
 */
public class DepthMap {

	
	int width;
	int height;
	
	DepthMapPixelHypothesis[] otherDepthMap;
	DepthMapPixelHypothesis[] currentDepthMap;
	
	Frame activeKeyFrame;
	boolean activeKeyFrameIsReactivated;
	
	

	Frame oldest_referenceFrame;
	Frame newest_referenceFrame;
	List<Frame> referenceFrameByID;
	int referenceFrameByID_offset;
	
	
	public DepthMap(int w, int h) {
		width = w;
		height = h;
		

		activeKeyFrame = null;
		activeKeyFrameIsReactivated = false;
		otherDepthMap = new DepthMapPixelHypothesis[width*height];
		currentDepthMap = new DepthMapPixelHypothesis[width*height];
	

//		validityIntegralBuffer = (int*)Eigen::internal::aligned_malloc(width*height*sizeof(int));
//
//		debugImageHypothesisHandling = cv::Mat(h,w, CV_8UC3);
//		debugImageHypothesisPropagation = cv::Mat(h,w, CV_8UC3);
//		debugImageStereoLines = cv::Mat(h,w, CV_8UC3);
//		debugImageDepth = cv::Mat(h,w, CV_8UC3);
	
		

		reset();
		
		
		
	}

	public void reset() {
		if (otherDepthMap == null) {
			otherDepthMap = new DepthMapPixelHypothesis[width*height];
		}
		if (currentDepthMap == null) {
			currentDepthMap = new DepthMapPixelHypothesis[width*height];
		}
		
		for (int i=0 ; i<width*height ; i++) {
			if (otherDepthMap[i] == null) {
				otherDepthMap[i] = new DepthMapPixelHypothesis();
			}
			if (currentDepthMap[i] == null) {
				currentDepthMap[i] = new DepthMapPixelHypothesis();
			}
			
			otherDepthMap[i].isValid = false;
			currentDepthMap[i].isValid = false;
		}
	}
	
	public boolean isValid() {
		return activeKeyFrame != null;
	}
	
	public void initializeRandomly(Frame newFrame) {
		activeKeyFrame = newFrame;
		activeKeyFrameIsReactivated = false;
		
		
		Random random = new Random(System.nanoTime());
		
		float[] maxGradients = newFrame.imageGradientMaxArrayLvl[0];
		
		for(int y=1;y<height-1;y++) {
			for(int x=1;x<width-1;x++) {
				if(maxGradients[x+y*width] > Constants.MIN_ABS_GRAD_CREATE) {
					float idepth = 0.5f + 1.0f * (random.nextInt(100001) / 100000.0f);
					currentDepthMap[x+y*width] = new DepthMapPixelHypothesis(
							idepth,
							idepth,
							Constants.VAR_RANDOM_INIT_INITIAL,
							Constants.VAR_RANDOM_INIT_INITIAL,
							20);
				} else {
					currentDepthMap[x+y*width].isValid = false;
					currentDepthMap[x+y*width].blacklisted = 0;
				}
			}
		}
		

		activeKeyFrame.setDepth(currentDepthMap);
		
	}
	
	/**
	 * Updates depth map with observations from deque of frames
	 */
	public void updateKeyframe(Deque<Frame> referenceFrames) {
		assert(isValid());

		// Get oldest/newest frames
		oldest_referenceFrame = referenceFrames.peekFirst();
		newest_referenceFrame = referenceFrames.peekLast();

		referenceFrameByID.clear();
		referenceFrameByID_offset = oldest_referenceFrame.id();

		// For each frame
		for(Frame frame : referenceFrames) {
			
			//Checks that tracking parent is valid
			assert(frame.hasTrackingParent());
			if(frame.getTrackingParent() != activeKeyFrame) {
				System.out.printf("WARNING: updating frame %d with %d,"
						+ " which was tracked on a different frame (%d)."
						+ "\nWhile this should work, it is not recommended.",
						activeKeyFrame.id(), frame.id(),
						frame.getTrackingParent().id());
			}

			SIM3 refToKf;
			// Get SIM3 from frame to keyframe
			if(frame.pose.trackingParent.frameID == activeKeyFrame.id()) {
				refToKf = frame.pose.thisToParent_raw;
			} else {
				refToKf = activeKeyFrame.getScaledCamToWorld().inverse().mul(frame.getScaledCamToWorld());
			}
			
			// Prepare for stereo ??
			// prepare frame for stereo with keyframe, SE3, K, level
			frame.prepareForStereoWith(activeKeyFrame, refToKf, 0);

			// TODO: ?? push what?
//			while((int)referenceFrameByID.size() + referenceFrameByID_offset <= frame->id()) {
//				referenceFrameByID.add(frame);
//			}
		}

		
//		if(plotStereoImages)
//		{
//			cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(), CV_32F, const_cast<float*>(activeKeyFrameImageData));
//			keyFrameImage.convertTo(debugImageHypothesisHandling, CV_8UC1);
//			cv::cvtColor(debugImageHypothesisHandling, debugImageHypothesisHandling, CV_GRAY2RGB);
//
//			cv::Mat oldest_refImage(oldest_referenceFrame->height(), oldest_referenceFrame->width(), CV_32F, const_cast<float*>(oldest_referenceFrame->image(0)));
//			cv::Mat newest_refImage(newest_referenceFrame->height(), newest_referenceFrame->width(), CV_32F, const_cast<float*>(newest_referenceFrame->image(0)));
//			cv::Mat rfimg = 0.5f*oldest_refImage + 0.5f*newest_refImage;
//			rfimg.convertTo(debugImageStereoLines, CV_8UC1);
//			cv::cvtColor(debugImageStereoLines, debugImageStereoLines, CV_GRAY2RGB);
//		}

		//*** OBSERVE DEPTH HERE
		observeDepth();
		//***


		// Regularize, fill holes?
		// TODO:
		//regularizeDepthMapFillHoles();

		// Regularize?
		// TODO:
		//regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
		
		// Update depth in keyframe
		if(!activeKeyFrame.depthHasBeenUpdatedFlag) {
			// Update keyframe with updated depth?
			activeKeyFrame.setDepth(currentDepthMap);
		}

		activeKeyFrame.numMappedOnThis++;
		activeKeyFrame.numMappedOnThisTotal++;

	}
	
	/**
	 * ObserveDepth
	 */
	void observeDepth() {
		// observeDepthRow in a multithreaded way.
		//threadReducer.reduce(boost::bind(&DepthMap::observeDepthRow, this, _1, _2, _3),
		//						3, height-3, 10);

		// TODO: make multithreaded
		observeDepthRow(3, height-3);
		
	}
	
	/**
	 * ObserveDepth for specified rows
	 * @param yMin
	 * @param yMax
	 */
	void observeDepthRow(int yMin, int yMax) {
		float[] keyFrameMaxGradBuf = activeKeyFrame.imageGradientMaxArrayLvl[0];

		int successes = 0;

		// For each row assigned
		for(int y=yMin;y<yMax; y++) {
			// For x 3 to width-3
			for(int x=3;x<width-3;x++) {

				//
				// For each pixel
				//

				int idx = x+y*width;
				DepthMapPixelHypothesis target = currentDepthMap[idx];
				boolean hasHypothesis = target.isValid;

				// ======== 1. check absolute grad =========
				if(hasHypothesis && keyFrameMaxGradBuf[idx] < 
						Constants.MIN_ABS_GRAD_DECREASE) {
					target.isValid = false;
					continue;
				}

				if(keyFrameMaxGradBuf[idx] < Constants.MIN_ABS_GRAD_CREATE ||
						target.blacklisted < Constants.MIN_BLACKLIST) {
					continue;
				}

				// Gradient is significant, pixel not blacklisted
				boolean success = false;
				if(!hasHypothesis) {
					// First time
					success = observeDepthCreate(x, y, idx);
				} else {
					// Observe depth
					// TODO: ***
					//success = observeDepthUpdate(x, y, idx, keyFrameMaxGradBuf);
				}
				if(success) {
					successes++;
				}
			}
		}
	}
	
	boolean observeDepthCreate(int x, int y, int idx) {
		DepthMapPixelHypothesis target = currentDepthMap[idx];

		// ???
		// What is activeKeyFrameIsReactivated?
		// Key frame was used before?
		Frame refFrame = activeKeyFrameIsReactivated ? 
				newest_referenceFrame : oldest_referenceFrame;

		// Frame tracked against activeKeyFrame?
		if(refFrame.getTrackingParent() == activeKeyFrame) {
			/*
			//TODO: implement this
			boolean wasGoodDuringTracking = refFrame.refPixelWasGoodNoCreate();

			// Check if pixel is good during tracking?
			if(wasGoodDuringTracking != false &&
			 !wasGoodDuringTracking[
			 	(x >> Constants.SE3TRACKING_MIN_LEVEL) + 
			 	(width >> Constants.SE3TRACKING_MIN_LEVEL)*(y >> Constants.SE3TRACKING_MIN_LEVEL)])
			{
				return false;
			}
			*/
		}


		// Get epipolar line??
		float epx, epy;
		// x, y pixel coordinate, refFrame
		float[] epl = makeAndCheckEPL(x, y, refFrame);
		
		if(epl == null) {
			return false;
		} else {
			epx = epl[0];
			epy = epl[1];
		}

		float new_u = x;
		float new_v = y;
		float result_idepth = 0;
		float result_var = 0;
		float result_eplLength = 0;
		
		// Do line stereo, get error, ^ results
		float error = doLineStereo(
				new_u, new_v, epx, epy,
				0.0f, 1.0f, 1.0f/Constants.MIN_DEPTH,
				refFrame, refFrame.imageArrayLvl[0],
				result_idepth, result_var, result_eplLength);

		if(error == -3 || error == -2) {
			target.blacklisted--;
		}

		if(error < 0 || result_var > Constants.MAX_VAR) {
			return false;
		}
		
		result_idepth = (float) UNZERO(result_idepth);

		// add hypothesis
		// Set/change the hypothesis
		target = new DepthMapPixelHypothesis(
				result_idepth,
				result_var,
				Constants.VALIDITY_COUNTER_INITIAL_OBSERVE);

		//if(plotStereoImages)
		//	debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(255,255,255); // white for GOT CREATED

		return true;
	}
	

	/**
	 * Return null if failed, return float[] {epx, epy} if found.
	 * */
	public float[] makeAndCheckEPL(int x, int y, Frame ref) {
		int idx = x + y*width;
		
		double fx = Constants.fx[0];
		double fy = Constants.fy[0];
		double cx = Constants.cx[0];
		double cy = Constants.cy[0];
		

		// ======= make epl ========
		// calculate the plane spanned by the two camera centers and the point (x,y,1)
		// intersect it with the keyframe's image plane (at depth=1)
		double epx = - fx * ref.thisToOther_t.get(0,0) + ref.thisToOther_t.get(2,0)*(x - cx);
		double epy = - fy * ref.thisToOther_t.get(1,0) + ref.thisToOther_t.get(2,0)*(y - cy);

		if(Double.isNaN(epx+epy)) {
			return null;
		}
		
		// ======== check epl length =========
		float eplLengthSquared = (float) (epx*epx+epy*epy);
		if(eplLengthSquared < Constants.MIN_EPL_LENGTH_SQUARED) {
			return null;
		}


		// ===== check epl-grad magnitude ======
		
		float gx = activeKeyFrame.imageGradientXArrayLvl[0][idx];
		float gy = activeKeyFrame.imageGradientYArrayLvl[0][idx];
		float eplGradSquared = activeKeyFrame.imageGradientMaxArrayLvl[0][idx];
		eplGradSquared = eplGradSquared*eplGradSquared / eplLengthSquared;	// square and norm with epl-length

		if(eplGradSquared < Constants.MIN_EPL_GRAD_SQUARED) {
			return null;
		}


		// ===== check epl-grad angle ======
		if(eplGradSquared / (gx*gx+gy*gy) < Constants.MIN_EPL_ANGLE_SQUARED) {
			return null;
		}


		// ===== DONE - return "normalized" epl =====
		float fac = (float) (Constants.GRADIENT_SAMPLE_DIST / Math.sqrt(eplLengthSquared));
		
		float pepx = (float) (epx * fac);
		float pepy = (float) (epy * fac);

		return new float[] {pepx, pepy};
	}


	private float doLineStereo(float new_u, float new_v, float epx, float epy,
			float f, float g, float h, Frame refFrame, byte[] bs,
			float result_idepth, float result_var, float result_eplLength) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	/*
	 * Make val non-zero
	 */
	double UNZERO(double val) {
		return (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val));
	}
	
}
