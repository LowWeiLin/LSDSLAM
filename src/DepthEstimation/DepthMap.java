package DepthEstimation;

import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Random;

import jeigen.DenseMatrix;
import DataStructures.Frame;
import LieAlgebra.SE3;
import LieAlgebra.SIM3;
import LieAlgebra.Vec;
import Utils.Constants;
import Utils.Utils;

/**
 * Keeps a detailed depth map (consisting of DepthMapPixelHypothesis) and does
 * stereo comparisons and regularization to update it.
 */
public class DepthMap {

	/** ============== Depth Variance Handling ======================= */
	public static final float SUCC_VAR_INC_FAC = (1.01f); // before an
															// ekf-update, the
															// variance is
															// increased by this
															// factor.
	public static final float FAIL_VAR_INC_FAC = 1.1f; // after a failed stereo
														// observation, the
														// variance is increased
														// by this factor.
	public static final float MAX_VAR = (0.5f * 0.5f); // initial variance on
														// creation - if
														// variance becomes
														// larter than this,
														// hypothesis is
														// removed.

	public static final float VAR_GT_INIT_INITIAL = 0.01f * 0.01f; // initial
																	// variance
																	// for
																	// Ground
																	// Truth
																	// Initialization
	public static final float VAR_RANDOM_INIT_INITIAL = (0.5f * MAX_VAR); // initial
																			// variance
																			// for
																			// Random
																			// Initialization

	int width;
	int height;

	DepthMapPixelHypothesis[] otherDepthMap;
	DepthMapPixelHypothesis[] currentDepthMap;
	int[] validityIntegralBuffer;

	Frame activeKeyFrame;
	boolean activeKeyFrameIsReactivated;

	Frame oldest_referenceFrame;
	Frame newest_referenceFrame;
	List<Frame> referenceFrameByID = new ArrayList<Frame>();
	int referenceFrameByID_offset;

	// Camera matrix
	float fx, fy, cx, cy;
	float fxi, fyi, cxi, cyi;

	public DepthMap(int w, int h) {
		width = w;
		height = h;

		activeKeyFrame = null;
		activeKeyFrameIsReactivated = false;
		otherDepthMap = new DepthMapPixelHypothesis[width * height];
		currentDepthMap = new DepthMapPixelHypothesis[width * height];

		validityIntegralBuffer = new int[width * height];

		fx = (float) Constants.K[0].get(0, 0);
		fy = (float) Constants.K[0].get(1, 1);
		cx = (float) Constants.K[0].get(0, 2);
		cy = (float) Constants.K[0].get(1, 2);

		fxi = (float) Constants.KInv[0].get(0, 0);
		fyi = (float) Constants.KInv[0].get(1, 1);
		cxi = (float) Constants.KInv[0].get(0, 2);
		cyi = (float) Constants.KInv[0].get(1, 2);

		reset();
	}

	public void reset() {
		if (otherDepthMap == null) {
			otherDepthMap = new DepthMapPixelHypothesis[width * height];
		}
		if (currentDepthMap == null) {
			currentDepthMap = new DepthMapPixelHypothesis[width * height];
		}

		for (int i = 0; i < width * height; i++) {
			if (otherDepthMap[i] == null) {
				otherDepthMap[i] = new DepthMapPixelHypothesis();
			}

			if (currentDepthMap[i] == null) {
				// Default hypothesis for all pixels
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

		// Random random = new Random(System.nanoTime());
		Random random = new Random(0);

		float[] maxGradients = newFrame.imageGradientMaxArrayLvl[0];

		int goodGrad = 0;

		// For each pixel
		for (int y = 1; y < height - 1; y++) {
			for (int x = 1; x < width - 1; x++) {

				// For pixels with significant gradient
				if (maxGradients[x + y * width] > Constants.MIN_ABS_GRAD_CREATE) {
					// TODO: Get random idepth
					float idepth = 0.5f + 1.0f * (random.nextInt(100001) /100000.0f);

					// Set fixed initial depth
					//float idepth = 0.5f + 1.0f * 0.5f;

					// Set hypothesis, random idepth and initial variance.
					currentDepthMap[x + y * width] = new DepthMapPixelHypothesis(
							idepth, idepth, Constants.VAR_RANDOM_INIT_INITIAL,
							Constants.VAR_RANDOM_INIT_INITIAL, 20);
					goodGrad++;
				} else {
					// Mark as invalid
					currentDepthMap[x + y * width].isValid = false;
					currentDepthMap[x + y * width].blacklisted = 0;
				}
			}
		}
		System.out.println("Good grad: " + goodGrad);
		// Set depth hypothesis depth values to keyframe
		activeKeyFrame.setDepth(currentDepthMap);
	}

	/**
	 * Updates depth map with observations from deque of frames
	 */
	public void updateKeyframe(Deque<Frame> referenceFrames) {
		assert (isValid());

		// Get oldest/newest frames
		oldest_referenceFrame = referenceFrames.peekFirst();
		newest_referenceFrame = referenceFrames.peekLast();

		referenceFrameByID.clear();
		referenceFrameByID_offset = oldest_referenceFrame.id();

		// For each frame
		for (Frame frame : referenceFrames) {
			//System.out.println("Updating keyframe " + activeKeyFrame.id() + " with " + frame.id() + ".");
			
			// Do not update keyframe with own frame.
			if (frame.id() == activeKeyFrame.id()) {
				continue;
			}
			
			// Checks that tracking parent is valid
			assert (frame.hasTrackingParent());
			if (frame.getTrackingParent() != activeKeyFrame) {
				System.out.printf("WARNING: updating frame %d with %d,"
						+ " which was tracked on a different frame (%d)."
						+ "\nWhile this should work, it is not recommended.",
						activeKeyFrame.id(), frame.id(), frame
								.getTrackingParent().id());
			}

			SIM3 refToKf;
			// Get SIM3 from frame to keyframe
			if (frame.pose.trackingParent.frameID == activeKeyFrame.id()) {
				refToKf = frame.pose.thisToParent_raw;
			} else {
				refToKf = activeKeyFrame.getScaledCamToWorld().inverse()
						.mul(frame.getScaledCamToWorld());
			}

			// prepare frame for stereo with keyframe, SE3, K, level
			frame.prepareForStereoWith(activeKeyFrame, refToKf, 0);

			while ((int) referenceFrameByID.size() + referenceFrameByID_offset <= frame
					.id()) {
				referenceFrameByID.add(frame);
			}
		}

		// *** OBSERVE DEPTH HERE
		observeDepth();

		// Regularize, fill holes?
		regularizeDepthMapFillHoles();

		// Regularize
		regularizeDepthMap(false, Constants.VAL_SUM_MIN_FOR_KEEP);

		// Update depth in keyframe
		if (!activeKeyFrame.depthHasBeenUpdatedFlag) {
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
		// TODO: make multithreaded
		observeDepthRow(3, height - 3);
	}

	/**
	 * ObserveDepth for specified rows
	 * 
	 * @param yMin
	 * @param yMax
	 */
	void observeDepthRow(int yMin, int yMax) {
		float[] keyFrameMaxGradBuf = activeKeyFrame.imageGradientMaxArrayLvl[0];

		int successes = 0;

		// For each row assigned
		for (int y = yMin; y < yMax; y++) {
			// For x 3 to width-3
			for (int x = 3; x < width - 3; x++) {
				// For each pixel

				int idx = x + y * width;
				DepthMapPixelHypothesis target = currentDepthMap[idx];
				boolean hasHypothesis = target.isValid;

				// ======== 1. check absolute grad =========
				if (hasHypothesis
						&& keyFrameMaxGradBuf[idx] < Constants.MIN_ABS_GRAD_DECREASE) {
					target.isValid = false;
					continue;
				}

				if (keyFrameMaxGradBuf[idx] < Constants.MIN_ABS_GRAD_CREATE
						|| target.blacklisted < Constants.MIN_BLACKLIST) {
					continue;
				}

				// Gradient is significant, pixel not blacklisted
				boolean success = false;
				if (!hasHypothesis) {
					// First time
					success = observeDepthCreate(x, y, idx);
				} else {
					// ***Observe depth***
					success = observeDepthUpdate(x, y, idx, keyFrameMaxGradBuf);
				}
				if (success) {
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
		Frame refFrame = activeKeyFrameIsReactivated ? newest_referenceFrame
				: oldest_referenceFrame;

		// Frame tracked against activeKeyFrame?
		if (refFrame.getTrackingParent() == activeKeyFrame) {
			/*
			 * //TODO: implement this boolean wasGoodDuringTracking =
			 * refFrame.refPixelWasGoodNoCreate();
			 * 
			 * // Check if pixel is good during tracking?
			 * if(wasGoodDuringTracking != false && !wasGoodDuringTracking[ (x
			 * >> Constants.SE3TRACKING_MIN_LEVEL) + (width >>
			 * Constants.SE3TRACKING_MIN_LEVEL)*(y >>
			 * Constants.SE3TRACKING_MIN_LEVEL)]) { return false; }
			 */
		}

		// Get epipolar line??
		float epx, epy;
		// x, y pixel coordinate, refFrame
		float[] epl = makeAndCheckEPL(x, y, refFrame);

		if (epl == null) {
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
		float[] lineStereoResult = doLineStereo(new_u, new_v, epx, epy, 0.0f,
				1.0f, 1.0f / Constants.MIN_DEPTH, refFrame,
				refFrame.imageArrayLvl[0], result_idepth, result_var,
				result_eplLength);

		float error = lineStereoResult[0];
		result_idepth = lineStereoResult[1];
		result_var = lineStereoResult[2];
		result_eplLength = lineStereoResult[3];

		if (error == -3 || error == -2) {
			target.blacklisted--;
		}

		if (error < 0 || result_var > Constants.MAX_VAR) {
			return false;
		}

		result_idepth = (float) UNZERO(result_idepth);

		// add hypothesis
		// Set/change the hypothesis
		target = new DepthMapPixelHypothesis(result_idepth, result_var,
				Constants.VALIDITY_COUNTER_INITIAL_OBSERVE);

		return true;
	}

	boolean observeDepthUpdate(int x, int y, int idx, float[] keyFrameMaxGradBuf) {
		DepthMapPixelHypothesis target = currentDepthMap[idx];
		Frame refFrame;

		// Keyframe was not used before?
		if (!activeKeyFrameIsReactivated) {
			if ((int) target.nextStereoFrameMinID - referenceFrameByID_offset >= (int) referenceFrameByID
					.size()) {
				return false;
			}
			// Set refFrame to oldest_referenceFrame or some other frame?
			if ((int) target.nextStereoFrameMinID - referenceFrameByID_offset < 0) {
				refFrame = oldest_referenceFrame;
			} else {
				refFrame = referenceFrameByID
						.get((int) target.nextStereoFrameMinID
								- referenceFrameByID_offset);
			}
		} else {
			refFrame = newest_referenceFrame;
		}

		if (refFrame.getTrackingParent() == activeKeyFrame) {
			boolean[] wasGoodDuringTracking = refFrame
					.refPixelWasGoodNoCreate();

			if (wasGoodDuringTracking != null
					&& !wasGoodDuringTracking[(x >> Constants.SE3TRACKING_MIN_LEVEL)
							+ (width >> Constants.SE3TRACKING_MIN_LEVEL)
							* (y >> Constants.SE3TRACKING_MIN_LEVEL)]) {
				return false;
			}
		}

		// Get epipolar line
		float epx, epy;
		// x, y pixel coordinate, refFrame
		float[] epl = makeAndCheckEPL(x, y, refFrame);

		if (epl == null) {
			return false;
		} else {
			epx = epl[0];
			epy = epl[1];
		}

		// Limits for search?
		// which exact point to track, and where from.
		float sv = (float) Math.sqrt(target.idepth_var_smoothed);
		float min_idepth = target.idepth_smoothed - sv
				* Constants.STEREO_EPL_VAR_FAC;
		float max_idepth = target.idepth_smoothed + sv
				* Constants.STEREO_EPL_VAR_FAC;
		if (min_idepth < 0)
			min_idepth = 0;
		if (max_idepth > 1f / Constants.MIN_DEPTH)
			max_idepth = 1f / Constants.MIN_DEPTH;

		float result_idepth = 0;
		float result_var = 0;
		float result_eplLength = 0;

		// Do stereo
		float[] lineStereoResult = doLineStereo(x, y, epx, epy, min_idepth,
				target.idepth_smoothed, max_idepth, refFrame,
				refFrame.imageArrayLvl[0], result_idepth, result_var,
				result_eplLength);

		float error = lineStereoResult[0];
		result_idepth = lineStereoResult[1];
		result_var = lineStereoResult[2];
		result_eplLength = lineStereoResult[3];

		float diff = result_idepth - target.idepth_smoothed;

		// if oob: (really out of bounds)
		if (error == -1) {
			// do nothing, pixel got oob, but is still in bounds in original. I
			// will want to try again.
			return false;
		} else if (error == -2) {
			// if just not good for stereo (e.g. some inf / nan occured; has
			// inconsistent minimum; ..)

			target.validity_counter -= Constants.VALIDITY_COUNTER_DEC;
			if (target.validity_counter < 0)
				target.validity_counter = 0;

			target.nextStereoFrameMinID = 0;

			target.idepth_var *= FAIL_VAR_INC_FAC;
			if (target.idepth_var > Constants.MAX_VAR) {
				target.isValid = false;
				target.blacklisted--;
			}
			return false;
		} else if (error == -3) {
			// if not found (error too high)
			return false;
		} else if (error == -4) {
			return false;
		} else if (Constants.DIFF_FAC_OBSERVE * diff * diff > result_var
				+ target.idepth_var_smoothed) {
			// if inconsistent
			target.idepth_var *= FAIL_VAR_INC_FAC;
			if (target.idepth_var > Constants.MAX_VAR)
				target.isValid = false;

			return false;
		} else {
			// one more successful observation!

			// do textbook ekf update:
			// increase var by a little (prediction-uncertainty)
			float id_var = target.idepth_var * SUCC_VAR_INC_FAC;

			// update var with observation
			float w = result_var / (result_var + id_var);
			float new_idepth = (1.0f - w) * result_idepth + w * target.idepth;

			target.idepth = (float) UNZERO(new_idepth);

			// variance can only decrease from observation; never increase.
			id_var = id_var * w;
			if (id_var < target.idepth_var) {
				target.idepth_var = id_var;
			}

			// increase validity!
			target.validity_counter += Constants.VALIDITY_COUNTER_INC;
			float absGrad = keyFrameMaxGradBuf[idx];
			if (target.validity_counter > Constants.VALIDITY_COUNTER_MAX
					+ absGrad * (Constants.VALIDITY_COUNTER_MAX_VARIABLE)
					/ 255.0f)
				target.validity_counter = (int) (Constants.VALIDITY_COUNTER_MAX + absGrad
						* (Constants.VALIDITY_COUNTER_MAX_VARIABLE) / 255.0f);

			// increase Skip!
			if (result_eplLength < Constants.MIN_EPL_LENGTH_CROP) {
				float inc = activeKeyFrame.numFramesTrackedOnThis
						/ (float) (activeKeyFrame.numMappedOnThis + 5);
				if (inc < 3)
					inc = 3;

				inc += ((int) (result_eplLength * 10000) % 2);

				if (result_eplLength < 0.5 * Constants.MIN_EPL_LENGTH_CROP)
					inc *= 3;

				target.nextStereoFrameMinID = refFrame.id() + inc;
			}

			return true;
		}
	}

	/**
	 * 
	 * Return null if failed, return float[] {epx, epy} if found.
	 * */
	public float[] makeAndCheckEPL(int x, int y, Frame ref) {
		int idx = x + y * width;

		double fx = Constants.fx[0];
		double fy = Constants.fy[0];
		double cx = Constants.cx[0];
		double cy = Constants.cy[0];

		// ======= make epl ========
		// calculate the plane spanned by the two camera centers and the point
		// (x,y,1)
		// intersect it with the keyframe's image plane (at depth=1)
		double epx = -fx * ref.thisToOther_t.get(0, 0)
				+ ref.thisToOther_t.get(2, 0) * (x - cx);
		double epy = -fy * ref.thisToOther_t.get(1, 0)
				+ ref.thisToOther_t.get(2, 0) * (y - cy);

		if (Double.isNaN(epx + epy)) {
			return null;
		}

		// ======== check epl length =========
		float eplLengthSquared = (float) (epx * epx + epy * epy);
		if (eplLengthSquared < Constants.MIN_EPL_LENGTH_SQUARED) {
			return null;
		}

		// ===== check epl-grad magnitude ======

		float gx = activeKeyFrame.imageArrayLvl[0][idx + 1]
				- activeKeyFrame.imageArrayLvl[0][idx - 1];
		float gy = activeKeyFrame.imageArrayLvl[0][idx + width]
				- activeKeyFrame.imageArrayLvl[0][idx - width];
		float eplGradSquared = (float) (gx * epx + gy * epy);
		eplGradSquared = eplGradSquared * eplGradSquared / eplLengthSquared; // square
																				// and
																				// norm
																				// with
																				// epl-length

		if (eplGradSquared < Constants.MIN_EPL_GRAD_SQUARED) {
			return null;
		}

		// ===== check epl-grad angle ======
		if (eplGradSquared / (gx * gx + gy * gy) < Constants.MIN_EPL_ANGLE_SQUARED) {
			return null;
		}

		// ===== DONE - return "normalized" epl =====
		float fac = (float) (Constants.GRADIENT_SAMPLE_DIST / Math
				.sqrt(eplLengthSquared));

		float pepx = (float) (epx * fac);
		float pepy = (float) (epy * fac);

		return new float[] { pepx, pepy };
	}

	// find pixel in image (do stereo along epipolar line).
	// mat: NEW image
	// KinvP: point in OLD image (Kinv * (u_old, v_old, 1)), projected
	// trafo: x_old = trafo * x_new; (from new to old image)
	// realVal: descriptor in OLD image.
	// returns: result_idepth : point depth in new camera's coordinate system
	// returns: result_u/v : point's coordinates in new camera's coordinate
	// system
	// returns: idepth_var: (approximated) measurement variance of inverse depth
	// of result_point_NEW
	// returns error if sucessful; -1 if out of bounds, -2 if not found.

	/**
	 * Returns float[4] array, {error, result_idepth, result_var,
	 * result_eplLength}
	 */
	float[] doLineStereo(float u, float v, float epxn, float epyn,
			float min_idepth, float prior_idepth, float max_idepth,
			Frame referenceFrame, float[] referenceFrameImage,
			float result_idepth, float result_var, float result_eplLength) {

		// calculate epipolar line start and end point in old image
		jeigen.DenseMatrix KinvP = Vec.array3ToVec(new double[] {
				fxi * u + cxi, fyi * v + cyi, 1.0f });
		jeigen.DenseMatrix pInf = referenceFrame.K_otherToThis_R.mmul(KinvP);
		jeigen.DenseMatrix pReal = pInf.div(prior_idepth).add(
				referenceFrame.K_otherToThis_t);

		float rescaleFactor = ((float) pReal.get(2, 0) * prior_idepth);

		// Start/end points of epipolar line on old image?
		float firstX = u - 2 * epxn * rescaleFactor;
		float firstY = v - 2 * epyn * rescaleFactor;
		float lastX = u + 2 * epxn * rescaleFactor;
		float lastY = v + 2 * epyn * rescaleFactor;
		// width - 2 and height - 2 comes from the one-sided gradient
		// calculation at the bottom
		if (firstX <= 0 || firstX >= width - 2 || firstY <= 0
				|| firstY >= height - 2 || lastX <= 0 || lastX >= width - 2
				|| lastY <= 0 || lastY >= height - 2) {
			return new float[] { -1, result_idepth, result_var,
					result_eplLength };
		}

		if (!(rescaleFactor > 0.7f && rescaleFactor < 1.4f)) {
			return new float[] { -1, result_idepth, result_var,
					result_eplLength };
		}

		float[] activeKeyFrameImageData = activeKeyFrame.imageArrayLvl[0];

		// calculate values to search for
		float realVal_p1 = Utils.interpolatedValue(activeKeyFrameImageData, u
				+ epxn * rescaleFactor, v + epyn * rescaleFactor, width);
		float realVal_m1 = Utils.interpolatedValue(activeKeyFrameImageData, u
				- epxn * rescaleFactor, v - epyn * rescaleFactor, width);
		float realVal = Utils.interpolatedValue(activeKeyFrameImageData, u, v,
				width);
		float realVal_m2 = Utils
				.interpolatedValue(activeKeyFrameImageData, u - 2 * epxn
						* rescaleFactor, v - 2 * epyn * rescaleFactor, width);
		float realVal_p2 = Utils
				.interpolatedValue(activeKeyFrameImageData, u + 2 * epxn
						* rescaleFactor, v + 2 * epyn * rescaleFactor, width);

		jeigen.DenseMatrix pClose = pInf.add(referenceFrame.K_otherToThis_t
				.mul(max_idepth));
		// if the assumed close-point lies behind the
		// image, have to change that.
		if (pClose.get(2, 0) < 0.001f) {
			max_idepth = (float) ((0.001f - pInf.get(2, 0)) / referenceFrame.K_otherToThis_t
					.get(2, 0));
			pClose = referenceFrame.K_otherToThis_t.mul(max_idepth).add(pInf);
		}
		pClose = pClose.div(pClose.get(2, 0)); // pos in new image of point
												// (xy), assuming max_idepth

		jeigen.DenseMatrix pFar = pInf.add(referenceFrame.K_otherToThis_t
				.mul(min_idepth));
		// if the assumed far-point lies behind the image or closter than the
		// near-point,
		// we moved past the Point it and should stop.
		if (pFar.get(2, 0) < 0.001f || max_idepth < min_idepth) {
			return new float[] { -1, result_idepth, result_var,
					result_eplLength };
		}
		pFar = pFar.div(pFar.get(2, 0)); // pos in new image of point (xy),
											// assuming min_idepth

		// check for nan due to eg division by zero.
		if (Float.isNaN((float) (pFar.get(0, 0) + pClose.get(0, 0)))) {
			return new float[] { -4, result_idepth, result_var,
					result_eplLength };
		}

		// calculate increments in which we will step through the epipolar line.
		// they are sampleDist (or half sample dist) long
		float incx = (float) (pClose.get(0, 0) - pFar.get(0, 0));
		float incy = (float) (pClose.get(1, 0) - pFar.get(1, 0));
		float eplLength = (float) Math.sqrt(incx * incx + incy * incy);
		if (!(eplLength > 0) || Float.isInfinite(eplLength)) {
			return new float[] { -4, result_idepth, result_var,
					result_eplLength };
		}

		if (eplLength > Constants.MAX_EPL_LENGTH_CROP) {
			pClose.set(0, 0, pFar.get(0, 0) + incx
					* Constants.MAX_EPL_LENGTH_CROP / eplLength);
			pClose.set(1, 0, pFar.get(1, 0) + incy
					* Constants.MAX_EPL_LENGTH_CROP / eplLength);
		}

		incx *= Constants.GRADIENT_SAMPLE_DIST / eplLength;
		incy *= Constants.GRADIENT_SAMPLE_DIST / eplLength;

		// extend one sample_dist to left & right.
		pFar.set(0, 0, pFar.get(0, 0) - incx);
		pFar.set(1, 0, pFar.get(1, 0) - incy);
		pClose.set(0, 0, pClose.get(0, 0) + incx);
		pClose.set(1, 0, pClose.get(1, 0) + incy);

		// make epl long enough (pad a little bit).
		if (eplLength < Constants.MIN_EPL_LENGTH_CROP) {
			float pad = (Constants.MIN_EPL_LENGTH_CROP - (eplLength)) / 2.0f;
			pFar.set(0, 0, pFar.get(0, 0) - incx * pad);
			pFar.set(1, 0, pFar.get(1, 0) - incy * pad);

			pClose.set(0, 0, pClose.get(0, 0) + incx * pad);
			pClose.set(1, 0, pClose.get(1, 0) + incy * pad);
		}

		// if inf point is outside of image: skip pixel.
		if (pFar.get(0, 0) <= Constants.SAMPLE_POINT_TO_BORDER
				|| pFar.get(0, 0) >= width - Constants.SAMPLE_POINT_TO_BORDER
				|| pFar.get(1, 0) <= Constants.SAMPLE_POINT_TO_BORDER
				|| pFar.get(1, 0) >= height - Constants.SAMPLE_POINT_TO_BORDER) {
			return new float[] { -1, result_idepth, result_var,
					result_eplLength };
		}

		// if near point is outside: move inside, and test length again.
		if (pClose.get(0, 0) <= Constants.SAMPLE_POINT_TO_BORDER
				|| pClose.get(0, 0) >= width - Constants.SAMPLE_POINT_TO_BORDER
				|| pClose.get(1, 0) <= Constants.SAMPLE_POINT_TO_BORDER
				|| pClose.get(1, 0) >= height
						- Constants.SAMPLE_POINT_TO_BORDER) {
			if (pClose.get(0, 0) <= Constants.SAMPLE_POINT_TO_BORDER) {
				float toAdd = (float) ((Constants.SAMPLE_POINT_TO_BORDER - pClose
						.get(0, 0)) / incx);

				pClose.set(0, 0, pClose.get(0, 0) + incx * toAdd);
				pClose.set(1, 0, pClose.get(1, 0) + incy * toAdd);
			} else if (pClose.get(0, 0) >= width
					- Constants.SAMPLE_POINT_TO_BORDER) {
				float toAdd = (float) ((width
						- Constants.SAMPLE_POINT_TO_BORDER - pClose.get(0, 0)) / incx);

				pClose.set(0, 0, pClose.get(0, 0) + incx * toAdd);
				pClose.set(1, 0, pClose.get(1, 0) + incy * toAdd);
			}

			if (pClose.get(1, 0) <= Constants.SAMPLE_POINT_TO_BORDER) {
				float toAdd = (float) ((Constants.SAMPLE_POINT_TO_BORDER - pClose
						.get(1, 0)) / incy);

				pClose.set(0, 0, pClose.get(0, 0) + incx * toAdd);
				pClose.set(1, 0, pClose.get(1, 0) + incy * toAdd);
			} else if (pClose.get(1, 0) >= height
					- Constants.SAMPLE_POINT_TO_BORDER) {
				float toAdd = (float) ((height
						- Constants.SAMPLE_POINT_TO_BORDER - pClose.get(1, 0)) / incy);

				pClose.set(0, 0, pClose.get(0, 0) + incx * toAdd);
				pClose.set(1, 0, pClose.get(1, 0) + incy * toAdd);
			}

			// get new epl length
			float fincx = (float) (pClose.get(0, 0) - pFar.get(0, 0));
			float fincy = (float) (pClose.get(1, 0) - pFar.get(1, 0));
			float newEplLength = (float) Math.sqrt(fincx * fincx + fincy
					* fincy);

			// test again
			if (pClose.get(0, 0) <= Constants.SAMPLE_POINT_TO_BORDER
					|| pClose.get(0, 0) >= width
							- Constants.SAMPLE_POINT_TO_BORDER
					|| pClose.get(1, 0) <= Constants.SAMPLE_POINT_TO_BORDER
					|| pClose.get(1, 0) >= height
							- Constants.SAMPLE_POINT_TO_BORDER
					|| newEplLength < 8.0f) {
				return new float[] { -1, result_idepth, result_var,
						result_eplLength };
			}

		}

		// from here on:
		// - pInf: search start-point
		// - p0: search end-point
		// - incx, incy: search steps in pixel
		// - eplLength, min_idepth, max_idepth: determines search-resolution,
		// i.e. the result's variance.

		float cpx = (float) pFar.get(0, 0);
		float cpy = (float) pFar.get(1, 0);

		float val_cp_m2 = Utils.interpolatedValue(referenceFrameImage, cpx
				- 2.0f * incx, cpy - 2.0f * incy, width);
		float val_cp_m1 = Utils.interpolatedValue(referenceFrameImage, cpx
				- incx, cpy - incy, width);
		float val_cp = Utils.interpolatedValue(referenceFrameImage, cpx, cpy,
				width);
		float val_cp_p1 = Utils.interpolatedValue(referenceFrameImage, cpx
				+ incx, cpy + incy, width);
		float val_cp_p2;

		/*
		 * Subsequent exact minimum is found the following way: - assuming lin.
		 * interpolation, the gradient of Error at p1 (towards p2) is given by
		 * dE1 = -2sum(e1*e1 - e1*e2) where e1 and e2 are summed over, and are
		 * the residuals (not squared).
		 * 
		 * - the gradient at p2 (coming from p1) is given by dE2 = +2sum(e2*e2 -
		 * e1*e2)
		 * 
		 * - linear interpolation => gradient changes linearely; zero-crossing
		 * is hence given by p1 + d*(p2-p1) with d = -dE1 / (-dE1 + dE2).
		 * 
		 * 
		 * 
		 * => I for later exact min calculation, I need
		 * sum(e_i*e_i),sum(e_{i-1}*e_{i-1}),sum(e_{i+1}*e_{i+1}) and sum(e_i *
		 * e_{i-1}) and sum(e_i * e_{i+1}), where i is the respective winning
		 * index.
		 */

		// walk in equally sized steps, starting at depth=infinity.
		int loopCounter = 0;
		float best_match_x = -1;
		float best_match_y = -1;
		float best_match_err = (float) 1e50;
		float second_best_match_err = (float) 1e50;

		// best pre and post errors.
		float best_match_errPre = Float.NaN;
		float best_match_errPost = Float.NaN;
		float best_match_DiffErrPre = Float.NaN;
		float best_match_DiffErrPost = Float.NaN;
		boolean bestWasLastLoop = false;

		float eeLast = -1; // final error of last comp.

		// alternating intermediate vars
		float e1A = Float.NaN, e1B = Float.NaN;
		float e2A = Float.NaN, e2B = Float.NaN;
		float e3A = Float.NaN, e3B = Float.NaN;
		float e4A = Float.NaN, e4B = Float.NaN;
		float e5A = Float.NaN, e5B = Float.NaN;

		int loopCBest = -1, loopCSecond = -1;
		while (((incx < 0) == (cpx > pClose.get(0, 0)) && (incy < 0) == (cpy > pClose
				.get(1, 0))) || loopCounter == 0) {
			// interpolate one new point
			val_cp_p2 = Utils.interpolatedValue(referenceFrameImage, cpx + 2
					* incx, cpy + 2 * incy, width);

			// hacky but fast way to get error and differential error: switch
			// buffer variables for last loop.
			float ee = 0;
			if (loopCounter % 2 == 0) {
				// calc error and accumulate sums.
				e1A = val_cp_p2 - realVal_p2;
				ee += e1A * e1A;
				e2A = val_cp_p1 - realVal_p1;
				ee += e2A * e2A;
				e3A = val_cp - realVal;
				ee += e3A * e3A;
				e4A = val_cp_m1 - realVal_m1;
				ee += e4A * e4A;
				e5A = val_cp_m2 - realVal_m2;
				ee += e5A * e5A;
			} else {
				// calc error and accumulate sums.
				e1B = val_cp_p2 - realVal_p2;
				ee += e1B * e1B;
				e2B = val_cp_p1 - realVal_p1;
				ee += e2B * e2B;
				e3B = val_cp - realVal;
				ee += e3B * e3B;
				e4B = val_cp_m1 - realVal_m1;
				ee += e4B * e4B;
				e5B = val_cp_m2 - realVal_m2;
				ee += e5B * e5B;
			}

			// do I have a new winner??
			// if so: set.
			if (ee < best_match_err) {
				// put to second-best
				second_best_match_err = best_match_err;
				loopCSecond = loopCBest;

				// set best.
				best_match_err = ee;
				loopCBest = loopCounter;

				best_match_errPre = eeLast;
				best_match_DiffErrPre = e1A * e1B + e2A * e2B + e3A * e3B + e4A
						* e4B + e5A * e5B;
				best_match_errPost = -1;
				best_match_DiffErrPost = -1;

				best_match_x = cpx;
				best_match_y = cpy;
				bestWasLastLoop = true;

			}
			// otherwise: the last might be the current winner, in which case i
			// have to save these values.
			else {
				if (bestWasLastLoop) {
					best_match_errPost = ee;
					best_match_DiffErrPost = e1A * e1B + e2A * e2B + e3A * e3B
							+ e4A * e4B + e5A * e5B;
					bestWasLastLoop = false;
				}

				// collect second-best:
				// just take the best of all that are NOT equal to current best.
				if (ee < second_best_match_err) {
					second_best_match_err = ee;
					loopCSecond = loopCounter;
				}
			}

			// shift everything one further.
			eeLast = ee;
			val_cp_m2 = val_cp_m1;
			val_cp_m1 = val_cp;
			val_cp = val_cp_p1;
			val_cp_p1 = val_cp_p2;

			cpx += incx;
			cpy += incy;

			loopCounter++;
		}

		// if error too big, will return -3, otherwise -2.
		if (best_match_err > 4.0f * (float) Constants.MAX_ERROR_STEREO) {
			return new float[] { -3, result_idepth, result_var,
					result_eplLength };
		}

		// check if clear enough winner
		if (Math.abs(loopCBest - loopCSecond) > 1.0f
				&& Constants.MIN_DISTANCE_ERROR_STEREO * best_match_err > second_best_match_err) {
			return new float[] { -2, result_idepth, result_var,
					result_eplLength };
		}

		boolean didSubpixel = false;
		if (Constants.useSubpixelStereo) {
			// ================== compute exact match =========================
			// compute gradients (they are actually only half the real gradient)
			float gradPre_pre = -(best_match_errPre - best_match_DiffErrPre);
			float gradPre_this = (best_match_err - best_match_DiffErrPre);
			float gradPost_this = -(best_match_err - best_match_DiffErrPost);
			float gradPost_post = (best_match_errPost - best_match_DiffErrPost);

			// final decisions here.
			boolean interpPost = false;
			boolean interpPre = false;

			// if one is oob: return false.
			if (/* enablePrintDebugInfo && */(best_match_errPre < 0 || best_match_errPost < 0)) {
				// stats->num_stereo_invalid_atEnd++;
			}
			// - if zero-crossing occurs exactly in between (gradient
			// Inconsistent),
			else if ((gradPost_this < 0) ^ (gradPre_this < 0)) {
				// return exact pos, if both central gradients are small
				// compared to their counterpart.
				/*
				 * if(enablePrintDebugInfo && (gradPost_this*gradPost_this >
				 * 0.1f*0.1f*gradPost_post*gradPost_post ||
				 * gradPre_this*gradPre_this >
				 * 0.1f*0.1f*gradPre_pre*gradPre_pre))
				 * stats->num_stereo_invalid_inexistantCrossing++;
				 */
			}

			// if pre has zero-crossing
			else if ((gradPre_pre < 0) ^ (gradPre_this < 0)) {
				// if post has zero-crossing
				if ((gradPost_post < 0) ^ (gradPost_this < 0)) {
					// if(enablePrintDebugInfo)
					// stats->num_stereo_invalid_twoCrossing++;
				} else
					interpPre = true;
			}

			// if post has zero-crossing
			else if ((gradPost_post < 0) ^ (gradPost_this < 0)) {
				interpPost = true;
			}

			// if none has zero-crossing
			else {
				// if(enablePrintDebugInfo)
				// stats->num_stereo_invalid_noCrossing++;
			}

			// DO interpolation!
			// minimum occurs at zero-crossing of gradient, which is a straight
			// line => easy to compute.
			// the error at that point is also computed by just integrating.
			if (interpPre) {
				float d = gradPre_this / (gradPre_this - gradPre_pre);
				best_match_x -= d * incx;
				best_match_y -= d * incy;
				best_match_err = best_match_err - 2 * d * gradPre_this
						- (gradPre_pre - gradPre_this) * d * d;
				// if(enablePrintDebugInfo) stats->num_stereo_interpPre++;
				didSubpixel = true;

			} else if (interpPost) {
				float d = gradPost_this / (gradPost_this - gradPost_post);
				best_match_x += d * incx;
				best_match_y += d * incy;
				best_match_err = best_match_err + 2 * d * gradPost_this
						+ (gradPost_post - gradPost_this) * d * d;
				// if(enablePrintDebugInfo) stats->num_stereo_interpPost++;
				didSubpixel = true;
			} else {
				// if(enablePrintDebugInfo) stats->num_stereo_interpNone++;
			}
		}

		// sampleDist is the distance in pixel at which the realVal's were
		// sampled
		float sampleDist = Constants.GRADIENT_SAMPLE_DIST * rescaleFactor;

		float gradAlongLine = 0;
		float tmp = realVal_p2 - realVal_p1;
		gradAlongLine += tmp * tmp;
		tmp = realVal_p1 - realVal;
		gradAlongLine += tmp * tmp;
		tmp = realVal - realVal_m1;
		gradAlongLine += tmp * tmp;
		tmp = realVal_m1 - realVal_m2;
		gradAlongLine += tmp * tmp;

		gradAlongLine /= sampleDist * sampleDist;

		// check if interpolated error is OK. use evil hack to allow more error
		// if there is a lot of gradient.
		if (best_match_err > (float) Constants.MAX_ERROR_STEREO
				+ Math.sqrt(gradAlongLine) * 20) {
			return new float[] { -3, result_idepth, result_var,
					result_eplLength };
		}

		// ================= calc depth (in KF) ====================
		// * KinvP = Kinv * (x,y,1); where x,y are pixel coordinates of point we
		// search for, in the KF.
		// * best_match_x = x-coordinate of found correspondence in the
		// reference frame.

		float idnew_best_match; // depth in the new image
		float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed
						// inverse depth derived by the pixel-disparity.
		if (incx * incx > incy * incy) {
			float oldX = fxi * best_match_x + cxi;
			float nominator = (float) (oldX
					* referenceFrame.otherToThis_t.get(2, 0) - referenceFrame.otherToThis_t
					.get(0, 0));

			float dot0 = (float) Vec.dot(Vec.vec3ToArray(KinvP),
					Vec.vec3ToArray(referenceFrame.otherToThis_R_row0));
			float dot2 = (float) Vec.dot(Vec.vec3ToArray(KinvP),
					Vec.vec3ToArray(referenceFrame.otherToThis_R_row2));

			idnew_best_match = (dot0 - oldX * dot2) / nominator;
			alpha = (float) (incx
					* fxi
					* (dot0 * referenceFrame.otherToThis_t.get(2, 0) - dot2
							* referenceFrame.otherToThis_t.get(0, 0)) / (nominator * nominator));

		} else {
			float oldY = fyi * best_match_y + cyi;

			float nominator = (float) (oldY
					* referenceFrame.otherToThis_t.get(2, 0) - referenceFrame.otherToThis_t
					.get(1, 0));

			float dot1 = (float) Vec.dot(Vec.vec3ToArray(KinvP),
					Vec.vec3ToArray(referenceFrame.otherToThis_R_row1));
			float dot2 = (float) Vec.dot(Vec.vec3ToArray(KinvP),
					Vec.vec3ToArray(referenceFrame.otherToThis_R_row2));

			idnew_best_match = (dot1 - oldY * dot2) / nominator;
			alpha = (float) (incy
					* fyi
					* (dot1 * referenceFrame.otherToThis_t.get(2, 0) - dot2
							* referenceFrame.otherToThis_t.get(1, 0)) / (nominator * nominator));

		}

		boolean allowNegativeIdepths = true;
		if (idnew_best_match < 0) {
			if (!allowNegativeIdepths)
				return new float[] { -2, result_idepth, result_var,
						result_eplLength };
		}

		// ================= calc var (in NEW image) ====================

		// TODO: setting
		float cameraPixelNoise2 = 4 * 4;

		// calculate error from photometric noise
		float photoDispError = 4.0f * cameraPixelNoise2
				/ (gradAlongLine + Constants.DIVISION_EPS);

		float trackingErrorFac = 0.25f * (1.0f + referenceFrame.initialTrackedResidual);

		// calculate error from geometric noise (wrong camera pose /
		// calibration)
		float[] gradsInterp = {
				Utils.interpolatedValue(
						activeKeyFrame.imageGradientXArrayLvl[0], u, v, width),
				Utils.interpolatedValue(
						activeKeyFrame.imageGradientYArrayLvl[0], u, v, width) };

		float geoDispError = (gradsInterp[0] * epxn + gradsInterp[1] * epyn)
				+ Constants.DIVISION_EPS;
		geoDispError = trackingErrorFac
				* trackingErrorFac
				* (gradsInterp[0] * gradsInterp[0] + gradsInterp[1]
						* gradsInterp[1]) / (geoDispError * geoDispError);

		// geoDispError *= (0.5 + 0.5 *result_idepth) * (0.5 + 0.5
		// *result_idepth);

		// final error consists of a small constant part (discretization error),
		// geometric and photometric error.
		result_var = alpha
				* alpha
				* ((didSubpixel ? 0.05f : 0.5f) * sampleDist * sampleDist
						+ geoDispError + photoDispError); // square to make
															// variance

		result_idepth = idnew_best_match;

		result_eplLength = eplLength;

		return new float[] { best_match_err, result_idepth, result_var,
				result_eplLength };
	}

	void regularizeDepthMapFillHoles() {

		buildRegIntegralBuffer();

		copyDepthMapArray();

		// TOOD: multithread
		regularizeDepthMapFillHolesRow(3, height - 2);

	}

	// Summed area table of number of valid DepthMapPixelHypothesis
	void buildRegIntegralBuffer() {

		// Sum horizontally
		// TODO: run in parallel
		buildRegIntegralBufferRow1(0, height);

		// Sum vertically
		int wh = height * width;
		for (int i = width; i < wh; i++) {
			validityIntegralBuffer[i] += validityIntegralBuffer[i - width];
		}

	}

	void buildRegIntegralBufferRow1(int yMin, int yMax) {
		for (int y = yMin; y < yMax; y++) {
			int validityIntegralBufferSUM = 0;
			for (int x = 0; x < width; x++) {
				int idx = y * width + x;
				DepthMapPixelHypothesis src = currentDepthMap[idx];

				if (src.isValid)
					validityIntegralBufferSUM += src.validity_counter;

				// Sum/number of valid DepthMapPixelHypothesis in same row,
				// before the pixel?
				validityIntegralBuffer[idx] = validityIntegralBufferSUM;
			}
		}
	}

	void regularizeDepthMapFillHolesRow(int yMin, int yMax) {
		// =========== regularize fill holes
		float[] keyFrameMaxGradBuf = activeKeyFrame.imageGradientMaxArrayLvl[0];

		// For rows yMin to yMax
		for (int y = yMin; y < yMax; y++) {
			// For pixels (3 to width-2) in row
			for (int x = 3; x < width - 2; x++) {
				int idx = x + y * width;
				// Get hypothessis
				DepthMapPixelHypothesis dest = otherDepthMap[idx];

				if (dest.isValid)
					continue;
				if (keyFrameMaxGradBuf[idx] < Constants.MIN_ABS_GRAD_DECREASE)
					continue;

				// Number of valid pixels in some neighbourhood??
				int[] io = validityIntegralBuffer;
				int val = io[idx + 2 + (2 * width)] - io[idx + 2 - (3 * width)]
						- io[idx + -3 + (2 * width)]
						+ io[idx + -3 - (3 * width)];

				// If blacklisted and surrounding has some number of valid
				// pixels
				if ((dest.blacklisted >= Constants.MIN_BLACKLIST && val > Constants.VAL_SUM_MIN_FOR_CREATE)
						|| val > Constants.VAL_SUM_MIN_FOR_UNBLACKLIST) {

					// Calculate average idepth?

					float sumIdepthObs = 0, sumIVarObs = 0;
					int s1max = (x - 2) + (y + 3) * width;
					for (int s1 = (x - 2) + (y - 2) * width; s1 < s1max; s1 += width) {
						for (int s2 = s1; s2 < s1 + 5; s2++) {
							DepthMapPixelHypothesis source = otherDepthMap[s2];
							if (!source.isValid)
								continue;

							sumIdepthObs += source.idepth / source.idepth_var;
							sumIVarObs += 1.0f / source.idepth_var;
						}
					}

					float idepthObs = sumIdepthObs / sumIVarObs;
					idepthObs = (float) UNZERO(idepthObs);

					// Create new hypothesis
					currentDepthMap[idx] = new DepthMapPixelHypothesis(
							idepthObs, VAR_RANDOM_INIT_INITIAL, 0);

					// if(enablePrintDebugInfo) stats->num_reg_created++;
				}
			}
		}
	}

	void regularizeDepthMap(boolean removeOcclusions, int validityTH) {

		copyDepthMapArray();

		regularizeDepthMapRow(validityTH, 2, height - 2, removeOcclusions);

	}

	private void copyDepthMapArray() {
		for (int i = 0; i < currentDepthMap.length; i++) {
			if (currentDepthMap[i] != null) {
				otherDepthMap[i] = new DepthMapPixelHypothesis(
						currentDepthMap[i]);
			} else {
				otherDepthMap[i] = null;
			}
		}
	}

	void regularizeDepthMapRow(int validityTH, int yMin, int yMax,
			boolean removeOcclusions) {

		final int regularize_radius = 2;
		final float regDistVar = Constants.REG_DIST_VAR;

		for (int y = yMin; y < yMax; y++) {
			for (int x = regularize_radius; x < width - regularize_radius; x++) {
				int idx = x + y * width;
				DepthMapPixelHypothesis dest = currentDepthMap[idx];
				DepthMapPixelHypothesis destRead = otherDepthMap[idx];

				// if isValid need to do better examination and then update.

				// if(enablePrintDebugInfo && destRead->blacklisted <
				// MIN_BLACKLIST)
				// stats->num_reg_blacklisted++;

				if (!destRead.isValid)
					continue;

				float sum = 0, val_sum = 0, sumIvar = 0;// , min_varObs = 1e20;
				int numOccluding = 0, numNotOccluding = 0;

				for (int dx = -regularize_radius; dx <= regularize_radius; dx++) {
					for (int dy = -regularize_radius; dy <= regularize_radius; dy++) {
						DepthMapPixelHypothesis source = otherDepthMap[idx + dx
								+ dy * width];

						if (!source.isValid)
							continue;
						// stats->num_reg_total++;

						float diff = source.idepth - destRead.idepth;
						if (Constants.DIFF_FAC_SMOOTHING * diff * diff > source.idepth_var
								+ destRead.idepth_var) {
							if (removeOcclusions) {
								if (source.idepth > destRead.idepth)
									numOccluding++;
							}
							continue;
						}

						val_sum += source.validity_counter;

						if (removeOcclusions)
							numNotOccluding++;

						float distFac = (float) (dx * dx + dy * dy)
								* regDistVar;
						float ivar = 1.0f / (source.idepth_var + distFac);

						sum += source.idepth * ivar;
						sumIvar += ivar;

					}
				}

				if (val_sum < validityTH) {
					dest.isValid = false;
					// if(enablePrintDebugInfo)
					// stats->num_reg_deleted_secondary++;
					// dest->blacklisted--;

					// if(enablePrintDebugInfo) stats->num_reg_setBlacklisted++;
					continue;
				}

				if (removeOcclusions) {
					if (numOccluding > numNotOccluding) {
						dest.isValid = false;
						// if(enablePrintDebugInfo)
						// stats->num_reg_deleted_occluded++;

						continue;
					}
				}

				sum = sum / sumIvar;
				sum = (float) UNZERO(sum);

				// update!
				dest.idepth_smoothed = sum;
				dest.idepth_var_smoothed = 1.0f / sumIvar;
				
				// if(enablePrintDebugInfo) stats->num_reg_smeared++;
			}
		}
	}

	// Regularize fill holes, regularize depth map, etc
	public void finalizeKeyFrame() {
		if (!isValid()) {
			System.err.println("Assertion failed!");
		}

		regularizeDepthMapFillHoles();

		regularizeDepthMap(false, Constants.VAL_SUM_MIN_FOR_KEEP);

		activeKeyFrame.setDepth(currentDepthMap);
		// activeKeyFrame.calculateMeanInformation(); // Not doing anything.
		activeKeyFrame.takeReActivationData(currentDepthMap);

	}

	public void createKeyFrame(Frame new_keyframe) {
		assert (isValid());
		assert (new_keyframe != null);
		assert (new_keyframe.hasTrackingParent());
		
		/*
		resetCounters(); // for stats

		if(plotStereoImages)
		{
			cv::Mat keyFrameImage(new_keyframe->height(), new_keyframe->width(), CV_32F, const_cast<float*>(new_keyframe->image(0)));
			keyFrameImage.convertTo(debugImageHypothesisPropagation, CV_8UC1);
			cv::cvtColor(debugImageHypothesisPropagation, debugImageHypothesisPropagation, CV_GRAY2RGB);
		}*/

		SE3 oldToNew_SE3 = SE3.inverse(new_keyframe.pose.thisToParent_raw.getSE3());

		// TODO: something wrong here
		propagateDepth(new_keyframe);
		
		activeKeyFrame = new_keyframe;
		// float[] activeKeyFrameImageData = new_keyframe.imageArrayLvl[0];
		activeKeyFrameIsReactivated = false;
		

		regularizeDepthMap(true, Constants.VAL_SUM_MIN_FOR_KEEP);

		regularizeDepthMapFillHoles();

		regularizeDepthMap(false, Constants.VAL_SUM_MIN_FOR_KEEP);

		// make mean inverse depth be one.
		float sumIdepth = 0, numIdepth = 0;
		for (int i = 0; i < width * height; i++) {
			if (!currentDepthMap[i].isValid)
				continue;
			sumIdepth += currentDepthMap[i].idepth_smoothed;
			numIdepth++;
		}

		float rescaleFactor = numIdepth / sumIdepth;
		float rescaleFactor2 = rescaleFactor * rescaleFactor;
		for (int i = 0; i < width * height; i++) {
			if (!currentDepthMap[i].isValid)
				continue;
			currentDepthMap[i].idepth *= rescaleFactor;
			currentDepthMap[i].idepth_smoothed *= rescaleFactor;
			currentDepthMap[i].idepth_var *= rescaleFactor2;
			currentDepthMap[i].idepth_var_smoothed *= rescaleFactor2;
		}
		activeKeyFrame.pose.thisToParent_raw = new SIM3(
				SE3.inverse(oldToNew_SE3), rescaleFactor);
		

		/*
		System.out.println("createKeyFrame: thisToParent_raw: " + activeKeyFrame.pose.thisToParent_raw.toString());

		System.out.println("rescaleFactor: " + rescaleFactor);
		*/
		
		
		activeKeyFrame.pose.invalidateCache();

		// Update depth in keyframe
		activeKeyFrame.setDepth(currentDepthMap);

	}

	void propagateDepth(Frame new_keyframe)
	{


		if(new_keyframe.getTrackingParent() != activeKeyFrame)
		{
			System.out.printf("WARNING: propagating depth from frame %d to %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
					activeKeyFrame.id(), new_keyframe.id(),
					new_keyframe.getTrackingParent().id());
		}

		int n = 0;
		// wipe depthmap
		for(int i=width*height-1 ; i>=0 ; i--)
		{
			DepthMapPixelHypothesis pt = otherDepthMap[i];
			pt.isValid = false;
			pt.blacklisted = 0;
		}
		
		// re-usable values.
		SE3 oldToNew_SE3 = SE3.inverse(new_keyframe.pose.thisToParent_raw.getSE3());
		jeigen.DenseMatrix trafoInv_t = oldToNew_SE3.getTranslationMat();
		jeigen.DenseMatrix trafoInv_R = oldToNew_SE3.getRotationMat();

		boolean[] trackingWasGood = new_keyframe.getTrackingParent() == activeKeyFrame ?
				new_keyframe.refPixelWasGoodNoCreate() : null;


		float[] activeKFImageData = activeKeyFrame.imageArrayLvl[0];
		float[] newKFMaxGrad = new_keyframe.imageGradientMaxArrayLvl[0];
		float[] newKFImageData = new_keyframe.imageArrayLvl[0];



		// go through all pixels of OLD image, propagating forwards.
		for(int y=0;y<height;y++)
			for(int x=0;x<width;x++)
			{
				DepthMapPixelHypothesis source = currentDepthMap[x + y*width];

				if(!source.isValid)
					continue;

				//if(enablePrintDebugInfo) runningStats.num_prop_attempts++;


				jeigen.DenseMatrix pn = trafoInv_R.mmul(new DenseMatrix(new double[][]
						{{x*fxi + cxi},{y*fyi + cyi},{1.0}}))
						.div(source.idepth_smoothed).add(trafoInv_t);

				float new_idepth = (float) (1.0f / pn.get(2,0));

				float u_new = (float) (pn.get(0,0)*new_idepth*fx + cx);
				float v_new = (float) (pn.get(1,0)*new_idepth*fy + cy);

				// check if still within image, if not: DROP.
				if(!(u_new > 2.1f && v_new > 2.1f && u_new < width-3.1f && v_new < height-3.1f))
				{
					//if(enablePrintDebugInfo) runningStats.num_prop_removed_out_of_bounds++;
					continue;
				}

				int newIDX = (int)(u_new+0.5f) + ((int)(v_new+0.5f))*width;
				float destAbsGrad = newKFMaxGrad[newIDX];
				
				if(trackingWasGood != null)
				{
					if(!trackingWasGood[(x >> Constants.SE3TRACKING_MIN_LEVEL)
					                    + (width >> Constants.SE3TRACKING_MIN_LEVEL)
					                    *(y >> Constants.SE3TRACKING_MIN_LEVEL)]
					                    || destAbsGrad < Constants.MIN_ABS_GRAD_DECREASE)
					{
						//if(enablePrintDebugInfo) runningStats.num_prop_removed_colorDiff++;
						continue;
					}
				}
				else
				{
					float sourceColor = activeKFImageData[x + y*width];
					float destColor = Utils.interpolatedValue(newKFImageData, u_new, v_new, width);

					float residual = destColor - sourceColor;


					if(residual*residual / 
							(Constants.MAX_DIFF_CONSTANT + Constants.MAX_DIFF_GRAD_MULT*destAbsGrad*destAbsGrad)
							> 1.0f || destAbsGrad < Constants.MIN_ABS_GRAD_DECREASE)
					{
						//if(enablePrintDebugInfo) runningStats.num_prop_removed_colorDiff++;
						continue;
					}
				}

				DepthMapPixelHypothesis targetBest = otherDepthMap[newIDX];

				// large idepth = point is near = large increase in variance.
				// small idepth = point is far = small increase in variance.
				float idepth_ratio_4 = new_idepth / source.idepth_smoothed;
				idepth_ratio_4 *= idepth_ratio_4;
				idepth_ratio_4 *= idepth_ratio_4;

				float new_var =idepth_ratio_4*source.idepth_var;
				
				// check for occlusion
				if(targetBest.isValid)
				{
					// if they occlude one another, one gets removed.
					float diff = targetBest.idepth - new_idepth;
					if(Constants.DIFF_FAC_PROP_MERGE*diff*diff >
						new_var +
						targetBest.idepth_var)
					{
						if(new_idepth < targetBest.idepth)
						{
							//if(enablePrintDebugInfo) runningStats.num_prop_occluded++;
							continue;
						}
						else
						{
							//if(enablePrintDebugInfo) runningStats.num_prop_occluded++;
							targetBest.isValid = false;
						}
					}
				}
				
				if(!targetBest.isValid)
				{
					//if(enablePrintDebugInfo) runningStats.num_prop_created++;

					targetBest = new DepthMapPixelHypothesis(
							new_idepth,
							new_var,
							source.validity_counter);
					otherDepthMap[newIDX] = targetBest;
				}
				else
				{
					//if(enablePrintDebugInfo) runningStats.num_prop_merged++;

					// merge idepth ekf-style
					float w = new_var / (targetBest.idepth_var + new_var);
					float merged_new_idepth = w*targetBest.idepth + (1.0f-w)*new_idepth;

					// merge validity
					int merged_validity = source.validity_counter + targetBest.validity_counter;
					if(merged_validity > Constants.VALIDITY_COUNTER_MAX+(Constants.VALIDITY_COUNTER_MAX_VARIABLE))
						merged_validity = (int) (Constants.VALIDITY_COUNTER_MAX+(Constants.VALIDITY_COUNTER_MAX_VARIABLE));

					targetBest = new DepthMapPixelHypothesis(
							merged_new_idepth,
							1.0f/(1.0f/targetBest.idepth_var + 1.0f/new_var),
							merged_validity);

					otherDepthMap[newIDX] = targetBest;
					
				}
			}

		// swap!
		DepthMapPixelHypothesis[] temp = currentDepthMap;
		currentDepthMap = otherDepthMap;
		otherDepthMap = temp;
		
		/*
		if(enablePrintDebugInfo && printPropagationStatistics)
		{
			printf("PROPAGATE: %d: %d drop (%d oob, %d color); %d created; %d merged; %d occluded. %d col-dec, %d grad-dec.\n",
					runningStats.num_prop_attempts,
					runningStats.num_prop_removed_validity + runningStats.num_prop_removed_out_of_bounds + runningStats.num_prop_removed_colorDiff,
					runningStats.num_prop_removed_out_of_bounds,
					runningStats.num_prop_removed_colorDiff,
					runningStats.num_prop_created,
					runningStats.num_prop_merged,
					runningStats.num_prop_occluded,
					runningStats.num_prop_color_decreased,
					runningStats.num_prop_grad_decreased);
		}*/
	}

	public void setFromExistingKF(Frame kf)
	{
		assert(kf.hasIDepthBeenSet);

		activeKeyFrame = kf;

		float[] idepth = activeKeyFrame.idepth_reAct;
		float[] idepthVar = activeKeyFrame.idepthVar_reAct;
		int[] validity = activeKeyFrame.validity_reAct;

		DepthMapPixelHypothesis[] pt = currentDepthMap;
		activeKeyFrame.numMappedOnThis = 0;
		activeKeyFrame.numFramesTrackedOnThis = 0;
		//activeKeyFrameImageData = activeKeyFrame.imageArrayLvl[0];
		activeKeyFrameIsReactivated = true;

		int idx = 0;
		for(int y=0;y<height;y++) {
			for(int x=0;x<width;x++) {
				
				if(idepthVar[idx] > 0)
				{
					currentDepthMap[idx] = new DepthMapPixelHypothesis(
							idepth[idx],
							idepthVar[idx],
							validity[idx]);
				}
				else
				{
					currentDepthMap[x+y*width].isValid = false;
					currentDepthMap[x+y*width].blacklisted = (idepthVar[idx] == -2) ? Constants.MIN_BLACKLIST-1 : 0;
				}

				idx++;
			}
		}

		regularizeDepthMap(false, Constants.VAL_SUM_MIN_FOR_KEEP);
	}
	
	/*
	 * Make val non-zero
	 */
	private double UNZERO(double val) {
		return (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10
				: val));
	}

}
