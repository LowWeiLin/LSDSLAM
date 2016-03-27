import g2o.g2o_RobustKernelHuber;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import jeigen.DenseMatrix;

import org.opencv.core.Mat;

import DataStructures.EdgeSim3;
import DataStructures.Frame;
import DataStructures.FramePoseStruct;
import DataStructures.KFConstraintStruct;
import DataStructures.KeyFrameGraph;
import DataStructures.TrackingReference;
import DepthEstimation.DepthMap;
import GlobalMapping.TrackableKeyFrameSearch;
import IO.KeyFrameGraphDisplay;
import LieAlgebra.SE3;
import LieAlgebra.SIM3;
import LieAlgebra.SO3;
import LieAlgebra.Vec;
import Tracking.SE3Tracker;
import Tracking.SIM3Tracker;
import Utils.Constants;

public class LSDSLAM {
	
	
	Frame currentKeyFrame;
	DepthMap map;
	SE3Tracker tracker;
	
	boolean createNewKeyFrame = false;
	boolean trackingIsGood = true;
	
	final boolean SLAMEnabled = true;
	
	TrackingReference trackingReference;
	TrackingReference mappingTrackingReference;
	KeyFrameGraph keyFrameGraph;

	KeyFrameGraphDisplay keyFrameGraphDisplay;
	
	Frame latestTrackedFrame;
	float lastTrackingClosenessScore;
	
	
	// Find Constraint 
	TrackableKeyFrameSearch trackableKeyFrameSearch = null;
	SIM3Tracker constraintTracker;
	SE3Tracker constraintSE3Tracker;
	TrackingReference newKFTrackingReference;
	TrackingReference candidateTrackingReference;

	// PUSHED in tracking, READ & CLEARED in mapping
	Deque<Frame> unmappedTrackedFrames = new LinkedList<Frame>();
	
	// PUSHED by Mapping, READ & CLEARED by constraintFinder
	Deque<Frame> newKeyFrames = new LinkedList<Frame>();
	
	int nextRelocIdx = -1;
	
	// optimization thread
	boolean newConstraintAdded;
	
	// optimization merging. SET in Optimization, merged in Mapping.
	boolean haveUnmergedOptimizationOffset;
	
	public boolean flushPC = false;
	
	public LSDSLAM() {

		trackingReference = new TrackingReference();
		mappingTrackingReference = new TrackingReference();
		keyFrameGraph = new KeyFrameGraph();

		keyFrameGraphDisplay = new KeyFrameGraphDisplay(keyFrameGraph);
		
		haveUnmergedOptimizationOffset = false;
		

	}
	
	public void randomInit(Mat image, int id) {
		
		if (trackableKeyFrameSearch == null) {
			
			int w = image.width();
			int h = image.height();
			
			trackableKeyFrameSearch = new TrackableKeyFrameSearch(keyFrameGraph, w, h);
		
			trackableKeyFrameSearch = new TrackableKeyFrameSearch(keyFrameGraph, w, h);
			constraintTracker = new SIM3Tracker();
			constraintSE3Tracker = new SE3Tracker();
			newKFTrackingReference = new TrackingReference();
			candidateTrackingReference = new TrackingReference();
		
		}
		
		map = new DepthMap(image.width(), image.height());
		tracker = new SE3Tracker();
		
		// New currentKeyframe
		currentKeyFrame = new Frame(image);
		currentKeyFrame.isKF = true;
		
		// Initialize map
		map.initializeRandomly(currentKeyFrame);
		
		keyFrameGraph.addFrame(currentKeyFrame);

		if(SLAMEnabled) {
			keyFrameGraph.idToKeyFrame.put(currentKeyFrame.id(), currentKeyFrame);
		}
		
		System.out.println("Done random initialization.");
	}

	public void trackFrame(Mat image, int frameID) {
		
		Frame trackingNewFrame = new Frame(image);
		
		// TODO: implement
		if(!trackingIsGood)
		{
			System.err.println("RELOCALIZE");
			//relocalizer.updateCurrentFrame(trackingNewFrame);
		}
		
		boolean my_createNewKeyframe = createNewKeyFrame; // pre-save here, to make decision afterwards.
		if(trackingReference.keyframe != currentKeyFrame ||
				currentKeyFrame.depthHasBeenUpdatedFlag) {
			// Set tracking reference to be the currentKeyFrame
			trackingReference.importFrame(currentKeyFrame);
			currentKeyFrame.depthHasBeenUpdatedFlag = false;
		}

		FramePoseStruct trackingReferencePose = trackingReference.keyframe.pose;

		// DO TRACKING & Show tracking result.
		
		/*
		System.out.println("trackingReferencePose.getCamToWorld() : " 
		+ trackingReferencePose.getCamToWorld().toString());
		System.out.println(keyFrameGraph.allFramePoses
				.get(keyFrameGraph.allFramePoses.size()-1).getCamToWorld());
		*/
		
		/*
		// THIS GIVES THE WRONG ESTIMATE!
		SE3 frameToReference_initialEstimate = 
				trackingReferencePose.getCamToWorld().inverse().mul(
					keyFrameGraph.allFramePoses.get(keyFrameGraph.allFramePoses.size()-1)
					.getCamToWorld()).getSE3();
		*/
		SE3 frameToReference_initialEstimate = null;
		if (trackingReference.keyframe.trackedOnPoses.size() > 0) {
			frameToReference_initialEstimate = trackingReference.keyframe.trackedOnPoses.get(
					trackingReference.keyframe.trackedOnPoses.size()-1);
		} else {
			frameToReference_initialEstimate = new SE3();
		}
		
		System.out.println("frameToReference_initialEstimate: " + Arrays.toString(SE3.ln(frameToReference_initialEstimate)));
		System.out.println("trackingReferencePose.getCamToWorld().inverse().getSE3()" + 
				Arrays.toString(SE3.ln(trackingReferencePose.getCamToWorld().inverse().getSE3())));
		
		System.out.println("trackingReferencePose.getCamToWorld().inverse()" + 
				trackingReferencePose.getCamToWorld().inverse().getScale());
		System.out.println("keyFrameGraph.allFramePoses.get(keyFrameGraph.allFramePoses.size()-1)"
				+ ".getCamToWorld()" + 
						keyFrameGraph.allFramePoses.get(keyFrameGraph.allFramePoses.size()-1)
												   .getCamToWorld().getScale());
		
		System.out.println("keyFrameGraph.allFramePoses.get(keyFrameGraph.allFramePoses.size()-1).getCamToWorld().getSE3()" + 
				Arrays.toString(SE3.ln(
						keyFrameGraph.allFramePoses.get(keyFrameGraph.allFramePoses.size()-1)
												   .getCamToWorld().getSE3())));
		
		
		SE3 newRefToFrame_poseUpdate = tracker.trackFrame(
				trackingReference,
				trackingNewFrame,
				frameToReference_initialEstimate);

		/*
		nTrackFrame++;

		tracking_lastResidual = tracker.lastResidual;
		tracking_lastUsage = tracker.pointUsage;
		tracking_lastGoodPerBad = tracker.lastGoodCount / (tracker.lastGoodCount + tracker.lastBadCount);
		tracking_lastGoodPerTotal = tracker.lastGoodCount / (trackingNewFrame.width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));


		 */
		
		if(Constants.manualTrackingLossIndicated || tracker.diverged || (keyFrameGraph.keyframesAll.size() > Constants.INITIALIZATION_PHASE_COUNT && !tracker.trackingWasGood))
		{
			System.out.println("manualTrackingLossIndicated: " + Boolean.toString(Constants.manualTrackingLossIndicated));
			System.out.println("keyFrameGraph.keyframesAll.size(): " + keyFrameGraph.keyframesAll.size());
			System.out.println("tracker.trackingWasGood: " + Boolean.toString(tracker.trackingWasGood));
			System.out.println("tracker.diverged: " + Boolean.toString(tracker.diverged));
			
			/*
			printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
					trackingNewFrame.id(),
					100*tracking_lastGoodPerTotal,
					100*tracking_lastGoodPerBad,
					tracker.diverged ? "DIVERGED" : "NOT DIVERGED");
			*/
			trackingReference.invalidate();

			trackingIsGood = false;
			nextRelocIdx = -1;

			Constants.manualTrackingLossIndicated = false;
			return;
		}
		 

		keyFrameGraph.addFrame(trackingNewFrame);
		


		
		// Keyframe selection
		
		latestTrackedFrame = trackingNewFrame;
		if (!my_createNewKeyframe &&
				currentKeyFrame.numMappedOnThisTotal > Constants.MIN_NUM_MAPPED) {
			
			DenseMatrix dist = newRefToFrame_poseUpdate.getTranslationMat().mul(currentKeyFrame.meanIdepth);
			double[] distVec = Vec.vec3ToArray(dist); 
			float minVal = Math.min(0.2f + keyFrameGraph.keyframesAll.size() * 0.8f / Constants.INITIALIZATION_PHASE_COUNT, 1.0f);

			if(keyFrameGraph.keyframesAll.size() < Constants.INITIALIZATION_PHASE_COUNT) {
				minVal *= 0.7;
			}

			lastTrackingClosenessScore = trackableKeyFrameSearch.getRefFrameScore(
					(float) Vec.dot(distVec, distVec), tracker.pointUsage);

			if (lastTrackingClosenessScore > minVal) {
				System.err.println("CREATE NEW KEYFRAME");
				createNewKeyFrame = true;
				trackingNewFrame.isKF = true;
				flushPC = true;
			} else {
			}
		}
		
		// TODO: REMOVE THIS
//		if (trackingNewFrame.id() >= 11 && trackingNewFrame.id() <=14) {
//			if (trackingNewFrame.id() == 14) {
//				createNewKeyFrame = true;
//				trackingNewFrame.isKF = true;
//			} else {
//				createNewKeyFrame = false;
//				trackingNewFrame.isKF = false;
//			}
//			
//		}
		

		// Push into deque for mapping
		if(unmappedTrackedFrames.size() < 50 || 
				(unmappedTrackedFrames.size() < 100 &&
				trackingNewFrame.getTrackingParent().numMappedOnThisTotal < 10)) {
			
			unmappedTrackedFrames.push(trackingNewFrame);
		}
		
		// TODO: block till mapping is done - just do sequentially for now.
		
		
	}
	
	public boolean doMappingIteration() {
		if(currentKeyFrame == null) {
			System.err.println("doMappingIteration: currentKeyFrame is null!");
			return false;
		}

		boolean trackingIsGood = tracker.trackingWasGood;
		
		// set mappingFrame
		if(trackingIsGood) {
			System.out.println("Tracking was good!");
			if (createNewKeyFrame) {
				System.out.println("doMappingIteration: create new keyframe");
				// create new key frame
				finishCurrentKeyframe();
				changeKeyframe(false, true, 1.0f);
			} else {
				System.out.println("doMappingIteration: update keyframe");
				// ***Update key frame here***
				boolean didSomething = updateKeyframe();

				if(!didSomething) {
					System.err.println("updateKeyFrame: false");
					return false;
				}
			}
			
			return true;
		} else { // Tracking is not good
			System.err.println("Tracking was bad!");
			
			
			// TODO: relocalize
			System.err.println("Relocalize (2)");
			
			
			// invalidate map if it was valid.
			if(map.isValid()) {
				if(currentKeyFrame.numMappedOnThisTotal >= Constants.MIN_NUM_MAPPED) {
					finishCurrentKeyframe();
				} else {
					System.err.println("DiscardCurrentkeyFrame");
					//discardCurrentKeyframe();
				}
				System.err.println("map.invalidate");
				//map.invalidate();
			}
			/*
			// start relocalizer if it isnt running already
			if(!relocalizer.isRunning)
				relocalizer.start(keyFrameGraph->keyframesAll);

			// did we find a frame to relocalize with?
			if(relocalizer.waitResult(50))
				takeRelocalizeResult();

			*/
			return true;
		}
	}
	

	// Updates key frame with measurements from a new frame.
	public boolean updateKeyframe() {
		Deque<Frame> references = new LinkedList<Frame>();

		// remove frames that have a different tracking parent.
		while(unmappedTrackedFrames.size() > 0 &&
				(!unmappedTrackedFrames.peekFirst().hasTrackingParent() ||
				unmappedTrackedFrames.peekFirst().getTrackingParent() != currentKeyFrame)) {
			
			System.out.println("Pop " + !unmappedTrackedFrames.peekFirst().hasTrackingParent()
					+ " " + (unmappedTrackedFrames.peekFirst().getTrackingParent() != currentKeyFrame));
			unmappedTrackedFrames.pop();//.clear_refPixelWasGood();
		}

		// clone list
		if(unmappedTrackedFrames.size() > 0) {
			// Copy from unmappedTrackedFrames to references
			references.addAll(unmappedTrackedFrames);
			Frame popped = unmappedTrackedFrames.pop();

			// ***Do the update here***
			// references - list of frames to map
			map.updateKeyframe(references);

			popped.clear_refPixelWasGood();
			
			// CLEAR DATA
			for (Frame f : references) {
				if (!createNewKeyFrame && f.isKF == false) {
					f.clearData();
				}
			}
			
			references.clear();
			unmappedTrackedFrames.clear();
			
		} else {
			return false;
		}

		return true;
	}

	void finishCurrentKeyframe()
	{
		System.out.println("FINALIZING KF: " + currentKeyFrame.id());
	
		map.finalizeKeyFrame();
	
	
		if(SLAMEnabled) {
			mappingTrackingReference.importFrame(currentKeyFrame);
			currentKeyFrame.setPermaRef(mappingTrackingReference);
			mappingTrackingReference.invalidate();
	
			if(currentKeyFrame.idxInKeyframes < 0) {
				currentKeyFrame.idxInKeyframes = keyFrameGraph.keyframesAll.size();
				keyFrameGraph.keyframesAll.add(currentKeyFrame);
				keyFrameGraph.totalPoints += currentKeyFrame.numPoints;
				keyFrameGraph.totalVertices++;
	
				newKeyFrames.add(currentKeyFrame);
				
			}
		}
		
		//if(outputWrapper!= 0)
		//	outputWrapper->publishKeyframe(currentKeyFrame.get());
	}
	

	void changeKeyframe(boolean noCreate, boolean force, float maxScore) {
		Frame newReferenceKF = null;
		Frame newKeyframeCandidate = latestTrackedFrame;
		
		if(Constants.doKFReActivation && SLAMEnabled)
		{
			// TODO: use relocalization
			//newReferenceKF = trackableKeyFrameSearch.findRePositionCandidate(newKeyframeCandidate, maxScore);
		}
	
		if(newReferenceKF != null) {
			System.err.println("RELOCALIZED!");
			loadNewCurrentKeyframe(newReferenceKF);
		} else {
			if(force) {
				if(noCreate) {
					trackingIsGood = false;
					nextRelocIdx = -1;
					System.out.println("mapping is disabled & moved outside of known map. Starting Relocalizer!\n");
				} else {
					createNewCurrentKeyframe(newKeyframeCandidate);
				}
			}
		}
	
	
		createNewKeyFrame = false;
	}
	
	void loadNewCurrentKeyframe(Frame keyframeToLoad)
	{
		System.out.printf("RE-ACTIVATE KF %d\n", keyframeToLoad.id());

		map.setFromExistingKF(keyframeToLoad);

		System.out.printf("re-activate frame %d!\n", keyframeToLoad.id());

		currentKeyFrame = keyFrameGraph.idToKeyFrame.get(keyframeToLoad.id());
		currentKeyFrame.depthHasBeenUpdatedFlag = false;
	}
	
	void createNewCurrentKeyframe(Frame newKeyframeCandidate)
	{
		System.out.printf("CREATE NEW KF %d from %d\n", newKeyframeCandidate.id(), currentKeyFrame.id());


		if(SLAMEnabled)
		{
			// add NEW keyframe to id-lookup
			keyFrameGraph.idToKeyFrame.put(newKeyframeCandidate.id(), newKeyframeCandidate);
		}

		// propagate & make new.
		map.createKeyFrame(newKeyframeCandidate);

		/*
		if(printPropagationStatistics)
		{

			Eigen::Matrix<float, 20, 1> data;
			data.setZero();
			data[0] = runningStats.num_prop_attempts / ((float)width*height);
			data[1] = (runningStats.num_prop_created + runningStats.num_prop_merged) / (float)runningStats.num_prop_attempts;
			data[2] = runningStats.num_prop_removed_colorDiff / (float)runningStats.num_prop_attempts;

			outputWrapper->publishDebugInfo(data);
		}
		*/

		currentKeyFrame = newKeyframeCandidate;
	}
	
	void constraintSearchIteration()
	{
		System.out.println("Started constraint search!");

		// Have newKeyFrames
		if (newKeyFrames.size() > 0) {
			Frame newKF = newKeyFrames.pop();
			
			findConstraintsForNewKeyFrames(newKF, true, 1.0f);
			
			
			
			// TODO: Should be here?
			optimizationIteration(50, 0.001);
		}
		
		
		
		
		// doFullReConstraintTrack
		/*
		int failedToRetrack = 0;
		
		int added = 0;
		for(int i=0;i<keyFrameGraph.keyframesAll.size();i++)
		{
			if(keyFrameGraph.keyframesAll.get(i).pose.isInGraph)
				added += findConstraintsForNewKeyFrames(keyFrameGraph.keyframesAll.get(i), false, 1.0f);
		}

		System.out.printf("Done optizing Full Map! Added %d constraints.\n", added);
		*/
	}
	
	int findConstraintsForNewKeyFrames(Frame newKeyFrame, boolean forceParent, float closeCandidatesTH)
	{
		System.out.println("findConstraintsForNewKeyFrames");
		if(!newKeyFrame.hasTrackingParent())
		{
			keyFrameGraph.addKeyFrame(newKeyFrame);
			newConstraintAdded = true;
			System.out.println("findConstraintsForNewKeyFrames - ret 0");
			return 0;
		}

		
		if(!forceParent && 
				Vec.magnitude((newKeyFrame.lastConstraintTrackedCamToWorld.mul(
						newKeyFrame.getScaledCamToWorld().inverse())).ln()) < 0.01) {
			System.out.println("findConstraintsForNewKeyFrames - 515");
			return 0;
		}


		newKeyFrame.lastConstraintTrackedCamToWorld = newKeyFrame.getScaledCamToWorld();

		
		System.out.println("findConstraintsForNewKeyFrames - get all potential candidates and their initial relative pose.");
		// =============== get all potential candidates and their initial relative pose. =================
		List<KFConstraintStruct> constraints = new ArrayList<KFConstraintStruct>();
		//Frame fabMapResult = null;
		Set<Frame> candidates = trackableKeyFrameSearch.findCandidates(newKeyFrame, closeCandidatesTH);
		Map<Frame, SIM3> candidateToFrame_initialEstimateMap = new HashMap<Frame, SIM3>();
		
		System.out.println("trackableKeyFrameSearch.findCandidates - " + candidates.size());

		// erase the ones that are already neighbours.
		for(Frame c : candidates)
		{
			if(newKeyFrame.neighbors.contains(c) == true)
			{
				System.out.printf("SKIPPING %d on %d cause it already exists as constraint.\n", c.id(), newKeyFrame.id());
				candidates.remove(c);
			}
		}

		for (Frame candidate : candidates)
		{
			SIM3 candidateToFrame_initialEstimate = newKeyFrame.getScaledCamToWorld().inverse().mul( 
					candidate.getScaledCamToWorld());
			//System.out.println("Add candidate: frame " + candidate.id() + ", " + Arrays.toString(candidateToFrame_initialEstimate.ln()));
			candidateToFrame_initialEstimateMap.put(candidate, candidateToFrame_initialEstimate);
			//System.out.println("candidateToFrame_initialEstimateMap " + candidateToFrame_initialEstimateMap.size());
		}

		Map<Frame, Integer> distancesToNewKeyFrame = new HashMap<Frame, Integer>();
		if(newKeyFrame.hasTrackingParent())
			keyFrameGraph.calculateGraphDistancesToFrame(
					newKeyFrame.getTrackingParent(), distancesToNewKeyFrame);


		System.out.println("findConstraintsForNewKeyFrames - distinguish between close and far candidates in Graph");
		// =============== distinguish between close and "far" candidates in Graph =================
		// Do a first check on trackability of close candidates.
		Set<Frame> closeCandidates = new HashSet<Frame>();
		List<Frame> farCandidates = new ArrayList<Frame>();
		Frame parent = newKeyFrame.hasTrackingParent() ? newKeyFrame.getTrackingParent() : null;

		int closeFailed = 0;
		int closeInconsistent = 0;

		SO3 disturbance = new SO3();
		disturbance.set31(new double[]{0.05,0,0});

		for (Frame candidate : candidates)
		{
			if (candidate.id() == newKeyFrame.id())
				continue;
			if(!candidate.pose.isInGraph)
				continue;
			if(newKeyFrame.hasTrackingParent() && candidate == newKeyFrame.getTrackingParent())
				continue;
			if(candidate.idxInKeyframes < Constants.INITIALIZATION_PHASE_COUNT)
				continue;
			
			SE3 c2f_init = candidateToFrame_initialEstimateMap.get(candidate).inverse().getSE3().inverse();
			c2f_init.rotation.mulEq(disturbance);
			
			// TRACK SE3
			constraintSE3Tracker.initialize(newKeyFrame.width(0), newKeyFrame.height(0));
			SE3 c2f = constraintSE3Tracker.trackFrameOnPermaref(candidate, newKeyFrame, c2f_init);
			if(!constraintSE3Tracker.trackingWasGood) {
				closeFailed++;
				continue;
			}


			SE3 f2c_init = candidateToFrame_initialEstimateMap.get(candidate).getSE3().inverse();
			disturbance.mulEq(f2c_init.rotation);
			f2c_init.rotation = disturbance;
			
			SE3 f2c = constraintSE3Tracker.trackFrameOnPermaref(newKeyFrame, candidate, f2c_init);
			if(!constraintSE3Tracker.trackingWasGood) {
				closeFailed++;
				continue;
			}

			if(Vec.magnitude(f2c.rotation.mul(c2f.rotation).ln()) >= 0.09) {
				closeInconsistent++;
				continue;
			}

			closeCandidates.add(candidate);
			
		}


	
		int farFailed = 0;
		int farInconsistent = 0;
		for (Frame candidate : candidates)
		{
			if (candidate.id() == newKeyFrame.id())
				continue;
			if(!candidate.pose.isInGraph)
				continue;
			if(newKeyFrame.hasTrackingParent() && candidate == newKeyFrame.getTrackingParent())
				continue;
			if(candidate.idxInKeyframes < Constants.INITIALIZATION_PHASE_COUNT)
				continue;

			if(distancesToNewKeyFrame.get(candidate) < 4)
				continue;

			farCandidates.add(candidate);
		}



		int closeAll = closeCandidates.size();
		int farAll = farCandidates.size();
		
		// TODO: implement
/*
		// erase the ones that we tried already before (close)
		for(Frame c : closeCandidates)
		{
			if(newKeyFrame.trackingFailed.contains(c) == false)
			{
				continue;
			}
			auto range = newKeyFrame.trackingFailed.equal_range(c);

			boolean skip = false;
			SIM3 f2c = candidateToFrame_initialEstimateMap.get(c).inverse();
			for (auto it = range.first; it != range.second; ++it)
			{
				if((f2c * it->second).log().norm() < 0.1)
				{
					skip=true;
					break;
				}
			}

			if(skip)
			{
				//if(enablePrintDebugInfo && printConstraintSearchInfo)
				//	printf("SKIPPING %d on %d (NEAR), cause we already have tried it.\n", (*c)->id(), newKeyFrame->id());
				//c = closeCandidates.erase(c);
				erase c
			}
			else {
				//++c;
			}
		}
*/

		// erase the ones that are already neighbours (far)
		// TODO: implement
		/*
		for(int i=0;i<farCandidates.size();i++)
		{
			if(newKeyFrame.trackingFailed.contains(farCandidates.get(i)) == false)
				continue;

			auto range = newKeyFrame.trackingFailed.equal_range(farCandidates.get(i));

			boolean skip = false;
			for (auto it = range.first; it != range.second; ++it)
			{
				if((it->second).log().norm() < 0.2)
				{
					skip=true;
					break;
				}
			}

			if(skip)
			{
//				if(enablePrintDebugInfo && printConstraintSearchInfo)
//					printf("SKIPPING %d on %d (FAR), cause we already have tried it.\n", farCandidates[i]->id(), newKeyFrame->id());
				farCandidates[i] = farCandidates.back();
				farCandidates.pop_back();
				i--;
			}
		}*/



//		if (enablePrintDebugInfo && printConstraintSearchInfo)
//			printf("Final Loop-Closure Candidates: %d / %d close (%d failed, %d inconsistent) + %d / %d far (%d failed, %d inconsistent) = %d\n",
//					(int)closeCandidates.size(),closeAll, closeFailed, closeInconsistent,
//					(int)farCandidates.size(), farAll, farFailed, farInconsistent,
//					(int)closeCandidates.size() + (int)farCandidates.size());



		// =============== limit number of close candidates ===============
		// while too many, remove the one with the highest connectivity.
		while((int)closeCandidates.size() > Constants.maxLoopClosureCandidates)
		{
			Frame worst = null;
			int worstNeighbours = 0;
			for(Frame f : closeCandidates)
			{
				int neightboursInCandidates = 0;
				for(Frame n : f.neighbors)
					if(closeCandidates.contains(n))
						neightboursInCandidates++;

				if(neightboursInCandidates > worstNeighbours || worst == null)
				{
					worst = f;
					worstNeighbours = neightboursInCandidates;
				}
			}

			closeCandidates.remove(worst);
		}


		// TODO: implement
		/*
		// =============== limit number of far candidates ===============
		// delete randomly
		int maxNumFarCandidates = (Constants.maxLoopClosureCandidates +1) / 2;
		if(maxNumFarCandidates < 5) maxNumFarCandidates = 5;
		while((int)farCandidates.size() > maxNumFarCandidates)
		{
			int toDelete = rand() % farCandidates.size();
			if(farCandidates[toDelete] != fabMapResult)
			{
				farCandidates[toDelete] = farCandidates.back();
				farCandidates.pop_back();
			}
		}
		 */


		// =============== TRACK! ===============
		System.out.println("findConstraintsForNewKeyFrames - Track!");
		
		// make tracking reference for newKeyFrame.
		newKFTrackingReference.importFrame(newKeyFrame);

		// For all close candidates of newKeyFrame
		for (Frame candidate : closeCandidates)
		{
			KFConstraintStruct e1 = null;
			KFConstraintStruct e2 = null;

			KFConstraintStruct[] result = testConstraint(
					candidate, e1, e2,
					candidateToFrame_initialEstimateMap.get(candidate),
					Constants.loopclosureStrictness);
			e1 = result[0];
			e2 = result[1];
			
			
			//if(enablePrintDebugInfo && printConstraintSearchInfo)
			//	printf(" CLOSE (%d)\n", distancesToNewKeyFrame.at(candidate));

			if(e1 != null)
			{
				constraints.add(e1);
				constraints.add(e2);

				// delete from far candidates if it's in there.
				for(int k=0 ; k<farCandidates.size();k++)
				{
					if(farCandidates.get(k) == candidate)
					{
						//if(enablePrintDebugInfo && printConstraintSearchInfo)
						//	printf(" DELETED %d from far, as close was successful!\n", candidate->id());

						//farCandidates[k] = farCandidates.back();
						//farCandidates.pop_back();
						
						// TODO: Replace with last, remove last
						
					}
				}
			}
		}

		// Far candidates, (for loop closure?)
		for (Frame candidate : farCandidates)
		{
			KFConstraintStruct e1 = null;
			KFConstraintStruct e2 = null;

			KFConstraintStruct[] result = testConstraint(
					candidate, e1, e2,
					new SIM3(),
					Constants.loopclosureStrictness);
			e1 = result[0];
			e2 = result[1];
			
//			if(enablePrintDebugInfo && printConstraintSearchInfo)
//				printf(" FAR (%d)\n", distancesToNewKeyFrame.at(candidate));

			if(e1 != null)
			{
				constraints.add(e1);
				constraints.add(e2);
			}
		}



		if(parent != null && forceParent)
		{
			if (candidateToFrame_initialEstimateMap.get(parent) == null) {
				System.out.println("candidateToFrame_initialEstimateMap.get(parent) == null");
				System.out.println("Frame " + newKeyFrame.id());
				System.out.println("Frame parent " + parent.id());
			}
			
			KFConstraintStruct e1 = null;
			KFConstraintStruct e2 = null;
			KFConstraintStruct[] result = testConstraint(
					parent, e1, e2,
					candidateToFrame_initialEstimateMap.get(parent),
					100);
			e1 = result[0];
			e2 = result[1];
			
//			if(enablePrintDebugInfo && printConstraintSearchInfo)
//				printf(" PARENT (0)\n");

			if(e1 != null)
			{
				constraints.add(e1);
				constraints.add(e2);
			}
			else
			{
				float downweightFac = 5;
				float kernelDelta = (float) (5f * Math.sqrt(6000f*Constants.loopclosureStrictness) / downweightFac);
				System.out.printf("warning: reciprocal tracking on new frame failed badly, added odometry edge (Hacky).\n");

				//poseConsistencyMutex.lock_shared();
				
				KFConstraintStruct k = new KFConstraintStruct();
				k.firstFrame = newKeyFrame;
				k.secondFrame = newKeyFrame.getTrackingParent();
				k.secondToFirst = k.firstFrame.getScaledCamToWorld().inverse().mul(
								  k.secondFrame.getScaledCamToWorld());
				k.information = new DenseMatrix(new double[][] {
						{0.8098,-0.1507,-0.0557, 0.1211, 0.7657, 0.0120, 0},
						{-0.1507, 2.1724,-0.1103,-1.9279,-0.1182, 0.1943, 0},
						{-0.0557,-0.1103, 0.2643,-0.0021,-0.0657,-0.0028, 0.0304},
						{ 0.1211,-1.9279,-0.0021, 2.3110, 0.1039,-0.0934, 0.0005},
						{ 0.7657,-0.1182,-0.0657, 0.1039, 1.0545, 0.0743,-0.0028},
						{ 0.0120, 0.1943,-0.0028,-0.0934, 0.0743, 0.4511, 0},
						{ 0,0, 0.0304, 0.0005,-0.0028, 0, 0.0228}});
				k.information = k.information.mul((1e9/(downweightFac*downweightFac)));
				
				// TODO:
//				k.information = DenseMatrix.eye(7).mul(100000);
//				k.information.set(6, 6, 0);

				k.robustKernel = new g2o_RobustKernelHuber();
				k.robustKernel.setDelta(kernelDelta);

				k.meanResidual = 10;
				k.meanResidualD = 10;
				k.meanResidualP = 10;
				k.usage = 0;
				
				constraints.add(k);
				
				System.err.println("Constraint Found: " + k.firstFrame.id() + " - " + k.secondFrame.id());
				if (k.firstFrame.pose.graphVertex != null)
					System.err.println(k.firstFrame.id() + " - " + k.firstFrame.pose.graphVertex.id);
				if (k.secondFrame.pose.graphVertex != null)
					System.err.println(k.secondFrame.id() + " - " + k.secondFrame.pose.graphVertex.id);
				
				//poseConsistencyMutex.unlock_shared();
			}
		}


		//newConstraintMutex.lock();

		keyFrameGraph.addKeyFrame(newKeyFrame);
		for(int i=0 ; i<constraints.size() ; i++)
			keyFrameGraph.insertConstraint(constraints.get(i));


		newConstraintAdded = true;
		//newConstraintCreatedSignal.notify_all();
		//newConstraintMutex.unlock();

		newKFTrackingReference.invalidate();
		candidateTrackingReference.invalidate();



		return constraints.size();
	}
	
	public KFConstraintStruct[] testConstraint(
			Frame candidate,
			KFConstraintStruct e1_out, KFConstraintStruct e2_out,
			SIM3 candidateToFrame_initialEstimate,
			float strictness)
	{	
		candidateTrackingReference.importFrame(candidate);

		SIM3 FtoC = candidateToFrame_initialEstimate.inverse();
		SIM3 CtoF = candidateToFrame_initialEstimate;
		DenseMatrix FtoCInfo, CtoFInfo; // 7x7
		
		// TODO : REMOVE
//		if (candidate.id() == 0) {
//			FtoC = SIM3.exp(new double[]{0.0656, -0.1039, 0.0033, -0.0308, 0.0010, 0.0184, 0.0180}).inverse();
//			CtoF = SIM3.inverse(FtoC);
//		}
		

		TryTrackSim3Result res_level3 = tryTrackSim3(
				newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
				Constants.SIM3TRACKING_MAX_LEVEL-1, 3,
				FtoC, CtoF);
		float err_level3 = res_level3.error;
		FtoC = res_level3.AtoB;
		CtoF = res_level3.BtoA;
		
		System.out.println("tryTrackSim3 err_level3: " + err_level3);

		if(err_level3 > 3000*strictness)
		{
//			if(enablePrintDebugInfo && printConstraintSearchInfo)
//				printf("FAILE %d -> %d (lvl %d): errs (%.1f / - / -).",
//					newKFTrackingReference->frameID, candidateTrackingReference->frameID,
//					3,
//					sqrtf(err_level3));

			e1_out = e2_out = null;

			newKFTrackingReference.keyframe.trackingFailed.putElement(
					candidate, candidateToFrame_initialEstimate);
					
			return new KFConstraintStruct[] {e1_out, e2_out};
		}

		TryTrackSim3Result res_level2 = tryTrackSim3(
				newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
				2, 2,
				FtoC, CtoF);
		float err_level2 = res_level2.error;
		FtoC = res_level2.AtoB;
		CtoF = res_level2.BtoA;

		System.out.println("tryTrackSim3 err_level2: " + err_level2);
		
		if(err_level2 > 4000*strictness)
		{
//			if(enablePrintDebugInfo && printConstraintSearchInfo)
//				printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / -).",
//					newKFTrackingReference->frameID, candidateTrackingReference->frameID,
//					2,
//					sqrtf(err_level3), sqrtf(err_level2));

			e1_out = e2_out = null;
			newKFTrackingReference.keyframe.trackingFailed.putElement(
					candidate, candidateToFrame_initialEstimate);
			return new KFConstraintStruct[] {e1_out, e2_out};
		}

		e1_out = new KFConstraintStruct();
		e2_out = new KFConstraintStruct();

		TryTrackSim3Result res_level1 = tryTrackSim3(
				newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
				1, 1,
				FtoC, CtoF, e1_out, e2_out);
		float err_level1 = res_level1.error;
		FtoC = res_level1.AtoB;
		CtoF = res_level1.BtoA;
		e1_out = res_level1.e1;
		e2_out = res_level1.e2;
	
		System.out.println("tryTrackSim3 err_level1: " + err_level3);

		if(err_level1 > 6000*strictness)
		{
//			if(enablePrintDebugInfo && printConstraintSearchInfo)
//				printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / %.1f).",
//						newKFTrackingReference->frameID, candidateTrackingReference->frameID,
//						1,
//						sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));

//			delete e1_out;
//			delete e2_out;
			e1_out = e2_out = null;
			newKFTrackingReference.keyframe.trackingFailed.putElement(
					candidate, candidateToFrame_initialEstimate);
			return new KFConstraintStruct[] {e1_out, e2_out};
		}


//		if(enablePrintDebugInfo && printConstraintSearchInfo)
//			printf("ADDED %d -> %d: errs (%.1f / %.1f / %.1f).",
//				newKFTrackingReference->frameID, candidateTrackingReference->frameID,
//				sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));


		float kernelDelta = (float) (5f * Math.sqrt(6000f*Constants.loopclosureStrictness));
		e1_out.robustKernel = new g2o_RobustKernelHuber();
		e1_out.robustKernel.setDelta(kernelDelta);
		e2_out.robustKernel = new g2o_RobustKernelHuber();
		e2_out.robustKernel.setDelta(kernelDelta);
		
		return new KFConstraintStruct[] {e1_out, e2_out};
	}
	
	
	public class TryTrackSim3Result {
		public float error;
		public SIM3 AtoB;
		public SIM3 BtoA;
		public KFConstraintStruct e1;
		public KFConstraintStruct e2;
		public TryTrackSim3Result(float error, SIM3 AtoB, SIM3 BtoA, KFConstraintStruct e1, KFConstraintStruct e2) {
			this.error = error;
			this.AtoB = AtoB;
			this.BtoA = BtoA;
			this.e1 = e1;
			this.e2 = e2;
		}
	}
	
	TryTrackSim3Result tryTrackSim3(
			TrackingReference A, TrackingReference B,
			int lvlStart, int lvlEnd,
			SIM3 AtoB, SIM3 BtoA) {
		return tryTrackSim3(
				A, B,
				lvlStart, lvlEnd,
				AtoB, BtoA,
				null, null);
	}
	TryTrackSim3Result tryTrackSim3(
			TrackingReference A, TrackingReference B,
			int lvlStart, int lvlEnd,
			SIM3 AtoB, SIM3 BtoA,
			KFConstraintStruct e1, KFConstraintStruct e2 )
	{
		assert(A != null);
		assert(B != null);
		assert(constraintTracker != null);
		
		System.out.println("tryTrackSim3 " + A.keyframe.id() + " - " + B.keyframe.id());
		System.out.println("AtoB: " + Arrays.toString(AtoB.ln()));
		
		BtoA = constraintTracker.trackFrameSim3(
				A,
				B.keyframe,
				BtoA,
				lvlStart,lvlEnd);
		DenseMatrix BtoAInfo = constraintTracker.lastSim3Hessian;//7x7
		float BtoA_meanResidual = constraintTracker.lastResidual;
		float BtoA_meanDResidual = constraintTracker.lastDepthResidual;
		float BtoA_meanPResidual = constraintTracker.lastPhotometricResidual;
		float BtoA_usage = constraintTracker.pointUsage;


		if (constraintTracker.diverged ||
			BtoA.getScale() > 1 / Constants.EPSILON ||
			BtoA.getScale() < Constants.EPSILON ||
			BtoAInfo.get(0,0) == 0 ||
			BtoAInfo.get(6,6) == 0)
		{
			System.err.println("tryTrackSim3 fail 1");
			System.err.println("constraintTracker.diverged: " + constraintTracker.diverged);
			System.err.println("BtoA.getScale(): " + BtoA.getScale());
			System.err.println("BtoA.getScale(): " + BtoA.getScale());
			System.err.println("BtoAInfo.get(0,0): " + BtoAInfo.get(0,0));

			System.err.println("BtoAInfo.shape: " + BtoAInfo.rows + ", " + BtoAInfo.cols);
			
			System.err.println("BtoAInfo.get(6,6): " + BtoAInfo.get(6,6));
			
			
			return new TryTrackSim3Result((float) 1e20, AtoB, BtoA, e1, e2);
		}


		AtoB = constraintTracker.trackFrameSim3(
				B,
				A.keyframe,
				AtoB,
				lvlStart,lvlEnd);
		DenseMatrix AtoBInfo = constraintTracker.lastSim3Hessian; //7x7
		float AtoB_meanResidual = constraintTracker.lastResidual;
		float AtoB_meanDResidual = constraintTracker.lastDepthResidual;
		float AtoB_meanPResidual = constraintTracker.lastPhotometricResidual;
		float AtoB_usage = constraintTracker.pointUsage;


		if (constraintTracker.diverged ||
			AtoB.getScale() > 1 / Constants.EPSILON ||
			AtoB.getScale() < Constants.EPSILON ||
			AtoBInfo.get(0,0) == 0 ||
			AtoBInfo.get(6,6) == 0)
		{

			System.err.println("tryTrackSim3 fail 2");
			System.err.println("constraintTracker.diverged: " + constraintTracker.diverged);
			System.err.println("AtoB.getScale(): " + AtoB.getScale());
			System.err.println("AtoB.getScale(): " + AtoB.getScale());
			System.err.println("AtoBInfo.get(0,0): " + AtoBInfo.get(0,0));
			System.err.println("AtoBInfo.get(6,6): " + AtoBInfo.get(6,6));
			return new TryTrackSim3Result((float) 1e20, AtoB, BtoA, e1, e2);
		}

		// Propagate uncertainty (with d(a * b) / d(b) = Adj_a) and calculate Mahalanobis norm
		
		// 7x7
		DenseMatrix datimesb_db = AtoB.adjoint();
		DenseMatrix diffHesse = (AtoBInfo.fullPivHouseholderQRSolve(DenseMatrix.eye(7)).add(
				datimesb_db.mmul(
						BtoAInfo.fullPivHouseholderQRSolve(DenseMatrix.eye(7))).mmul(
						datimesb_db.t())))
				.fullPivHouseholderQRSolve(DenseMatrix.eye(7));
		// 7x1
		DenseMatrix diff = Vec.array7ToVec((AtoB.mul(BtoA)).ln());


		float reciprocalConsistency = (float) (diffHesse.mmul(diff)).mmul(diff.t()).get(0, 0); // dot product


		if(e1 != null && e2 != null)
		{
			e1.firstFrame = A.keyframe;
			e1.secondFrame = B.keyframe;
			e1.secondToFirst = BtoA;
			e1.information = BtoAInfo;
			e1.meanResidual = BtoA_meanResidual;
			e1.meanResidualD = BtoA_meanDResidual;
			e1.meanResidualP = BtoA_meanPResidual;
			e1.usage = BtoA_usage;

			e2.firstFrame = B.keyframe;
			e2.secondFrame = A.keyframe;
			e2.secondToFirst = AtoB;
			e2.information = AtoBInfo;
			e2.meanResidual = AtoB_meanResidual;
			e2.meanResidualD = AtoB_meanDResidual;
			e2.meanResidualP = AtoB_meanPResidual;
			e2.usage = AtoB_usage;

			e1.reciprocalConsistency = e2.reciprocalConsistency = reciprocalConsistency;
		}

		return new TryTrackSim3Result(reciprocalConsistency, AtoB, BtoA, e1, e2);
	}
	
	


	boolean optimizationIteration(int itsPerTry, double minChange)
	{
		keyFrameGraph.addElementsFromBuffer();
		
	
		// Do the optimization. This can take quite some time!
		int its = keyFrameGraph.optimize(itsPerTry);
		
	
		// save the optimization result.
		float maxChange = 0;
		float sumChange = 0;
		float sum = 0;
		for(int i=0;i<keyFrameGraph.keyframesAll.size(); i++)
		{
			// set edge error sum to zero
			keyFrameGraph.keyframesAll.get(i).edgeErrorSum = 0;
			keyFrameGraph.keyframesAll.get(i).edgesNum = 0;
	
			if(!keyFrameGraph.keyframesAll.get(i).pose.isInGraph)
				continue;
	
	
			// get change from last optimization
			SIM3 a = keyFrameGraph.keyframesAll.get(i).pose.graphVertex.estimate();
			SIM3 b = keyFrameGraph.keyframesAll.get(i).getScaledCamToWorld();
			double[] diff = (a.mul(b.inverse())).ln();

			for(int j=0;j<7;j++)
			{
				float d = Math.abs((float)(diff[j]));
				if(d > maxChange) maxChange = d;
				sumChange += d;
			}
			sum +=7;
	
			// set change
			keyFrameGraph.keyframesAll.get(i).pose.setPoseGraphOptResult(
					keyFrameGraph.keyframesAll.get(i).pose.graphVertex.estimate());
	
			// add error
			for(EdgeSim3 edge : keyFrameGraph.keyframesAll.get(i).pose.graphVertex.edges())
			{
				keyFrameGraph.keyframesAll.get(i).edgeErrorSum += edge.chi2();
				keyFrameGraph.keyframesAll.get(i).edgesNum++;
			}
			
			
			// TODO:
			// applyPoseGraphOptResult()
			keyFrameGraph.keyframesAll.get(i).pose.applyPoseGraphOptResult();
			
			
		}
	
		haveUnmergedOptimizationOffset = true;
	
		System.out.printf("did %d optimization iterations. Max Pose Parameter Change: %f; avgChange: %f. %s\n", its, maxChange, sumChange / sum,
				maxChange > minChange && its == itsPerTry ? "continue optimizing":"Waiting for addition to graph.");
	
		return maxChange > minChange && its == itsPerTry;
	}
	
}
