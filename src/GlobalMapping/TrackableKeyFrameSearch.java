package GlobalMapping;

import java.util.ArrayList;

import jeigen.DenseMatrix;
import DataStructures.Frame;
import DataStructures.KeyFrameGraph;
import LieAlgebra.SE3;
import LieAlgebra.Vec;
import Tracking.Tracker;
import Utils.Constants;

public class TrackableKeyFrameSearch {
	
	static float KFDistWeight = 4;
	static float KFUsageWeight = 3;
	static float relocalizationTH = 0.7f;
	
	class TrackableKFStruct {
		public Frame ref;
		public SE3 refToFrame;
		public float dist;
		public float angle;
	};


	KeyFrameGraph graph;
	Tracker tracker;
	float fowX, fowY;
	

	public TrackableKeyFrameSearch(KeyFrameGraph graph, int w, int h)
	{
		tracker = new Tracker();
		tracker.initialize(w, h);
		this.graph = graph;
	
		fowX = (float) (2 * Math.atan((float)((w / Constants.K[0].get(0,0)) / 2.0f)));
		fowY = (float) (2 * Math.atan((float)((h / Constants.K[0].get(1,1)) / 2.0f)));
	
		System.out.printf("Relocalization Values: fowX %f, fowY %f\n", fowX, fowY);
	}

	
	public float getRefFrameScore(float distanceSquared, float usage) {
		return distanceSquared*KFDistWeight*KFDistWeight
				+ (1-usage)*(1-usage) * KFUsageWeight * KFUsageWeight;
	}
	
	

	public Frame findRePositionCandidate(Frame frame, float maxScore)
	{
	    ArrayList<TrackableKFStruct> potentialReferenceFrames = findEuclideanOverlapFrames(frame, maxScore / (KFDistWeight*KFDistWeight), 0.75f);
	    
		float bestScore = maxScore;
		float bestDist, bestUsage;
		float bestPoseDiscrepancy = 0;
		Frame bestFrame = null;
		SE3 bestRefToFrame = new SE3();
		SE3 bestRefToFrame_tracked = new SE3();
	
		int checkedSecondary = 0;
		for(int i=0 ; i<potentialReferenceFrames.size() ; i++)
		{
			if(frame.getTrackingParent() == potentialReferenceFrames.get(i).ref)
				continue;
	
			if(potentialReferenceFrames.get(i).ref.idxInKeyframes < Constants.INITIALIZATION_PHASE_COUNT)
				continue;
	
			tracker.checkPermaRefOverlap(potentialReferenceFrames.get(i).ref, potentialReferenceFrames.get(i).refToFrame);
			
			float score = getRefFrameScore(potentialReferenceFrames.get(i).dist, tracker.pointUsage);
	
			if(score < maxScore)
			{
				SE3 RefToFrame_tracked = tracker.trackFrameOnPermaref(potentialReferenceFrames.get(i).ref, frame, potentialReferenceFrames.get(i).refToFrame);
				DenseMatrix dist = RefToFrame_tracked.getTranslationMat().mul(potentialReferenceFrames.get(i).ref.meanIdepth);
	
				float newScore = getRefFrameScore((float) dist.mmul(dist).get(0,0), tracker.pointUsage);
				float poseDiscrepancy = (float) Vec.magnitude((potentialReferenceFrames.get(i).refToFrame.mul(RefToFrame_tracked.inverse())).ln());
				float goodVal = tracker.pointUsage * tracker.lastGoodCount / (tracker.lastGoodCount+tracker.lastBadCount);
				checkedSecondary++;
	
				if(tracker.trackingWasGood && goodVal > relocalizationTH && newScore < bestScore && poseDiscrepancy < 0.2)
				{
					bestPoseDiscrepancy = poseDiscrepancy;
					bestScore = score;
					bestFrame = potentialReferenceFrames.get(i).ref;
					bestRefToFrame = potentialReferenceFrames.get(i).refToFrame;
					bestRefToFrame_tracked = RefToFrame_tracked;
					bestDist = (float) dist.mmul(dist).get(0, 0);
					bestUsage = tracker.pointUsage;
				}
			}
		}
	
		if(bestFrame != null)
		{
			return bestFrame;
		}
		else
		{
			return null;
		}
	}	
	
	ArrayList<TrackableKFStruct> findEuclideanOverlapFrames(Frame frame, float distanceTH, float angleTH)
	{
		return findEuclideanOverlapFrames(frame, distanceTH, angleTH, false);
	}
	ArrayList<TrackableKFStruct> findEuclideanOverlapFrames(Frame frame, float distanceTH, float angleTH, boolean checkBothScales)
	{
		// basically the maximal angle-difference in viewing direction is angleTH*(average FoV).
		// e.g. if the FoV is 130°, then it is angleTH*130°.
		float cosAngleTH = (float) Math.cos(angleTH*0.5f*(fowX + fowY));
	
	
		DenseMatrix pos = frame.getScaledCamToWorld().getTranslationMat();
		DenseMatrix viewingDir = frame.getScaledCamToWorld().getRotationMat().row(1);//TODO check which row.
	
		ArrayList<TrackableKFStruct> potentialReferenceFrames = new ArrayList<TrackableKFStruct>();
	
		float distFacReciprocal = 1;
		if(checkBothScales)
			distFacReciprocal = (float) (frame.meanIdepth / frame.getScaledCamToWorld().getScale());
	
		// for each frame, calculate the rough score, consisting of pose, scale and angle overlap.
		for(int i=0;i<graph.keyframesAll.size();i++)
		{
			DenseMatrix otherPos = graph.keyframesAll.get(i).getScaledCamToWorld().getTranslationMat();
	
			// get distance between the frames, scaled to fit the potential reference frame.
			float distFac = (float) (graph.keyframesAll.get(i).meanIdepth / graph.keyframesAll.get(i).getScaledCamToWorld().getScale());
			if(checkBothScales && distFacReciprocal < distFac)
				distFac = distFacReciprocal;
			DenseMatrix dist = (pos.sub(otherPos)).mul(distFac);
			float dNorm2 = (float) dist.mmul(dist.t()).get(0, 0);
			if(dNorm2 > distanceTH)
				continue;
	
			DenseMatrix otherViewingDir = graph.keyframesAll.get(i).getScaledCamToWorld().getRotationMat().row(1);
			float dirDotProd = (float) otherViewingDir.mmul(viewingDir.t()).get(0, 0);
			if(dirDotProd < cosAngleTH)
				continue;
	
			TrackableKFStruct tkfs = new TrackableKFStruct();
			tkfs.ref = graph.keyframesAll.get(i);
			tkfs.refToFrame = (graph.keyframesAll.get(i).getScaledCamToWorld().inverse().mul(frame.getScaledCamToWorld())).inverse().getSE3();
			tkfs.dist = dNorm2;
			tkfs.angle = dirDotProd;
		}
	
		return potentialReferenceFrames;
	}
}
